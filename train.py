#!/usr/bin/env python
# coding=utf-8

"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

For more information, refer to: https://github.com/huggingface/transformers/blob/main/examples/pytorch/contrastive-image-text/run_clip.py
"""

import os
import time
import json
import logging
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from tqdm import tqdm
import datetime

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import transformers
from transformers import (
    VisionTextDualEncoderModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from dataset import ImageTextDataset
from const import *


logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or train from scratch.
    """

    text_model_name_or_path: str = field(
        metadata={
            "help": "The text model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    vision_model_name_or_path: str = field(
        metadata={
            "help": "The vision model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    from_pt: bool = field(
        default=True,
        metadata={"help": "Whether to load the text and vision model using PyTorch checkpoints."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizers (backed by the tokenizers library) or not."},
    )
    mixed_precision: Optional[str] = field(
        default=None,
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained."
            "Choose one of `[no, fp16, bf16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file (a jsonlines file)."}
    )
    max_seq_length: Optional[int] = field(
        default=72,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated; sequences shorter will be padded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8, metadata={"help": "The number of processes to use for the preprocessing."}
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class CustomTrainingArguments(TrainingArguments):
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )


def create_learning_rate_fn(
    total_train_steps: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], float]:
    """Returns a linear warmup, linear decay learning rate function."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        return max(
            0.0, float(total_train_steps - current_step) / max(1, total_train_steps - num_warmup_steps)
        )

    return lr_lambda


def collate_fn(examples, tokenizer, max_seq_length):
    examples = list(filter(lambda x: x is not None, examples))
    pixel_values = torch.stack([example[0] for example in examples])
    captions = [example[1] for example in examples]

    inputs = tokenizer(
        captions,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    batch = {
        "pixel_values": pixel_values,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    return batch


def collate_fn_eval(batch, tokenizer, classes, class_mapping):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    class_inputs = tokenizer(
            [f"{class_mapping[c]}의 사진" for c in classes],
            return_tensors="pt",
            padding=True
    )
    return images, labels, class_inputs


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=model_args.mixed_precision,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    training_args.output_dir = os.path.join(training_args.output_dir, timestamp)
    os.makedirs(training_args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.text_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.text_model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.save_pretrained(training_args.output_dir)

    # Load model
    model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        model_args.vision_model_name_or_path,
        model_args.text_model_name_or_path,
    )

    # Move model to GPU if available
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    device = accelerator.device 

    model.to(device, dtype=weight_dtype)

    # Initialize transforms
    logger.info(f"Resize image ({model.config.vision_config.image_size})")
    image_transform = T.Compose([
        T.Resize([model.config.vision_config.image_size], interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(model.config.vision_config.image_size),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    # Initialize the image-text datasets
    train_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.train_file,
        transform=image_transform,
    )

    eval_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.validation_file,
        transform=image_transform,
    )

    # Store some constants
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size)
    eval_batch_size = int(training_args.per_device_eval_batch_size)
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    collate_fn_with_args = partial(collate_fn, tokenizer=tokenizer, max_seq_length=data_args.max_seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=data_args.preprocessing_num_workers,
        drop_last=True,
        collate_fn=collate_fn_with_args,
        shuffle=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=data_args.preprocessing_num_workers,
        collate_fn=collate_fn_with_args,
        shuffle=False,
    )

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=create_learning_rate_fn(
            total_train_steps, training_args.warmup_steps, training_args.learning_rate
        ),
    )

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    global_step = 0
    train_time = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not accelerator.is_local_main_process)

        for batch in train_progress_bar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text

                ground_truth = torch.arange(len(logits_per_image)).to(device)

                loss_i = F.cross_entropy(logits_per_image, ground_truth)
                loss_t = F.cross_entropy(logits_per_text, ground_truth)
                loss = (loss_i + loss_t) / 2

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

                epoch_loss += loss.item()
                train_progress_bar.set_postfix({"loss": loss.item()})
                global_step += 1

        train_time += time.time() - start_time
        avg_train_loss = epoch_loss / len(train_loader)

        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {avg_train_loss:.4f}")

        # Evaluation
        # model.eval()
        # eval_loss = 0.0
        # eval_progress_bar = tqdm(eval_loader, desc="Evaluating", disable=not accelerator.is_local_main_process)
        # with torch.no_grad():
        #     for batch in eval_progress_bar:
        #         input_ids = batch["input_ids"].to(device)
        #         attention_mask = batch["attention_mask"].to(device)
        #         pixel_values = batch["pixel_values"].to(device)

        #         outputs = model(
        #             input_ids=input_ids,
        #             pixel_values=pixel_values,
        #             attention_mask=attention_mask,
        #             return_dict=True,
        #         )

        #         logits_per_image = outputs.logits_per_image
        #         logits_per_text = outputs.logits_per_text

        #         ground_truth = torch.arange(len(logits_per_image)).to(device)

        #         loss_i = F.cross_entropy(logits_per_image, ground_truth)
        #         loss_t = F.cross_entropy(logits_per_text, ground_truth)
        #         loss = (loss_i + loss_t) / 2

        #         eval_loss += loss.item()

        # avg_eval_loss = eval_loss / len(eval_loader)
        # logger.info(f"Epoch {epoch + 1}/{num_epochs} - Eval loss: {avg_eval_loss:.4f}")

        # Save checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f"{training_args.output_dir}", save_function=accelerator.save)
        logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()

