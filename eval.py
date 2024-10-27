import argparse
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from transformers import AutoProcessor, AutoModel, AutoTokenizer

from const import *


templates = [
    "{}의 사진.",
    "흐릿한 {}의 사진.",
    "종이접기 {}.",
    "큰 {}의 사진.",
    "비디오 게임 속 {}.",
    "예술적인 {}.",
    "작은 {}의 사진."
]


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item() for k in topk]


def evaluation(model, text_features, test_loader, device):
    model.eval()

    total_correct, step = 0., 0.
    top1, top5, n = 0., 0., 0.
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(test_loader)):
            images, labels = images.to(device), labels.to(device)

            image_features = model.get_image_features(pixel_values=images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ text_features 

            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100 

        print(f"Top-1 accuracy: {top1:.4f}")
        print(f"Top-5 accuracy: {top5:.4f}")
    return top1, top5
             

def load_data(dataset_name, batch_size=32, image_size=224, num_workers=2):
    if dataset_name.lower() == "cifar10":
        test_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        classes = KOR_COCO10_CLSSES

    elif dataset_name.lower() == "cifar100":
        test_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = torchvision.datasets.CIFAR100(root="./data", train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        classes = KOR_COCO100_CLASSES

    elif dataset_name.lower() == "imagenet":
        try:
            from imagenetv2_pytorch import ImageNetV2Dataset
        except ImportError:
            raise ImportError(
                "The library is not installed. Please install it by running:\n"
                "pip install git+https://github.com/modestyachts/ImageNetV2_pytorch"
            )
        test_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        images = ImageNetV2Dataset(transform=test_transform, location="./data")
        test_loader = DataLoader(images, batch_size=batch_size, num_workers=num_workers)
        classes = KOR_IMAGENET_CLASSES

    return test_loader, classes 


def get_text_features(model, tokenizer, classes, device):
    with torch.no_grad():
        text_features = []
        for classname in tqdm(classes):
            texts = [template.format(classname) for template in templates] #format with class
            text_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True
            ).to(device)

            class_embeddings = model.get_text_features(**text_inputs)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)
    return text_features 


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    test_loader, classes = load_data(
        args.dataset, 
        args.batch_size,
        model.config.vision_config.image_size
    )

    text_features = get_text_features(model, tokenizer, classes, device)

    evaluation(model, text_features, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="thisisiron/korclip-vit-base-patch32")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()
    main(args)
