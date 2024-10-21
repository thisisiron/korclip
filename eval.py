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


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item() for k in topk]


def eval(model, text_inputs, test_loader, device):
    model.eval()

    total_correct, step = 0., 0.
    top1, top5, n = 0., 0., 0.
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(test_loader)):
            images, labels = images.to(device), labels.to(device)

            output = model(pixel_values=images, **text_inputs)

            acc1, acc5 = accuracy(output.logits_per_image, labels, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100 

        print(f"Top-1 accuracy: {top1:.4f}")
        print(f"Top-5 accuracy: {top5:.4f}")
    return top1, top5
             

def load_data(args, eval_data='CIFAR10'):
    if args.dataset == 'CIFAR10':
        test_transform = transforms.Compose([
            transforms.Resize(args.input_resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.input_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        class_mapping = dict(zip(coco_classes, kor_coco_classes))

    elif args.dataset == 'CIFAR100':
        test_transform = transforms.Compose([
            transforms.Resize(args.input_resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.input_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        class_mapping = dict(zip(coco_classes, kor_coco_classes))

    return test_data, test_loader, class_mapping


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = AutoTokenizer.from_pretrained("koclip/koclip-base-pt")
    # model = AutoModel.from_pretrained("koclip/koclip-base-pt").to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    args.input_resolution = 224

    test_data, test_loader, class_mapping = load_data(args)

    text_inputs = tokenizer(
            [f"{class_mapping[c]}의 사진" for c in test_data.classes],
            return_tensors="pt",
            padding=True
    ).to(device)

    eval(model, text_inputs, test_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='output/checkpoint')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--checkpoints', type=str, default=None)

    args = parser.parse_args()
    main(args)
