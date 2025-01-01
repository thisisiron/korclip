# KorCLIP

KorCLIP is a project that implements a Korean version of the CLIP (Contrastive Language–Image Pretraining) model. 

## Usage

```python
import io
import requests
from PIL import Image
import torch
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

MODEL_PATH = "thisisiron/korclip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)

image = Image.open(io.BytesIO(requests.get("http://images.cocodataset.org/val2014/COCO_val2014_000000537955.jpg").content))
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = preprocess(image).unsqueeze(0).to(device)
text = tokenizer(["강아지", "고양이", "거북이"], return_tensors="pt").to(device)

with torch.no_grad():
    image_features = model.get_image_features(image)
    text_features = model.get_text_features(**text)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

## Dataset

### COCO 2014 
- Image download
  ```
  cd data
  ./download.sh
  ```
- Korean annotation download [[link]](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=261)
  - You need the `MSCOCO_train_val_Korean.json` file.
- Preprocessing (json -> csv)
  ```
  python csv_converter.py
  ```
- Diretory structure
  ```
  ├── data
  │   ├── **download.sh**
  │   └── **download.sh**
  │       ├── **download.sh**
  │       ├── **csv_converter.py**
  │       ├── train2014.zip
  │       ├── val2014.zip
  │       ├── train2014/
  │       ├── val2014/
  │       ├── MSCOCO_train_val_Korean.json
  │       ├── COCO_train.csv
  │       └── COCO_val.csv
  ```

## Training
- Single-GPU
    ```
    ./run.sh 1
    ```
- Multi-GPU
  ```
  ./run.sh NUM_GPU
  ```

## Evaluation (Zero-Shot Prediction)
- I am currently evaluating using only one template. I plan to add additional datasets and templates for future evaluations.
- The following metric is the results of training on the "COCO2014" Korean dataset only.
```
python eval.py
```

| Dataset | Acc@1 | Acc@5 |
|---|---|---|
|CIFAR10| 61.99 | 93.82 |

## Inference
- You can refer to `infer.ipynb`.
