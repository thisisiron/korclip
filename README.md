# korclip

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
|CIFAR10| 48.32 | 87.24 |

## Inference
- You can refer to `infer.ipynb`.
