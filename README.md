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
  │   ├── **csv_converter.py**
  │   ├── train2014.zip
  │   ├── val2014.zip
  │   ├── train2014/
  │   ├── val2014/
  │   ├── MSCOCO_train_val_Korean.json
  │   ├── COCO_train.csv
  │   └── COCO_val.csv
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

## Inference
- You can refer to `infer.ipynb`.
