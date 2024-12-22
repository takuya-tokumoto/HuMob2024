※作業中

# 概要
[HuMob Challenge 2024](https://wp.nyu.edu/humobchallenge2024/)における開発コード

# 環境構築
[こちら](https://github.com/takuya-tokumoto/EnvHuMOB2024)のrepositoryを参考に準備ください

# 実行手順
1. データ準備

コンペ用の公開データを格納 
今回は`/kaggle/s3storage/01_public/humob-challenge-2024/input/`へ格納

2. Train

```bash
python train_taskA.py --batch_size 128 --epochs 200 --embed_size 128 --layers_num 4 --heads_num 8
```

3. Predict
`${PTH_FILE_PATH}`は、対応するタスクのトレーニング後に得られるPTHファイルのパスを指す。

```bash
python val_taskA.py --pth_file ${PTH_FILE_PATH} --embed_size 128 --layers_num 4 --heads_num 8
```

