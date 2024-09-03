# Human Mobility Prediction Challenge: Next Location Prediction using Spatiotemporal BERT

## Overview

This repository contains an **unofficial** PyTorch implementation of the ["Human Mobility Prediction Challenge: Next Location Prediction using Spatiotemporal BERT"](https://dl.acm.org/doi/10.1145/3615894.3628498) method, as part of the [HuMob Challenge 2023](https://connection.mit.edu/humob-challenge-2023).

## Setup

```bash
pip install -r requirements.txt
```

## Run

1. Prepare data

Download the data from [here](https://zenodo.org/records/10142719) and place it in the `data` directory.

2. Train

```bash
# 本番
$ python train_taskb_prod.py --batch_size 128 --epochs 1 --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
$ python train_taskc_prod.py --batch_size 128 --epochs 1 --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
$ python train_taskd_prod.py --batch_size 128 --epochs 1 --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}

# 精度評価用
$ python train_taskb_test.py --batch_size 128 --epochs 1 --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
$ python train_taskc_test.py --batch_size 128 --epochs 1 --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
$ python train_taskd_test.py --batch_size 128 --epochs 1 --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
```

3. Predict

Here, `${PTH_FILE_PATH}` refers to the path where the PTH file obtained after training the corresponding task.

```bash
# file_nameの設定
$ export file_name=batchsize{batch_sizeの設定値}_epochs{epochsの設定値}_embedsize{embed_sizeの設定値}_layersnum{layers_numの設定値}_headsnum{heads_numの設定値}_cuda{cudaの設定値}_lr{lrの設定値}_seed{seedの設定値}

# 本番
$ python val_taskb_prod.py --file_name ${file_name} --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
$ python val_taskc_prod.py --file_name ${file_name} --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
$ python val_taskd_prod.py --file_name ${file_name} --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}

# 精度評価用
$ python val_taskb_test.py --file_name ${file_name} --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
$ python val_taskc_test.py --file_name ${file_name} --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
$ python val_taskd_test.py --file_name ${file_name} --embed_size 128 --layers_num 4 --heads_num 8 --run_name {hoge}
```

## License

This project is licensed under the [MIT License](https://github.com/caoji2001/Human-Mobility-Prediction-Challenge-Next-Location-Prediction-using-Spatiotemporal-BERT/blob/main/LICENSE).

## Citations

```bibtex
@inproceedings{10.1145/3615894.3628498,
author = {Terashima, Haru and Tamura, Naoki and Shoji, Kazuyuki and Katayama, Shin and Urano, Kenta and Yonezawa, Takuro and Kawaguchi, Nobuo},
title = {Human Mobility Prediction Challenge: Next Location Prediction Using Spatiotemporal BERT},
year = {2023},
isbn = {9798400703560},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3615894.3628498},
doi = {10.1145/3615894.3628498},
abstract = {Understanding, modeling, and predicting human mobility patterns in urban areas has become a crucial task from the perspectives of traffic modeling, disaster risk management, urban planning, and more. HuMob Challenge 2023 aims to predict future movement trajectories based on the past movement trajectories of 100,000 users[1]. Our team, "uclab2023", considered that model design significantly impacts training and prediction times in the task of human mobility trajectory prediction. To address this, we proposed a model based on BERT, commonly used in natural language processing, which allows parallel predictions, thus reducing both training and prediction times.In this challenge, Task 1 involves predicting the 15-day daily mobility trajectories of target users using the movement trajectories of 100,000 users. Task 2 focuses on predicting the 15-day emergency mobility trajectories of target users with data from 25,000 users. Our team achieved accuracy scores of GEOBLEU: 0.3440 and DTW: 29.9633 for Task 1 and GEOBLEU: 0.2239 and DTW: 44.7742 for Task 2[2][3].},
booktitle = {Proceedings of the 1st International Workshop on the Human Mobility Prediction Challenge},
pages = {1–6},
numpages = {6},
keywords = {transformer, human mobility, next location prediction},
location = {Hamburg, Germany},
series = {HuMob-Challenge '23}
}
```
