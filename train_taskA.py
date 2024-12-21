import argparse
import datetime
import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset import *
from eval import caluc_test_score
from model import *


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    d = [item["d"] for item in batch]
    t = [item["t"] for item in batch]
    input_x = [item["input_x"] for item in batch]
    input_y = [item["input_y"] for item in batch]
    time_delta = [item["time_delta"] for item in batch]
    label_x = [item["label_x"] for item in batch]
    label_y = [item["label_y"] for item in batch]
    len_tensor = torch.tensor([item["len"] for item in batch])

    d_padded = pad_sequence(d, batch_first=True, padding_value=0)
    t_padded = pad_sequence(t, batch_first=True, padding_value=0)
    input_x_padded = pad_sequence(input_x, batch_first=True, padding_value=0)
    input_y_padded = pad_sequence(input_y, batch_first=True, padding_value=0)
    time_delta_padded = pad_sequence(time_delta, batch_first=True, padding_value=0)
    label_x_padded = pad_sequence(label_x, batch_first=True, padding_value=0)
    label_y_padded = pad_sequence(label_y, batch_first=True, padding_value=0)

    return {
        "d": d_padded,
        "t": t_padded,
        "input_x": input_x_padded,
        "input_y": input_y_padded,
        "time_delta": time_delta_padded,
        "label_x": label_x_padded,
        "label_y": label_y_padded,
        "len": len_tensor,
    }


def compute_validation_metrics(args, load_model_path):
    # 評価データでGeoBlue, DTWを算出
    dataset_val = HuMobDatasetTaskAVal(
        "/kaggle/s3storage/01_public/humob-challenge-2024/input/cityA_groundtruthdata.csv.gz", True
    )
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=2)
    device = torch.device(f"cuda:{args.cuda}")
    load_model = LPBERT(args.layers_num, args.heads_num, args.embed_size).to(device)
    load_model.load_state_dict(torch.load(load_model_path, map_location=device))

    result = dict()
    result["generated"] = []
    result["reference"] = []

    load_model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader_val):
            data["d"] = data["d"].to(device)
            data["t"] = data["t"].to(device)
            data["input_x"] = data["input_x"].to(device)
            data["input_y"] = data["input_y"].to(device)
            data["time_delta"] = data["time_delta"].to(device)
            data["label_x"] = data["label_x"].to(device)
            data["label_y"] = data["label_y"].to(device)
            data["len"] = data["len"].to(device)

            output = load_model(data["d"], data["t"], data["input_x"], data["input_y"], data["time_delta"], data["len"])
            label = torch.stack((data["label_x"], data["label_y"]), dim=-1)

            assert torch.all((data["input_x"] == 201) == (data["input_y"] == 201))
            pred_mask = data["input_x"] == 201
            output = output[pred_mask]
            pred = []
            pre_x, pre_y = -1, -1
            for step in range(len(output)):
                if step > 0:
                    output[step][0][pre_x] *= 0.9
                    output[step][1][pre_y] *= 0.9

                pred.append(torch.argmax(output[step], dim=-1))
                pre_x, pre_y = pred[-1][0].item(), pred[-1][1].item()

            pred = torch.stack(pred)
            generated = (
                torch.cat(
                    (data["d"][pred_mask].unsqueeze(-1) - 1, data["t"][pred_mask].unsqueeze(-1) - 1, pred + 1),
                    dim=-1,
                )
                .cpu()
                .tolist()
            )
            generated = [tuple(x) for x in generated]

            reference = (
                torch.cat(
                    (
                        data["d"][pred_mask].unsqueeze(-1) - 1,
                        data["t"][pred_mask].unsqueeze(-1) - 1,
                        label[pred_mask] + 1,
                    ),
                    dim=-1,
                )
                .cpu()
                .tolist()
            )
            reference = [tuple(x) for x in reference]

            result["generated"].append(generated)
            result["reference"].append(reference)

    geobleu_scr, dtw_scr = caluc_test_score(result)
    wandb.log({"valid_geobleu_score": geobleu_scr, "valid_dtw_score": dtw_scr})


def taskA(args):
    name = f"batchsize{args.batch_size}_epochs{args.epochs}_embedsize{args.embed_size}_layersnum{args.layers_num}_headsnum{args.heads_num}_cuda{args.cuda}_lr{args.lr}_seed{args.seed}"
    current_time = datetime.datetime.now()

    # 初期化 wandb
    if args.use_sampling:
        wandb.init(project=f"LPBERT-taskA-sampling", name=name, config=args)
    else:
        wandb.init(project=f"LPBERT-taskA", name=name, config=args)
    wandb.run.name = name  # Set the run name
    wandb.run.save()

    taskA_dataset_train = HuMobDatasetTaskATrain(
        "/kaggle/s3storage/01_public/humob-challenge-2024/input/cityA_groundtruthdata.csv.gz", args.use_sampling
    )
    taskA_dataloader_train = DataLoader(
        taskA_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    device = torch.device(f"cuda:{args.cuda}")
    model = LPBERT(args.layers_num, args.heads_num, args.embed_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch_id in range(args.epochs):
        for batch_id, batch in enumerate(tqdm(taskA_dataloader_train)):
            batch["d"] = batch["d"].to(device)
            batch["t"] = batch["t"].to(device)
            batch["input_x"] = batch["input_x"].to(device)
            batch["input_y"] = batch["input_y"].to(device)
            batch["time_delta"] = batch["time_delta"].to(device)
            batch["label_x"] = batch["label_x"].to(device)
            batch["label_y"] = batch["label_y"].to(device)
            batch["len"] = batch["len"].to(device)

            output = model(
                batch["d"], batch["t"], batch["input_x"], batch["input_y"], batch["time_delta"], batch["len"]
            )
            label = torch.stack((batch["label_x"], batch["label_y"]), dim=-1)

            pred_mask = batch["input_x"] == 201
            pred_mask = torch.cat((pred_mask.unsqueeze(-1), pred_mask.unsqueeze(-1)), dim=-1)

            loss = criterion(output[pred_mask], label[pred_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step = epoch_id * len(taskA_dataloader_train) + batch_id

            # wandb lossの記録
            wandb.log({"loss": loss.detach().item(), "step": step})

        scheduler.step()

        # wandb 各エポック終了時のlossを記録
        wandb.log({"epoch_loss": loss.detach().item(), "epoch": epoch_id})

        # wandb 学習済みモデルの保存
        model_save_path = os.path.join(
            wandb.run.dir, f'model_{current_time.strftime("%Y_%m_%d_%H_%M_%S")}_epoch{epoch_id}.pth'
        )
        torch.save(model.state_dict(), model_save_path)
        wandb.save(model_save_path)

        compute_validation_metrics(args, model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--layers_num", type=int, default=4)
    parser.add_argument("--heads_num", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_sampling", type=bool, default=False)
    args = parser.parse_args()

    set_random_seed(args.seed)

    taskA(args)
