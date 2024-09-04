import argparse
import datetime
import glob
import json
import os

import torch
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.dataset_prod import HuMobDatasetTaskBVal


def task(args):

    models_path = f"/kaggle/s3storage/01_public/humob-challenge-2024/models/"
    mode = "prod"  # 本番
    _pth_file = os.path.join(models_path, mode, args.run_name, "checkpoint", "taskB", args.file_name, "*.pth")
    pth_file = glob.glob(_pth_file)[0]

    output_path = f"/kaggle/s3storage/01_public/humob-challenge-2024/output/"
    result_path = os.path.join(output_path, mode, args.run_name, "taskB", args.file_name)
    os.makedirs(result_path, exist_ok=True)

    task_dataset_val = HuMobDatasetTaskBVal(
        "/kaggle/s3storage/01_public/humob-challenge-2024/input/cityB_challengedata.csv.gz"
    )
    task_dataloader_val = DataLoader(task_dataset_val, batch_size=1, num_workers=args.num_workers)

    device = torch.device(f"cuda:{args.cuda}")
    model = LPBERT(args.layers_num, args.heads_num, args.embed_size).to(device)
    model.load_state_dict(torch.load(pth_file, map_location=device))

    result = dict()
    result["generated"] = []
    result["reference"] = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(task_dataloader_val):
            data["d"] = data["d"].to(device)
            data["t"] = data["t"].to(device)
            data["input_x"] = data["input_x"].to(device)
            data["input_y"] = data["input_y"].to(device)
            data["time_delta"] = data["time_delta"].to(device)
            data["label_x"] = data["label_x"].to(device)
            data["label_y"] = data["label_y"].to(device)
            data["len"] = data["len"].to(device)

            output = model(data["d"], data["t"], data["input_x"], data["input_y"], data["time_delta"], data["len"])
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
                    (data["d"][pred_mask].unsqueeze(-1) - 1, data["t"][pred_mask].unsqueeze(-1) - 1, pred + 1), dim=-1
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

    current_time = datetime.datetime.now()
    with open(os.path.join(result_path, f'{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.json'), "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--layers_num", type=int, default=4)
    parser.add_argument("--heads_num", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="init")
    args = parser.parse_args()

    task(args)
