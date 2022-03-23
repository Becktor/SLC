import torch
import glob
from tqdm import tqdm
import random
import os
from pathlib import Path


def main():
    old = r"Q:\classification\IROS_DS\percent_test\train_set_5\*\*.jpg"
    oldies = glob.glob(old)
    for val in tqdm(oldies):
        r = random.random()
        if r > 0.05:
            os.remove(val)


def remDup():
    val = r"Q:\classification\IROS_DS\val_set\*\*.jpg"
    val = glob.glob(val)
    val_ids = [Path(x).name for x in val]

    train = r"Q:\classification\IROS_DS\train_set\*\*.jpg"
    train = glob.glob(train)

    for t in train:
        t_id = Path(t).name
        if t_id in val_ids:
            os.remove(t)


if __name__ == "__main__":
    main()
