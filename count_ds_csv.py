import numpy as np
from PIL import Image
import csv
import os
from tqdm import tqdm



with open(r'Q:\newData\dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    tqdm_rows = tqdm(iter(csv_reader))
    update_dict = {}
    for x in tqdm_rows:
        print(x)
        nm = x[0]
        if nm in update_dict:
            update_dict[nm]+=1
        else:
            update_dict[nm]=1
    print(update_dict)
        # lbl = matching_dict[int(tmp_val[0][-1])]
        # from_epoch = int(tmp_val[2])
        # img_path = tmp_val[3].replace('\'', '').replace('\\\\', '\\')[1:-1]
        # update_dict[img_path] = lbl