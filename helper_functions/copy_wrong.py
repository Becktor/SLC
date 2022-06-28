# Python 3 program to visualize 4th image
import os.path
import shutil
import matplotlib.pyplot as plt
import numpy as np

def read_lines(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines


files = r'Q:\git\SLC\wrong_train.txt'


path = r'Q:\uncert_data\moved_images_train'

l = read_lines(files)
for e in l:
    a, b, c = e.replace('\n','').split(',')
    # os.remove(a)
    elm = a.split('\\')
    id_path = os.path.join(path, b)
    ep = os.path.join(id_path, elm[-1])
    if os.path.exists(id_path):
        shutil.copy(a, ep)
    else:
        os.makedirs(id_path)
        shutil.copy(a, ep)
