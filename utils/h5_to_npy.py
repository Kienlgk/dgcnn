import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import os
import sys
from shutil import rmtree
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider

def save_npy(batches):
    pass

def main(args):
    data_dir = args.data_dir
    out_dir = os.path.abspath(data_dir) + "_npy"
    out_data_dir = os.path.join(out_dir, "data")
    out_label_dir = os.path.join(out_dir, "label")
    if not os.path.exists(out_data_dir):
        pass
        # os.
    ALL_FILES = provider.getDataFiles('indoor3d_sem_seg_hdf5_data/all_files.txt') 
    room_filelist = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

    data_batch_list = []
    label_batch_list = []

    for file_idx, h5_filename in enumerate(ALL_FILES):
        data_batch, label_batch = provider.loadDataFile(h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)

    if os.path.join(out_dir, "data")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Dir contains h5 file')
    parser.add_argument('--out_dir', help='Where npy is going to be saved to')
    FLAGS = parser.parse_args()
    main(FLAGS)
