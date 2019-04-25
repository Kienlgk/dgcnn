"""
    Generated data format will be like: 000001.npy in folder data and label
"""
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import os
import sys
from shutil import rmtree, copyfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider

def save_npy(batches):
    pass

def main(args):
    data_dir = os.path.expanduser(args.data_dir)
    out_dir = os.path.abspath(data_dir) + "_npy"
    out_data_dir = os.path.join(out_dir, "data")
    out_label_dir = os.path.join(out_dir, "label")
    
    if not os.path.exists(out_data_dir):
        pass
        # os.
    ALL_FILES = provider.getDataFiles(os.path.join(data_dir, 'all_files.txt'))
    ALL_FILES = provider.getDataFilesPath(ALL_FILES, data_dir)
    room_filelist = [line.rstrip() for line in open(os.path.join(data_dir, 'room_filelist.txt'))]

    data_batch_list = []
    label_batch_list = []

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_data_dir):
        os.mkdir(out_data_dir)
    if not os.path.exists(out_label_dir):
        os.mkdir(out_label_dir)

    pointcloud_indices = 0
    for file_idx, h5_filename in enumerate(ALL_FILES):
        data_batch, label_batch = provider.loadDataFile(h5_filename)
        h5_batch_size = int(data_batch.shape[0])
        for batch_indices, batch_points in enumerate(data_batch):
            filename = str(pointcloud_indices+batch_indices).zfill(6)
            np.save(os.path.join(out_data_dir, filename), batch_points)
            np.save(os.path.join(out_label_dir, filename), label_batch[batch_indices])
        pointcloud_indices += h5_batch_size


def arguments_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Dir contains h5 file')
    # parser.add_argument('--out_dir', help='Where npy is going to be saved to')
    return parser.parse_args()

if __name__ == "__main__":
    main(arguments_parser(sys.argv[1:]))
