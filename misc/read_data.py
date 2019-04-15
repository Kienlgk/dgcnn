import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from open3d import *
# from open3d import open3d


def plot_3d(_3d_array, normal=None, normalize_scale=False):
    """
    _3d_array is numpy array with shape: (n, 3)
    """
    if normalize_scale:
        _3d_array_norm = normalize_pointcloud(_3d_array)
        _3d_array_trans = np.transpose(_3d_array_norm, [1, 0])
    else:
        _3d_array_trans = np.transpose(_3d_array, [1, 0])
    print("3d_array_trans shape: ", _3d_array_trans.shape)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_line = _3d_array_trans[0]
    y_line = _3d_array_trans[1]
    z_line = _3d_array_trans[2]
    if normal is None:
        ax.scatter3D(x_line, y_line, z_line, c=z_line, cmap='Greens')
    else:
        ax.scatter3D(x_line, y_line, z_line, c=normal, cmap='Greens')
    plt.show()


def plot_pointcloud(pcl_matrix):
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.Vector3dVector(pcl_matrix)
    # open3d.visualization.draw_geometries([pcd])
    pcd = PointCloud()
    pcd.points = Vector3dVector(pcl_matrix)
    draw_geometries([pcd])


def normalize_pointcloud(_3d_array):
    """
    _3d_array is numpy array with shape: (n, 3)
    return: 
        _3d_array_return: a normalize-scaled point` cloud matrix in range [`0, 1], shape=(n, 3) 
    """
    _3d_array_trans = np.transpose(_3d_array, [1, 0])
    denominator = (np.amax(_3d_array_trans, axis=1) - np.amin(_3d_array_trans, axis=1))
    denominator[denominator <= 0] += 0.0001
    _3d_array_return = (_3d_array - np.amin(_3d_array_trans, axis=1))/denominator
    return _3d_array_return

def plot_dist(data):
    data = np.transpose(data)
    fig, tuples = plt.subplots(nrows=3, ncols=3, num=10)
    tuples_list = list(tuples)
    for row_idx, row in enumerate(tuples_list):
        for idx, ax in enumerate(row):
            counts, bins, ignored = ax.hist(np.asarray(data[idx + 3*row_idx]), bins=500, density=True)
            mean = np.mean(data[idx+3*row_idx])
            ax.plot(bins, np.tile(mean, 501), linewidth=1, color='r')
    plt.show()

def main():
    # _file = os.path.join("..", 'data', 'modelnet40_ply_hdf5_2048', 'ply_data_train0.h5')
    _file = os.path.join("..", 'sem_seg', 'indoor3d_sem_seg_hdf5_data', 'ply_data_all_0.h5')
    # ['data', 'faceId', 'label', 'normal']
    file_content = h5py.File(_file)
    data = file_content['data']
    # face_id = file_content['faceId']
    label = file_content['label']
    sample_data = data[1]
    print(sample_data.shape)
    plot_dist(sample_data)
    # sample_data_norm = normalize_pointcloud(sample_data)
    # plot_pointcloud(sample_data)
    # plot_3d(sample_data_norm, normalize_scale=True)


def plot_3d_ex():
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray')

    # Data for three-dimensional scattered points
    zdata = 15 * np.random.random(100)
    xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    plt.show()

main()
# plot_3d_ex()