import os

import numpy as np

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from dataset import generate_sample_points

plt_prefix = '../images'


def plot_basins(data_name, dyn_sys_parameter_dict, labels):
    """
    Helper for plotting the basins found by the clustering algorithm. This assumes the state space is two-dimensional.

    :param data_name:                   str, Name of the dataset
    :param dyn_sys_parameter_dict:      dict, Dictionary defining the dynamical system
    :param labels:                      numpy.array(shape=(num_datapoints,)), Cluster labels found by clustering.
    :return:                            None
    """
    # Calculating and reformatting the state space of the dynamical system
    dimensions = dyn_sys_parameter_dict['dimensions']
    datapoints = generate_sample_points(dimensions, dyn_sys_parameter_dict['num_points_per_dim'])
    x, y = tuple(zip(*datapoints))

    # Defining the coloring scheme for the basins
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # Setting the bounds of the plot
    plt.xlim(dimensions[0])
    plt.ylim(dimensions[1])

    # Labeling axes
    axis_labels = dyn_sys_parameter_dict['kf_paras']
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])

    # Plotting cluster labels
    int_label_list = labels.astype(int).tolist()
    plt.scatter(x=x, y=y, c=int_label_list, cmap=cmap)

    # Saving the plot
    os.makedirs(plt_prefix, exist_ok=True)
    plt_path = os.path.join(plt_prefix, 'plot_'+ data_name + '_' + '_'.join(
        [str(key) + ':' + str(value) for key, value in dyn_sys_parameter_dict.items()]) + '_labels' + str(int(np.max(labels)+1)) + '.png')
    plt_path = plt_path.replace('\'', '')

    plt.savefig(plt_path)
