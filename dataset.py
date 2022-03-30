import os
import warnings

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston, load_diabetes

from datafold.pcfold import TSCDataFrame

from Parametric.Frechet.KF_parametric_frechet import KernelFlowsP

kernel_data_prefix = '../data/kernel'
dyn_sys_prefix = '../data/dyn_sys'


def generate_dym_sys_path(data_name, dyn_sys_parameter_dict):
    """
    Helper for generating the path based on name and parameters of the dynamical system.

    :param data_name:                   str, Name of the dataset on which Kernel Flows will be trained.
    :param dyn_sys_parameter_dict:      dict, Dictionary of parameters of the dynamical system.
    :return:                            str, Path for the dataset
    """
    data_path = data_name + '_' + '_'.join(
        [str(key) + ':' + str(value) for key, value in dyn_sys_parameter_dict.items()])
    data_path = os.path.join(dyn_sys_prefix, data_path + '.csv')
    return data_path


def load(data_name, dyn_sys_parameter_dict):
    """
    Wrapper for loading a dataset by name and parameters of the dynamical system.

    :param data_name:                   str, Name of the dataset
    :param dyn_sys_parameter_dict:      dict, Dictionary of parameters of the dynamical system
    :return:                            TSCDataFrame, Dataset to be loaded
    """
    # Get the path of the dataset
    data_path = generate_dym_sys_path(data_name, dyn_sys_parameter_dict)

    # If the dataset has already been generated, load it, otherwise generate the dataset
    if os.path.exists(data_path):
        data = TSCDataFrame.from_csv(data_path)
    else:
        data = generate_dataset(data_name, dyn_sys_parameter_dict)

    return data


def generate_dataset(data_name, dyn_sys_parameter_dict):
    """
    Further wrapper to distinguish between the simple case of Himmelblau's function and other data sets.

    :param data_name:                   str, Name of the dataset
    :param dyn_sys_parameter_dict:      dict, Dictionary of parameters of the dynamical system
    :return:                            TSCDateFrame, Newly generated dataset
    """
    if data_name == 'himmelblau':
        data = generate_himmelblau(**dyn_sys_parameter_dict)
    else:
        data = generate_kernel_flows_data(data_name, dyn_sys_parameter_dict)

    return data


def generate_himmelblau(dimensions, num_points_per_dim):
    """
    Method for generating the himmelblau dataset, which contains gradient descent data on Himmelblau's function.

    :param dimensions:          [(lower_bound, upper_bound)], List of bounds for each dimension
    :param num_points_per_dim:  int, Number of points for each dimension
    :return:                    TSCDataFrame
    """
    # Helper function for calculating one gradient descent step
    def calc_step(y, alpha=1e-3):
        grad = np.array((
            2 * (y[0] ** 2 + y[1] - 11) * 2 * y[0] + 2 * (y[0] + y[1] ** 2 - 7),
            2 * (y[0] ** 2 + y[1] - 11) + 2 * (y[0] + y[1] ** 2 - 7) * 2 * y[1]
        ))
        return y - alpha * grad

    # Generating relevant base points, making a gradient step, and formatting them into a TSCDataFrame
    d_c = generate_sample_points(dimensions, num_points_per_dim)
    df_list = []
    for point in d_c:
        result_df = pd.DataFrame(columns=['x1', 'x2'], dtype=float)
        result_df.loc[0, :] = point
        result_df.loc[1, :] = calc_step(point)
        df_list.append(result_df)
    frame = TSCDataFrame.from_frame_list(df_list)

    # Saving the dataset
    os.makedirs(dyn_sys_prefix, exist_ok=True)
    frame.to_csv(generate_dym_sys_path('himmelblau', {'dimensions': dimensions, 'num_points_per_dim': num_points_per_dim}))

    return frame


def load_d_kernel(data_name):
    """
    Loads the dataset for training Kernel Flows based on name.

    :param data_name:   str, Name of the dataset.
    :return:            (numpy.array, numpy.array), X and y values of the loaded labeled dataset.
    """
    if data_name == 'boston':
        warnings.filterwarnings('ignore')
        X, y = load_boston(return_X_y=True)
    elif data_name == 'diabetes':
        X, y = load_diabetes(return_X_y=True)
    elif data_name.startswith('gaussian'):
        sigmas = data_name.split(':')[1].split(',')
        X, y = generate_gaussian_data(*list(map(float, sigmas)))
    else:
        X, y = None, None
    return X, y


def generate_gaussian_data(sigma_1=1.5, sigma_2=0.9):
    """
    Helper to generate Gaussian Kernel data according to Darcy.

    :param sigma_1:     float, standard deviation of the first Gaussian Kernel
    :param sigma_2:     float, standard deviation of the second Gaussian Kernel
    :return:            (numpy.array, numpy.array), X and y values of the loaded labeled dataset.
    """
    X = np.random.uniform(-10, 10, (100, 1))
    y = np.exp(-np.linalg.norm(X, axis=1) ** 2 / (2 * sigma_1 ** 2)) + np.exp(-np.linalg.norm(X, axis=1) ** 2 / (2 * sigma_2 ** 2))
    return X, y


def generate_kernel_flows_data(data_name, dyn_sys_parameter_dict):
    """
    Loads a dataset and trains Kernel Flows with the parameters provided for the dynamical system.

    :param data_name:                   str, Name of the trainings dataset
    :param dyn_sys_parameter_dict:      dict, Dictionary containing the parameters for the dynamical system
    :return:                            TSCDataFrame, Parameter data of Kernel Flows
    """
    # First, loading training data
    X, y = load_d_kernel(data_name)

    # Using SGD as default optimizer
    if 'optimizer' in dyn_sys_parameter_dict:
        optimizer = dyn_sys_parameter_dict['optimizer']
    else:
        optimizer = 'SGD'

    # Extracting some information from the parameter dictionary for convenience
    dimensions = dyn_sys_parameter_dict['dimensions']
    num_points_per_dim = dyn_sys_parameter_dict['num_points_per_dim']

    # Generate the state space of the dynamical system, which is also the parameter space of Kernel Flows
    d_c = generate_sample_points(dimensions, num_points_per_dim)

    # Training Kernel Flows for each set of parameters for a single step and storing the results in a TSCDataFrame
    df_list = []
    for point in d_c:
        result_df = pd.DataFrame(columns=dyn_sys_parameter_dict['kf_paras'], dtype=float)
        result_df.loc[0, :] = point
        processed_point = np.stack([point, [1, 1]])
        K = KernelFlowsP('gaussian multi', processed_point)
        result_paras = K.fit(X, y, 1, optimizer=optimizer, batch_size=64)
        result_df.loc[1, :] = result_paras[0]
        df_list.append(result_df)
    frame = TSCDataFrame.from_frame_list(df_list)

    # Saving the generated dataset
    os.makedirs(dyn_sys_prefix, exist_ok=True)
    frame.to_csv(
        generate_dym_sys_path(data_name, dyn_sys_parameter_dict))
    
    return frame


def generate_sample_points(set_of_interest_dimension_bounds, num_points_per_dim):
    linear_spaces = np.apply_along_axis(lambda dim_bounds: np.linspace(*dim_bounds, num_points_per_dim), 1,
                                        set_of_interest_dimension_bounds)
    mesh = np.meshgrid(*linear_spaces)
    point_list = list(zip(*(x.flat for x in mesh)))
    return point_list
