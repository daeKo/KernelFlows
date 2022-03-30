from analysis import run_experiment

if __name__ == '__main__':
    if False:
        data_name = 'himmelblau'
        dyn_sys_dict = {'dimensions': [[-4, 4], [-4, 4]], 'num_points_per_dim': 71, 'kf_paras': ['x1', 'x2']}
        run_experiment(data_name, dyn_sys_dict)
    if True:
        data_name = 'gaussian:0.9,1.5'
        dyn_sys_dict = {'dimensions': [[0.1, 3], [0.1, 3]], 'num_points_per_dim': 71, 'kf_paras': ['sigma_1', 'sigma_2']}
        run_experiment(data_name, dyn_sys_dict)
    if True:
        data_name = 'gaussian:2,3'
        dyn_sys_dict = {'dimensions': [[0.1, 6], [0.1, 6]], 'num_points_per_dim': 71, 'kf_paras': ['sigma_1', 'sigma_2']}
        run_experiment(data_name, dyn_sys_dict)
    if True:
        data_name = 'gaussian:2,3'
        dyn_sys_dict = {'dimensions': [[0.1, 6], [0.1, 6]], 'num_points_per_dim': 71, 'kf_paras': ['sigma_1', 'sigma_2'], 'optimizer': 'Nesterov'}
        run_experiment(data_name, dyn_sys_dict)
    if True:
        data_name = 'boston'
        dyn_sys_dict = {'dimensions': [[1, 1000], [1, 1000]], 'num_points_per_dim': 71, 'kf_paras': ['sigma_1', 'sigma_2']}
        run_experiment(data_name, dyn_sys_dict)
    if True:
        data_name = 'diabetes'
        dyn_sys_dict = {'dimensions': [[1, 1000], [1, 1000]], 'num_points_per_dim': 71, 'kf_paras': ['sigma_1', 'sigma_2']}
        run_experiment(data_name, dyn_sys_dict)