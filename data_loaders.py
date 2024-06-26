from __future__ import print_function

import pandas as pd
import numpy as np
import random
import math
import os

import torch
import torchvision
import torch.utils.data


def get_shapes(dataset):
    """
    Get the input and output shapes of the data examples for each dataset used.
    dataset: name of the dataset used
    """
    if dataset == 'HAR':
        num_inputs = 561 # (MS) Suspicion: This is linked to LR, so they're not properly abstracted
        num_outputs = 6
        num_labels = 6
    elif dataset == 'NWDAF':
        num_inputs = 8
        num_outputs = 8
        num_labels = 2
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels


def load_data(dataset, seed):
    """
    Load the dataset from the drive.
    The har datasets need to be downloaded first with the provided scripts in /data.
    dataset: name of the dataset
    seed: seed for randomness
    """
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    if dataset == 'NWDAF':
        data_folder = "./data/Data v5"
        df = pd.read_csv(os.path.join(data_folder, "amari_ue_data_final_v5_smoothed_scaled.csv"))
        df = df.sort_values(["imeisv", "_time"], ascending = True)
        df['imeisv'] = df['imeisv'].astype(str)
        feature_columns = [
            'dl_bitrate','ul_bitrate', 
            'cell_x_dl_retx', 'cell_x_dl_tx',
            'cell_x_ul_retx', 'cell_x_ul_tx',
            'ul_total_bytes_non_incr', 'dl_total_bytes_non_incr'
        ]
        benign_data_starting_point = "2024-03-20 14:14:50.19"
        benign_data_ending_point = "2024-03-23 16:26:19.00"
        benign_filter_1 = (df['_time'].between(benign_data_starting_point, benign_data_ending_point))
        benign_filter_2 = (~df['imeisv'].isin(['8642840401594200', '8642840401612300','8642840401624200','3557821101183501']))
        benign_filter_3 = (df['label'] == 0)
        benign_data_filter = (benign_filter_1 & benign_filter_2 & benign_filter_3)
        benign_data_train = df[benign_data_filter].copy()
        benign_data_train = benign_data_train.sort_values(['imeisv','_time'])
        benign_data_test_period_start = "2024-03-24 01:20:00.19"
        benign_devices_for_testing = ['8609960468879057', '8628490433231157','8677660403123800']
        benign_filter_4 = (df['_time'] >= benign_data_test_period_start)
        benign_filter_5 = (df['imeisv'].isin(benign_devices_for_testing))
        benign_data_filter_test = (benign_filter_3 & benign_filter_4 & benign_filter_5)
        benign_data_test = df[benign_data_filter_test].copy()
        benign_data_test = benign_data_test.sort_values(['imeisv','_time'])
        #malicious data
        attck_1_start = "2024-03-23 21:26:00"
        attck_1_end = "2024-03-23 22:23:00"
        ues_to_exclude_in_1st_attck = [
            '8628490433231157','8609960480666910',
            '3557821101183501'] #'8677660403123800' '8642840401594200'

        attck_2_start = "2024-03-23 22:56:00"
        attck_2_end = "2024-03-23 23:56:00"
        ues_to_exclude_in_2nd_attck = [
            '8609960480666910','8642840401612300'
        ]

        mal_filter_1 = (
            df['_time'].between(attck_1_start, attck_1_end)
            & (~df['imeisv'].isin(ues_to_exclude_in_1st_attck))
        )

        mal_filter_2 = (
            df['_time'].between(attck_2_start, attck_2_end)
            & (~df['imeisv'].isin(ues_to_exclude_in_2nd_attck))
        )

        mal_filter_3 = (df['label'] == 1)

        malicious_data = df[(mal_filter_1 | mal_filter_2) & mal_filter_3].copy()
        malicious_data = malicious_data.sort_values(['imeisv','_time'])
        params = {'parameters': {'window_size': 120,
          'step_size': 40,
          'batch_size': 32,
          'hidden_dim1': 50,
          'hidden_dim2': 100,
          'dropout': 0.2,
          'layer_norm_flag': False,
          'loss_function': 'L1Loss',
          'lr': 0.001,
          'num_epochs': 52},
         'min_train_loss': 0.2222,
         'min_val_loss': 0.348,
         'min_train_val_gap': 0.1235,
         'epochs_trained': 44,
         'results_file': '../results/5bfa52f8-e8c6-4899-963d-3ebd80be60f9_history.pkl',
         'timestamp': '2024-04-16 00:52:07.473140',
         'rolling_avg': False,
         'feature_columns': ['ul_bitrate'],
         'dataset_used': 'no_outliers_scaled'}
        train_loader, val_data_loader, mal_data_loader = create_ds_loader(
            benign_data_train, 
            malicious_data, 
            params['parameters']['window_size'], 
            params['parameters']['step_size'], 
            feature_columns, 
            params['parameters']['batch_size']
        )
        test_batch_size = 1
        benign_test_data_loader, mal_test_data_loader = create_test_ds_loaders(
            benign_data_test, 
            malicious_data, 
            120, 
            10, 
            features = feature_columns, 
            batch_size = test_batch_size
        )
        test_loader = mal_test_data_loader

                
    elif dataset == 'HAR':
        train_dir = os.path.join("data", "HAR", "train", "")
        test_dir = os.path.join("data", "HAR", "test", "")

        file = open(train_dir + "X_train.txt", 'r')
        X_train = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
        file.close()

        file = open(train_dir + "y_train.txt", 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_train = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        file = open(test_dir + "X_test.txt", 'r')
        X_test = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
        file.close()

        file = open(test_dir + "y_test.txt", 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_test = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        # Loading which datapoint belongs to which client
        file = open(train_dir + "subject_train.txt", 'r')
        train_clients = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        file = open(test_dir + "subject_test.txt", 'r')
        test_clients = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        y_train, y_test, X_train, X_test = [], [], [], []

        clients = np.concatenate((train_clients, test_clients))
        for client in range(1, 31):
            mask = tuple([clients == client])
            x_client = X[mask]
            y_client = y[mask]

            split = np.concatenate((np.ones(int(np.ceil(0.75*len(y_client))), dtype=bool), np.zeros(int(np.floor(0.25*len(y_client))), dtype=bool)))
            np.random.shuffle(split)  # Generate mask for train test split with ~0.75 1
            x_train_client = x_client[split]
            y_train_client = y_client[split]
            x_test_client = x_client[np.invert(split)]
            y_test_client = y_client[np.invert(split)]

            # Attach vector of client id to training data for data assignment in assign_data()
            x_train_client = np.insert(x_train_client, 0, client, axis=1)
            if len(X_train) == 0:
                X_train = x_train_client
                X_test = x_test_client
                y_test = y_test_client
                y_train = y_train_client
            else:
                X_train = np.append(X_train, x_train_client, axis=0)
                X_test = np.append(X_test, x_test_client, axis=0)
                y_test = np.append(y_test, y_test_client)
                y_train = np.append(y_train, y_train_client)

        tensor_train_X = torch.tensor(X_train, dtype=torch.float32)
        tensor_test_X = torch.tensor(X_test, dtype=torch.float32)
        tensor_train_y = torch.tensor(y_train, dtype=torch.int64) - 1
        tensor_test_y = torch.tensor(y_test, dtype=torch.int64) - 1
        train_dataset = torch.utils.data.TensorDataset(tensor_train_X, tensor_train_y)
        test_dataset = torch.utils.data.TensorDataset(tensor_test_X, tensor_test_y)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    else:
        raise NotImplementedError
    return train_loader, test_loader


def assign_data(train_data, bias, device, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="HAR", seed=1):
    """
    Assign the data to the clients.
    train_data: dataloader of the training dataset
    bias: degree of non-iid between the classes loaded by each client
    device: device used in training and inference
    num_labels: number of classes
    num_workers: number of benign and malicious clients used during training
    server_pc: number of data examples in the server dataset
    p: bias probability in server dataset
    dataset: name of the dataset
    seed: seed for randomness
    """
    other_group_size = (1 - bias) / (num_labels - 1)
    if dataset == "HAR":
        worker_per_group = 30 / num_labels
    elif dataset == 'NWDAF':
        worker_per_group = 30 / num_labels
    else:
        raise NotImplementedError

    # assign training data to each worker
    if dataset == "HAR":
        each_worker_data = [[] for _ in range(30)]
        each_worker_label = [[] for _ in range(30)]
    if dataset == "NWDAF":
        each_worker_data = [[] for _ in range(30)]
        each_worker_label = [[] for _ in range(30)]
    else:
        raise NotImplementedError
    server_data = []
    server_label = []

    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])
    server_counter = [0 for _ in range(num_labels)]

    # compute the labels needed for each class
    if dataset == "HAR":
        for _, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)

            for (x, y) in zip(data, label):
                clientId = int(x[0].item())-1
                x = x[1:len(x)]
                x = x.reshape(1, 561)
                # Assign x and y to appropriate client or server based on method by original code
                if server_counter[int(y.cpu().numpy())] < samp_dis[int(y.cpu().numpy())]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[int(y.cpu().numpy())] += 1
                else:
                    each_worker_data[clientId].append(x)
                    each_worker_label[clientId].append(y)
    else:
       raise NotImplementedError

    if server_pc != 0:
        server_data = torch.cat(server_data, dim=0)
        server_label = torch.stack(server_label, dim=0)
    else:
        if dataset == "HAR":
            server_data = torch.empty(size=(0, 561)).to(device)
        else:
            raise NotImplementedError

        server_label = torch.empty(size=(0, )).to(device)

    each_worker_data = [torch.cat(each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [torch.stack(each_worker, dim=0) for each_worker in each_worker_label]

    # randomly permute the workers
    if dataset == "HAR":
        random_order = np.random.RandomState(seed=seed).permutation(30)
    else:
        raise NotImplementedError
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return server_data, server_label, each_worker_data, each_worker_label
