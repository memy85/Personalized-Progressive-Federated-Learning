from pathlib import Path
import collections
import numpy as np


def create_base_client_data(num_clients, input, label):
    total_num_samples = len(input)
    num_samples_per_set = int(np.floor(total_num_samples/num_clients))
    federated_data = collections.OrderedDict()
    data = collections.OrderedDict()
    for i in range(1, num_clients+1):
        client_name = "client-" + str(i)
        start = num_samples_per_set * (i-1)
        end = num_samples_per_set * i

        if i == num_clients:
            data['input'] = input[start:]
            data['label'] = label[start:]
        else:
            data['input'] = input[start:end]
            data['label'] = label[start:end]
        print(f"{client_name}: {len(data['input'])} data")
        federated_data[client_name] = data
    return federated_data


def create_base_server_data(num_clients, input):
    total_num_samples = len(input)
    num_samples_per_set = int(np.floor(total_num_samples/num_clients))
    federated_data = collections.OrderedDict()
    data=collections.OrderedDict()
    for i in range(1, num_clients+1):
        client_name = "client-" + str(i)
        start = num_samples_per_set * (i-1)
        end = num_samples_per_set * i

        if i == num_clients:
            data['input'] = input[start:]
        else:
            data['input'] = input[start:end]
        print(f"{client_name}: {len(data['input'])} data")
        federated_data[client_name] = data
    return federated_data


if __name__ == "__main__":
    PROJECT_PATH = Path('.').absolute()
    DATA_PATH = Path(PROJECT_PATH, 'data/titanic/kaggle')
    edge_train = np.load(Path(DATA_PATH, 'edge_train_input.npy'))
    cloud_train = np.load(Path(DATA_PATH, 'cloud_train_input.npy'))
    edge_test = np.load(Path(DATA_PATH, 'edge_test_input.npy'))
    cloud_test = np.load(Path(DATA_PATH, 'cloud_test_input.npy'))
    label_train = np.load(Path(DATA_PATH, 'label_train.npy'))
    label_test = np.load(Path(DATA_PATH, 'label_test.npy'))

    client_train = create_base_client_data(20, edge_train, label_train)
    server_train = create_base_server_data(20, cloud_train)



