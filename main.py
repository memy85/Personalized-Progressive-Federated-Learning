import utils.gpu_utils
from ppfl.client import *
from ppfl.server import *
from utils.save_load import *
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra

def load_fl_data(data_dir):
    client_common_train = pkl_to_OrderedDict('icu_client_common_train.pkl', data_dir)
    client_vertical_train = pkl_to_OrderedDict('icu_client_vertical_train.pkl', data_dir)
    client_full_train = pkl_to_OrderedDict('icu_client_full_train.pkl', data_dir)
    client_common_valid = pkl_to_OrderedDict('icu_client_common_valid.pkl', data_dir)
    client_vertical_valid = pkl_to_OrderedDict('icu_client_vertical_valid.pkl', data_dir)
    client_full_valid = pkl_to_OrderedDict('icu_client_full_valid.pkl', data_dir)
    client_common_test = pkl_to_OrderedDict('icu_client_common_test.pkl', data_dir)
    client_vertical_test = pkl_to_OrderedDict('icu_client_vertical_test.pkl', data_dir)
    client_full_test = pkl_to_OrderedDict('icu_client_full_test.pkl', data_dir)
    external_data = pkl_to_OrderedDict('icu_external_data.pkl', data_dir)

    for client in client_full_train:
        client_full_train[client]['label'] = tf.keras.utils.to_categorical(client_full_train[client]['label'].values, num_classes=2)
    for client in client_full_valid:
        client_full_valid[client]['label'] = tf.keras.utils.to_categorical(client_full_valid[client]['label'].values, num_classes=2)
    for client in client_full_test:
        client_full_test[client]['label'] = tf.keras.utils.to_categorical(client_full_test[client]['label'].values, num_classes=2)
    external_data['label'] = tf.keras.utils.to_categorical(external_data['label'])

    return client_common_train, client_vertical_train, client_full_train, \
           client_common_valid, client_vertical_valid, client_full_valid, \
           client_common_test, client_vertical_test, client_full_test, external_data

def expeirment(distribute=False):
    # PROJECT_PATH = Path('.').absolute().parents[1]
    PROJECT_PATH = Path('.').absolute()
    DATA_PATH = Path(PROJECT_PATH, 'data', 'physionet2012')
    config = OmegaConf.load('config/config.yaml')
    client_common_train, client_vertical_train, client_full_train, \
    client_common_valid, client_vertical_valid, client_full_valid, \
    client_common_test, client_vertical_test, client_full_test, external_data = load_fl_data(DATA_PATH)

    lst = [client_common_train, client_vertical_train, client_full_train, client_common_valid, client_vertical_valid, client_full_valid,
           client_common_test, client_vertical_test]
    for data_dict in lst:
        for client in data_dict:
            data_dict[client]['label'] = tf.keras.utils.to_categorical(data_dict[client]['label'], 2)

    clients = create_clients_(4, client_common_train, input_str='input_train', label_str='label',
                             client_str='client_', distribute=distribute)
    central_server = CentralServer(config,
                            MLPNetwork, BaseNetworkLearn, BaseFederatedLearn, FedAvg, distribute=distribute)

    central_server.learn(clients, valid_data=[external_data['input_common'], external_data['label']])

    clients[0].create_personalized_network(
        config, MLPNetwork, BaseNetworkLearn,
        PersonalizedProgressiveNetwork,PersonalizedNetworkLearn,
        freeze = True, vertical = True
        )
    clients[0].personalized_learn(client_common_train['client_1']['input_train'],
                                  client_vertical_train['client_1']['input_train'],
                                  tf.keras.utils.to_categorical(client_common_train['label']),
                                  valid_data=(client_common_valid['client_1']['input_valid'], client_vertical_valid['client_1']['input_valid'],
                                              tf.keras.utils.to_categorical(client_common_train['label'])),
                                  verbose=1, vertical=1
                                  )



if __name__=="__main__":
    utils.gpu_utils.disable_tensorflow_debugging_logs()
    distribute = False
    print("gpu distribute: ", distribute)
    expeirment(distribute)