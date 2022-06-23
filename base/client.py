import gc
from typing import List, Dict

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics
from base.network import *

class Client(object):
    def __init__(self, client_id, inputs, labels, distribute=False):
        """
        class for controlling individual client's network and information.
        :param client_id: index for client
        :param inputs: inputs of individual client
        :param labels: labels of individual client
        """
        self.distribute = distribute
        self.client_id: int = client_id
        self.weights: List[np.ndarray] = None
        self.inputs: np.ndarray = inputs
        self.labels: np.ndarray = labels

        self.n_k_sample: int = len(self.inputs)
        self.clientnet: Network = None

    def create_network(self, config, network_module: Network, learn_module):
        self.clientnet = network_module(config, learn_module, distribute=self.distribute)

        pass  # statement for client network, not implemented, since the limitation of gpu memory in experiment.

    def learn(self, valid_data=None, verbose=0):
        self.clientnet.learn(self.inputs, self.labels, valid_data=valid_data, verbose=verbose)

    def end_learn(self):
        self.set_client_weights(self.clientnet.model.get_weights())
        gc.collect()
        del self.clientnet.model
        K.clear_session()
        self.clientnet = None

    def send_weights(self):
        pass  # not implemented, statement for client - server communication

    def receive_global_weights(self):
        return  # not implemented, statement for client - server communication

    def set_client_weights(self, global_weights):
        self.weights = global_weights
        if self.clientnet: self.clientnet.model.set_weights(self.weights)


def create_clients(num_clients, client_data, input_str='input', label_str='label', client_str='client-',
                   distribute=False):
    """
    create K clients
    :param config: network_config
    :param num_clients: the number of clients
    :param client_data: Dictionary of clients data. data[client][input or label]
    :param input_str: input string of client_data
    :param label_str: label string of label_data
    :return: List(Client)
    """
    clients = []
    for i in range(num_clients):
        client_id = i+1
        client = Client(client_id,
                        client_data[f'{client_str}{client_id}'][input_str],
                        client_data[f'{client_str}{client_id}'][label_str],
                        distribute)
        clients.append(client)
    return clients