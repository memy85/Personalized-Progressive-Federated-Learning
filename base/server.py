from typing import List, Dict
from omegaconf import DictConfig
from copy import deepcopy
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics
from tqdm import tqdm
from base.learn import *
from base.network import *
from base.client import *
from base.aggregate import *


class Server(object):
    def __init__(self, config: DictConfig,
                 network_module, network_learn_module, federated_module, aggregate_fn,
                 distribute=False):
        """
        class for controlling server's global network and information.
        This class implements the federated process.

        """
        self.config = config
        self.distribute = distribute

        self.num_rounds = config.federate.num_rounds
        self.c_fraction = config.federate.c_fraction
        self.num_clients = config.federate.num_clients
        self.random_seed = config.random.random_seed

        self.network_module = network_module
        self.network_learn_module = network_learn_module

        self.selected_clients_index = None
        self.globalnet: Network = None
        self.create_global_network(config)

        self.federated_module: FederatedLearningProcedure = federated_module(self, aggregate_fn, config)

        # results during a round
        self.selected_client_n_k_list: List = None
        self.selected_client_weight_list: List = None
        self.selected_client_loss_list: List = None


    def learn(self, clients: List[Client], valid_data: List = None):
        self.federated_module.learn(clients, valid_data)

    def create_global_network(self, config):
        self.globalnet = self.network_module(config, self.network_learn_module, self.distribute)
        self.globalnet.model._name = 'server_network'

    def select_clients(self, clients):
        n_selected_clients = max(int(self.num_clients * self.c_fraction), 1)
        np.random.seed(self.random_seed)
        self.selected_clients_index = np.random.choice(range(1, self.num_clients + 1), n_selected_clients,
                                                       replace=False)
        print(f"{len(self.selected_clients_index)} selected clients: ", self.selected_clients_index,
              f"with C fraction {self.c_fraction}")
        selected_clients = [client for client in clients if
                            client.client_id in self.selected_clients_index]  # client list
        return selected_clients

    def set_global_weights(self, global_weights):
        self.globalnet.model.set_weights(global_weights)  # set global weights as server weights

    def send_global_weights(self):
        pass  # not implemented, statement for client - server communication

    def receive_weights(self):
        """
        receive weights from selected clients
        :return:
        """
        pass  # not implemented, statement for client - server communication