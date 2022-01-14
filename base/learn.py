from typing import Dict, List
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, metrics, models


class NetworkLearningProcedure(object):
    def __init__(self, network, config):
        pass

    def learn(self):
        raise NotImplementedError

    def create_train_dataset(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

class BaseNetworkLearn(NetworkLearningProcedure):
    def __init__(self, network, config):
        """
        This class is a module for learning network model.
        :param network: tensorflow Model
        :param config: omegaconf - DictConf
        """
        super().__init__(network, config)
        self.network: models.Model = network
        self.tape = None

        self.epochs: int = config.train.epochs
        self.loss_fn: losses.Loss = eval(config.train.loss_fn)
        self.optimize_fn: optimizers.Optimizer = eval(config.train.optimize_fn)
        self.learning_rate: float = config.train.learning_rate
        self.batch_size: int = config.train.batch_size
        self.buffer_size: int = config.train.buffer_size
        self.loss_metric: tf.keras.metrics.Metric = eval(config.train.loss_metric)
        self.evaluate_metric: tf.keras.metrics.Metric = eval(config.train.evaluate_metric)
        self.random_seed: int = config.random.random_seed

        self.result_state: Dict = None

    def learn(self, inputs, labels, valid_data=None, verbose=1):
        train_dataset = self.create_train_dataset(inputs, labels)

        self.result_state = {}
        self.result_state[f'train_{self.loss_metric().name}'] = []
        self.result_state[f'train_{self.evaluate_metric().name}'] = []
        self.result_state[f'valid_{self.loss_metric().name}'] = []
        self.result_state[f'valid_{self.evaluate_metric().name}'] = []

        for epoch in range(self.epochs):
            if verbose==1: print("=====" * 10, f"epoch {epoch + 1}: ")
            train_loss, train_eval = self.train_one_epoch(train_dataset)

            self.result_state[f'train_{self.loss_metric().name}'].append(train_loss)
            self.result_state[f'train_{self.evaluate_metric().name}'].append(train_eval)

            if verbose==1: print(f"train {self.loss_metric().name}: {train_loss}, "
                              f"train {self.evaluate_metric().name}: {train_eval}")
            if valid_data is not None:
                valid_loss_, valid_eval_ = self.validation(valid_data[0], valid_data[1])

                self.result_state[f'valid_{self.loss_metric().name}'].append(valid_loss_)
                self.result_state[f'valid_{self.evaluate_metric().name}'].append(valid_eval_)
                if verbose==1: print(f"valid {self.loss_metric().name}: {valid_loss_}, "
                                  f"valid {self.evaluate_metric().name}: {valid_eval_}")

    def create_train_dataset(self, inputs, labels):
        train_dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(
            buffer_size=self.buffer_size,
            seed=self.random_seed).batch(self.batch_size)
        return train_dataset

    def forward(self, inputs, labels):
        """
        Base forward inference function for tensorflow network.
        This method calculates model's forward inference value h and empirical loss
        :param input: server input data
        :return: inference value = intermediate vector h
        """
        with tf.GradientTape(persistent=True) as self.tape:
            predictions = self.network(inputs, training=True)
            empirical_loss = tf.reduce_mean(self.loss_fn()(labels, predictions))
        return predictions, empirical_loss

    def backward(self, empirical_loss):
        """
        backward backpropagation function for Server network.
        calculate model's weight gradients with h gradient from client
        (dE/dh)*(dh/dw)=dE/dw
        :param h: intermediate vector h from server forward function
        :param h_grad_from_client: gradients of h from client backward function
        :return: weight gradients of clients model
        """
        grads = self.tape.gradient(empirical_loss, self.network.trainable_variables)
        return grads

    def update(self, grads):
        self.optimize_fn(self.learning_rate).apply_gradients(
            zip(grads, self.network.trainable_variables)
        )

    def validation(self, inputs, labels):
        predictions = self.network(inputs, training=False)
        valid_loss = tf.reduce_sum(self.loss_metric()(labels, predictions)).numpy()
        valid_eval = tf.reduce_sum(self.evaluate_metric()(labels, predictions)).numpy()
        return valid_loss, valid_eval

    def train_one_epoch(self, train_dataset:tf.data.Dataset):
        loss_metric = self.loss_metric()
        evaluate_metric = self.evaluate_metric()
        for step, (train_batch, label_batch) in enumerate(train_dataset):
            predictions, empirical_loss = self.forward(train_batch, label_batch)
            grads = self.backward(empirical_loss)
            self.update(grads)
            loss_metric.update_state(y_true=label_batch, y_pred=predictions)
            evaluate_metric.update_state(y_true=label_batch, y_pred=predictions)
        train_loss = loss_metric.result()
        train_eval = evaluate_metric.result()
        return train_loss, train_eval


class FederatedLearningProcedure(object):
    def __init__(self):
        pass

    def learn(self):
        raise NotImplementedError

    def client_updates_one_round(self, selected_clients: List):
        raise NotImplementedError
    
    def train_one_round(self):
        raise NotImplementedError

    def valid_one_round(self):
        raise NotImplementedError

class BaseFederatedLearn(FederatedLearningProcedure):
    def __init__(self, server, aggregate_fn, config):
        """
        This class is a module for learning federated model with multiple clients.
        :param server: server module
        :param aggregate_fn: aggregate function e.g. FedAvg
        :param config: omegaconf - DictConf
        """
        super().__init__()
        self.server = server
        self.aggregator = aggregate_fn()

        self.config = config
        self.num_rounds = config.federate.num_rounds
        self.c_fraction = config.federate.c_fraction
        self.num_clients = config.federate.num_clients
        self.loss_metric = eval(config.train.loss_metric)
        self.evaluate_metric = eval(config.train.evaluate_metric)
        self.predict_batch_size: int = config.federate.predict_batch_size

        # result
        self.federated_loss_per_round: List = None
        self.train_loss_per_round: List = None
        self.train_eval_per_round: List = None
        self.valid_loss_per_round: List = None
        self.valid_eval_per_round: List = None

    def client_updates_one_round(self, selected_clients: List):
        selected_client_n_k_list = []
        selected_client_weight_list = []
        selected_client_loss_list = []
        for client in tqdm(selected_clients, desc='client update', unit=' client'):  # parallel
            client.create_network(self.server.config, self.server.network_module, self.server.network_learn_module
                                  )
            client.receive_global_weights()  # receive weights from server
            client.set_client_weights(self.server.global_net.network.get_weights())  # set global weights
            client.learn(valid_data=None, verbose=0)  # client learn
            # send client information to server (weights, loss, n_k_sample)
            client.send_weights()
            selected_client_n_k_list.append(client.n_k_sample)
            selected_client_weight_list.append(client.client_net.network.get_weights())
            selected_client_loss_list.append(client.client_net.learn_module.result_state[f'train_{self.loss_metric().name}'][-1])
            client.end_learn() # del client.net / K.clear_session()
        return selected_client_n_k_list, selected_client_weight_list, selected_client_loss_list
            
    def train_one_round(self, clients: List):  # server
        """
        train one round of learning procedure
        clients learn its local parameter and sever aggregates the weights of selected clients

        :param clients:
        :return:
        """
        # server select clients
        selected_clients = self.server.select_clients(clients)  # 이 때는 client 객체 필요 없음.

        n_sample = sum(client.n_k_sample for client in selected_clients)  # total n

        # clients update parallel
        selected_client_n_k_list, selected_client_weight_list, selected_client_loss_list = \
            self.client_updates_one_round(selected_clients)

        # aggregate clients weights
        federated_weights, federated_loss = self.aggregator.aggregate(
            n_sample=n_sample,
            selected_client_n_k_list=selected_client_n_k_list,
            selected_client_loss_list=selected_client_loss_list,
            selected_client_weight_list=selected_client_weight_list
        )
        self.server.set_global_weights(federated_weights)  # set global weights as server weights

        # train eval for all client's input
        predictions = self.server.global_net.network.predict(
            np.concatenate([client.inputs for client in selected_clients], axis=0),
            batch_size=self.predict_batch_size)
        labels = np.concatenate([client.labels for client in selected_clients], axis=0)
        # train loss
        train_loss = (tf.reduce_sum(self.loss_metric()(labels, predictions))).numpy()
        # train eval metric
        train_eval = (tf.reduce_sum(self.evaluate_metric()(labels, predictions))).numpy()
        return federated_loss, train_loss, train_eval

    def valid_one_round(self, valid_data: List):
        """

        :param valid_data: List(input_valid, label_valid)
        :return:
        """
        # valid eval for external validation dataset
        predictions = self.server.global_net.network.predict(valid_data[0], self.predict_batch_size)
        labels = valid_data[1]
        # valid loss
        valid_loss = (tf.reduce_sum(self.loss_metric()(labels, predictions))).numpy()
        # train eval metric
        valid_eval = (tf.reduce_sum(self.evaluate_metric()(labels, predictions))).numpy()
        return valid_loss, valid_eval

    def learn(self, clients: List, valid_data: List=None):
        self.federated_loss_per_round = []
        self.train_loss_per_round = []
        self.train_eval_per_round = []
        if valid_data is not None:
            self.valid_loss_per_round = []
            self.valid_eval_per_round = []

        # send initialized global weights to all clients and set client's weights
        # it doesn't actually work, since client object has not network model yet.
        # we kill the network model before after client update for each client
        self.server.send_global_weights()
        for client in clients:  # parallel
            client.receive_global_weights()
            client.set_client_weights(self.server.global_net.network.get_weights())

        # start learning round / client parally do
        for round in range(self.num_rounds):
            print("==========" * 5, f"Round {round + 1}")
            federated_loss, train_loss, train_eval = self.train_one_round(clients)
            self.federated_loss_per_round.append(federated_loss)
            self.train_loss_per_round.append(train_loss)
            self.train_eval_per_round.append(train_eval)
            # Valid
            if valid_data is not None:
                valid_loss, valid_eval = self.valid_one_round(valid_data)
                self.valid_loss_per_round.append(valid_loss)
                self.valid_eval_per_round.append(valid_eval)
                print(f"federated loss: {np.round_(self.federated_loss_per_round[-1], 4)}, "
                      f"train loss: {np.round_(self.train_loss_per_round[-1], 4)}, "
                      f"train {self.evaluate_metric().name}: "
                      f"{np.round_(self.train_eval_per_round[-1], 4)}, \n"
                      f"valid loss: {np.round_(self.valid_loss_per_round[-1], 4)}, "
                      f"valid {self.evaluate_metric().name}: "
                      f"{np.round_(self.valid_eval_per_round[-1], 4)}, "
                      )
            else:
                print(f"federated loss: {np.round_(self.federated_loss_per_round[-1], 4)}, "
                      f"train loss: {np.round_(self.train_loss_per_round[-1], 4)}, "
                      f"train {self.evaluate_metric().name}: "
                      f"{np.round_(self.train_eval_per_round[-1], 4)},"
                      )
        # send and receive final global weights
        # it doesn't actually work, since client object has not network model yet.
        # we kill the network model before after client update for each client
        self.server.send_global_weights()
        for client in clients:
            client.receive_global_weights()
            client.set_client_weights(self.server.global_net.network.get_weights())