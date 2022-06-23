from typing import Dict, List
from omegaconf import DictConfig
from tqdm import tqdm
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, metrics, models

class NetworkLearningProcedure(object):
    def __init__(self, network, config, distribute_strategy):
        self.model = None
        self.tape = None

        self.result_state = None

    def learn(self):
        raise NotImplementedError

    def create_dataset(self):
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
    def __init__(self, network, config, distribute_strategy):
        """
        This class is a module for learning network model.
        :param network: tensorflow Model
        :param config: omegaconf - DictConf
        """
        super().__init__(network, config, distribute_strategy)
        self.model: models.Model = network
        self.tape = None
        self.strategy = distribute_strategy

        self.epochs: int = config.train.epochs
        self.random_seed: int = config.random.random_seed
        self.batch_size: int = config.train.batch_size
        self.buffer_size: int = config.train.buffer_size
        self.learning_rate: float = config.train.learning_rate

        if self.strategy is not None:
            self.global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync
            with self.strategy.scope():
                self.loss_fn: losses.Loss = eval(config.train.loss_fn)(reduction=losses.Reduction.NONE)
                self.optimizer: optimizers.Optimizer = eval(config.train.optimize_fn)(learning_rate=self.learning_rate)
                self.loss_metric: tf.keras.metrics.Metric = eval(config.train.loss_metric)()
                self.result_metric: tf.keras.metrics.Metric = eval(config.train.evaluate_metric)()
        else:
            self.global_batch_size: int = None
            self.loss_fn: losses.Loss = eval(config.train.loss_fn)()
            self.optimizer: optimizers.Optimizer = eval(config.train.optimize_fn)(learning_rate=self.learning_rate)
            self.loss_metric: tf.keras.metrics.Metric = eval(config.train.loss_metric)()
            self.result_metric: tf.keras.metrics.Metric = eval(config.train.evaluate_metric)()
        self.result_state: Dict = None

    def learn(self, inputs, labels, valid_data=None, verbose=1):
        train_dataset = self.create_train_dataset(inputs, labels)
        if valid_data is not None: valid_dataset = self.create_valid_dataset(valid_data[0], valid_data[1])

        self.result_state = {}
        self.result_state[f'train_{self.loss_metric.name}'] = []
        self.result_state[f'train_{self.result_metric.name}'] = []
        self.result_state[f'valid_{self.loss_metric.name}'] = []
        self.result_state[f'valid_{self.result_metric.name}'] = []

        for epoch in range(self.epochs):
            if verbose==1: print("=====" * 10, f"epoch {epoch + 1}: ")
            start_time = time.time()
            train_loss, train_eval = self.train_one_epoch(train_dataset)
            train_loss, train_eval = np.round(float(train_loss), 5), np.round(float(train_eval), 5)

            self.result_state[f'train_{self.loss_metric.name}'].append(train_loss)
            self.result_state[f'train_{self.result_metric.name}'].append(train_eval)

            if verbose==1: print(f"train {self.loss_metric.name}: {train_loss}, "
                              f"train {self.result_metric.name}: {train_eval}")

            if valid_data is not None:
                valid_loss, valid_eval = self.valid_one_epoch(valid_dataset)
                valid_loss, valid_eval = np.round(float(valid_loss), 4), np.round(float(valid_eval), 4)

                self.result_state[f'valid_{self.loss_metric.name}'].append(valid_loss)
                self.result_state[f'valid_{self.result_metric.name}'].append(valid_eval)
                if verbose==1: print(f"valid {self.loss_metric.name}: {valid_loss}, "
                                  f"valid {self.result_metric.name}: {valid_eval}")

            if verbose==1: print("Time taken: %.2fs" % (time.time() - start_time))

    def create_train_dataset(self, inputs, labels):
        if self.strategy is None:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(
                buffer_size=self.buffer_size, seed=self.random_seed).batch(
                self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset
        else:  # self.strategy is not None
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(
                buffer_size=self.buffer_size, seed=self.random_seed).batch(
                self.global_batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            return dist_dataset

    def create_valid_dataset(self, inputs, labels):
        if self.strategy is None:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset
        else:  # self.strategy is not None
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.global_batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            return dist_dataset

    def forward(self, inputs, labels):
        """
        Base forward inference function for tensorflow network.
        This method calculates model's forward inference value h and empirical loss
        :param input: server input data
        :return: inference value = intermediate vector h
        """
        with tf.GradientTape(persistent=True) as self.tape:
            predictions = self.model(inputs, training=True)
            if self.strategy is None:
                empirical_loss = tf.reduce_mean(self.loss_fn(labels, predictions))
            else:  #self.strategy is not None:
                per_example_loss = self.loss_fn(labels, predictions)
                empirical_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)
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
        grads = self.tape.gradient(empirical_loss, self.model.trainable_variables)
        return grads

    def update(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function
    def train_one_step(self, inputs, labels):
        predictions, empirical_loss = self.forward(inputs, labels)
        grads = self.backward(empirical_loss)
        self.update(grads)
        self.loss_metric.update_state(y_true=labels, y_pred=predictions)
        self.result_metric.update_state(y_true=labels, y_pred=predictions)
        return empirical_loss

    @tf.function
    def valid_one_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        self.loss_metric.update_state(y_true=labels, y_pred=predictions)
        self.result_metric.update_state(y_true=labels, y_pred=predictions)

    @tf.function
    def distributed_train_step(self, inputs, labels):
        per_replica_losses = self.strategy.run(self.train_one_step, args=(inputs, labels))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_valid_step(self, inputs, labels):
        self.strategy.run(self.valid_one_step, args=(inputs, labels,))

    def train_one_epoch(self, train_dataset:tf.data.Dataset):
        for step, (input_batch, label_batch) in enumerate(train_dataset):
            if self.strategy is None: self.train_one_step(input_batch, label_batch)
            else: self.distributed_train_step(input_batch, label_batch)
        train_loss = self.loss_metric.result()
        train_eval = self.result_metric.result()
        self.loss_metric.reset_states()
        self.result_metric.reset_states()
        return train_loss, train_eval

    def valid_one_epoch(self, valid_dataset):
        for input_batch, label_batch in valid_dataset:
            if self.strategy is None: self.valid_one_step(input_batch, label_batch)
            else: self.distributed_valid_step(input_batch, label_batch)
        valid_loss = self.loss_metric.result()
        valid_eval = self.result_metric.result()
        self.loss_metric.reset_states()
        self.result_metric.reset_states()
        return valid_loss, valid_eval


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
        self.strategy = self.server.globalnet.strategy
        self.aggregator = aggregate_fn()

        self.config = config
        self.num_rounds = config.federate.num_rounds
        self.c_fraction = config.federate.c_fraction
        self.num_clients = config.federate.num_clients
        self.loss_metric = eval(config.train.loss_metric)()
        self.result_metric = eval(config.train.evaluate_metric)()

        if self.strategy is not None:
            self.global_batch_size = self.config.train.batch_size * self.strategy.num_replicas_in_sync
            with self.strategy.scope():
                self.loss_metric: tf.keras.metrics.Metric = eval(config.train.loss_metric)()
                self.result_metric: tf.keras.metrics.Metric = eval(config.train.evaluate_metric)()
        else:
            self.global_batch_size: int = None
            self.loss_metric: tf.keras.metrics.Metric = eval(config.train.loss_metric)()
            self.result_metric: tf.keras.metrics.Metric = eval(config.train.evaluate_metric)()

        # result
        self.result_state: Dict = None

    def client_updates_one_round(self, selected_clients: List):
        selected_client_n_k_list = []
        selected_client_weight_list = []
        selected_client_loss_list = []
        for client in tqdm(selected_clients, desc='client update', unit=' client'):  # parallel
            client.create_network(self.server.config, self.server.network_module, self.server.network_learn_module)
            client.receive_global_weights()  # receive weights from server
            client.set_client_weights(self.server.globalnet.model.get_weights())  # set global weights
            client.learn(valid_data=None, verbose=0)  # client learn
            # send client information to server (weights, loss, n_k_sample)
            client.send_weights()
            selected_client_n_k_list.append(client.n_k_sample)
            selected_client_weight_list.append(client.clientnet.model.get_weights())
            selected_client_loss_list.append(client.clientnet.learn_module.result_state[f'train_{self.loss_metric.name}'][-1])
            client.end_learn() # del client.net / K.clear_session()
        return selected_client_n_k_list, selected_client_weight_list, selected_client_loss_list

    def create_global_dataset(self, inputs, labels):
        if self.strategy is None:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.config.train.batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset
        else:  # self.strategy is not None
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.global_batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            return dist_dataset

    @tf.function
    def global_one_step(self, inputs, labels):
        predictions = self.server.globalnet.model(inputs, training=False)
        self.loss_metric.update_state(y_true=labels, y_pred=predictions)
        self.result_metric.update_state(y_true=labels, y_pred=predictions)

    @tf.function
    def distributed_global_one_step(self, inputs, labels):
        self.strategy.run(self.global_one_step, args=(inputs, labels,))

    def global_one_epoch(self, dataset):
        for input_batch, label_batch in dataset:
            if self.strategy is None: self.global_one_step(input_batch, label_batch)
            else: self.distributed_global_one_step(input_batch, label_batch)
        global_loss = self.loss_metric.result()
        global_eval = self.result_metric.result()
        self.loss_metric.reset_states()
        self.result_metric.reset_states()
        return global_loss, global_eval

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
        global_train_set = self.create_global_dataset(
            inputs=np.concatenate([client.inputs for client in selected_clients], axis=0),
            labels=np.concatenate([client.labels for client in selected_clients], axis=0)
        )
        train_loss, train_eval = self.global_one_epoch(global_train_set)
        return federated_loss, train_loss, train_eval

    def valid_one_round(self, dataset):
        """

        :param valid_data: tf.data
        :return: valid_loss, valid_eval
        """
        # valid eval for external validation dataset
        valid_loss, valid_eval = self.global_one_epoch(dataset)
        return valid_loss, valid_eval

    def learn(self, clients: List, valid_data: List=None):

        self.result_state = {}
        self.result_state['federated_loss_per_round'] = []
        self.result_state['train_loss_per_round'] = []
        self.result_state['train_eval_per_round'] = []
        if valid_data is not None:
            self.result_state['valid_loss_per_round'] = []
            self.result_state['valid_eval_per_round'] = []
            valid_dataset = self.create_global_dataset(valid_data[0], valid_data[1])

        # send initialized global weights to all clients and set client's weights
        # it doesn't actually work, since client obj
        # ect has not network model yet.
        # w kill the network model before after client update for each client
        self.server.send_global_weights()
        for client in clients:  # parallel
            client.receive_global_weights()
            client.set_client_weights(self.server.globalnet.model.get_weights())

        # start learning round / client parally do
        for round in range(self.num_rounds):
            start_time = time.time()
            print("==========" * 5, f"Round {round + 1}")
            federated_loss, train_loss, train_eval = self.train_one_round(clients)
            federated_loss, train_loss, train_eval = np.round(float(federated_loss), 5), \
                                                     np.round(float(train_loss), 5), np.round(float(train_eval), 5)
            
            self.result_state['federated_loss_per_round'].append(federated_loss)
            self.result_state['train_loss_per_round'].append(train_loss)
            self.result_state['train_eval_per_round'].append(train_eval)
            print(
                f"federated loss: {np.round_(self.result_state['federated_loss_per_round'][-1], 4)}, "
                f"train loss: {np.round_(self.result_state['train_loss_per_round'][-1], 4)}, "
                f"train {self.result_metric.name}: "
                f"{np.round_(self.result_state['train_eval_per_round'][-1], 4)},"
                  )
            # Valid
            if valid_data is not None:
                valid_loss, valid_eval = self.valid_one_round(valid_dataset)
                valid_loss, valid_eval = np.round(float(valid_loss), 5), np.round(float(valid_eval), 5)
                self.result_state['valid_loss_per_round'].append(valid_loss)
                self.result_state['valid_eval_per_round'].append(valid_eval)
                print(
                    f"valid loss: {np.round_(self.result_state['valid_loss_per_round'][-1], 4)}, "
                    f"valid {self.result_metric.name}: "
                    f"{np.round_(self.result_state['valid_eval_per_round'][-1], 4)}, "
                      )
            print("Time taken: %.2fs" % (time.time() - start_time))

        # send and receive final global weights
        # it doesn't actually work, since client object has not network model yet.
        # we kill the network model before after client update for each client
        self.server.send_global_weights()
        for client in clients:
            client.receive_global_weights()
            client.set_client_weights(self.server.globalnet.model.get_weights())
