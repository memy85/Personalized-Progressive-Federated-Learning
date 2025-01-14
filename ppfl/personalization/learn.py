from typing import Dict
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, metrics, models
import hydra
import mlflow

from base.learn import NetworkLearningProcedure
from utils.utils import log_params_from_omegaconf_dict


class PersonalizedNetworkLearn(NetworkLearningProcedure):
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

        self.config = config
        self.experiment_name = config.mlflow.experiment_name

        self.epochs: int = config.train.epochs
        self.random_seed: int = eval(config.random.random_seed)
        self.batch_size: int = config.train.batch_size
        self.buffer_size: int = config.train.buffer_size
        self.learning_rate: float = config.train.learning_rate
        self.early_stopping: bool = config.train.early_stopping
        self.stop_patience: int = config.train.stop_patience

        self.current_epoch = 0
        self.current_iter = 0

        if self.strategy is not None:
            self.global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync
            with self.strategy.scope():
                self.loss_fn = eval(config.train.loss_fn)(reduction=losses.Reduction.NONE)
                self.optimizer = eval(config.train.optimize_fn)(learning_rate=self.learning_rate)
                self.loss_metric = eval(config.train.loss_metric)()
                self.result_metric = eval(config.train.evaluate_metric)()
        else:
            self.global_batch_size: int = None
            self.loss_fn = eval(config.train.loss_fn)()
            self.optimizer = eval(config.train.optimize_fn)(learning_rate=self.learning_rate)
            self.loss_metric = eval(config.train.loss_metric)()
            self.result_metric = eval(config.train.evaluate_metric)()
        self.result_state: Dict = None


    def learn(self, horizontal_inputs, vertical_inputs, labels, valid_data=None, verbose=1, project_path=None, save_path=None):
        if vertical_inputs is not None: vertical = True
        else: vertical = False

        train_dataset = self.create_train_dataset(horizontal_inputs, vertical_inputs, labels)
        # vertical 없을 경우 vertical inputs = None

        if valid_data is not None: valid_dataset = self.create_valid_dataset(valid_data[0], valid_data[1], valid_data[2])
        # valid_data : [horizontal_inputs, vertical_inputs or None, labels]

        self.result_state = {}
        self.result_state[f'train_{self.loss_metric.name}'] = []
        self.result_state[f'train_{self.result_metric.name}'] = []
        self.result_state[f'valid_{self.loss_metric.name}'] = []
        self.result_state[f'valid_{self.result_metric.name}'] = []

        if project_path: pass
        else: project_path = hydra.utils.get_original_cwd()
        mlrun_path = Path(project_path, 'mlruns')
        mlflow.set_tracking_uri('file://'+str(mlrun_path))
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            log_params_from_omegaconf_dict(self.config)

            stop_wait = 0
            best = 0
            self.current_epoch = 0
            self.current_iter = 0
            for epoch in tqdm(range(self.epochs), desc='learn', unit=' epoch'):
                if verbose==1: print("\n", "=====" * 10, f"epoch {epoch + 1}: ")
                start_time = time.time()
                train_loss, train_eval = self.train_one_epoch(train_dataset, vertical=vertical)
                train_loss, train_eval = np.round(float(train_loss), 5), np.round(float(train_eval), 5)

                self.result_state[f'train_{self.loss_metric.name}'].append(train_loss)
                self.result_state[f'train_{self.result_metric.name}'].append(train_eval)

                if verbose==1: print(f"train {self.loss_metric.name}: {train_loss}, "
                                  f"train {self.result_metric.name}: {train_eval}")

                if valid_data is not None:
                    valid_loss, valid_eval = self.valid_one_epoch(valid_dataset, vertical=vertical)
                    valid_loss, valid_eval = np.round(float(valid_loss), 4), np.round(float(valid_eval), 4)

                    self.result_state[f'valid_{self.loss_metric.name}'].append(valid_loss)
                    self.result_state[f'valid_{self.result_metric.name}'].append(valid_eval)
                    if verbose==1: print(f"valid {self.loss_metric.name}: {valid_loss}, "
                                      f"valid {self.result_metric.name}: {valid_eval}")

                    # The early stopping strategy: stop the training if `val_loss` does not
                    # decrease over a certain number of epochs.
                    if self.early_stopping:
                        stop_wait += 1
                        if valid_loss > best:
                            best = valid_loss
                            stop_wait = 0
                        if stop_wait >= self.stop_patience:
                            print("=====" * 5, 'early stop', "=====" * 5)
                            break

                if verbose == 1: print("Time taken: %.2fs" % (time.time() - start_time))

            if save_path: self.model.save(Path(save_path, self.config.network.task))

    def create_train_dataset(self, horizontal_inputs, vertical_inputs, labels):
        if self.strategy is None:
            dataset = tf.data.Dataset.from_tensor_slices((horizontal_inputs, vertical_inputs, labels)).shuffle(
                buffer_size=self.buffer_size, seed=self.random_seed).batch(
                self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset
        else:  # self.strategy is not None
            dataset = tf.data.Dataset.from_tensor_slices((horizontal_inputs, vertical_inputs, labels)).shuffle(
                buffer_size=self.buffer_size, seed=self.random_seed).batch(
                self.global_batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            return dist_dataset

    def create_valid_dataset(self, horizontal_inputs, vertical_inputs, labels):

        if self.strategy is None:
            if vertical_inputs is not None: dataset = tf.data.Dataset.from_tensor_slices((horizontal_inputs, vertical_inputs, labels)).batch(self.batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            else: dataset = tf.data.Dataset.from_tensor_slices((horizontal_inputs, labels)).batch(self.batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset

        else:  # self.strategy is not None
            if vertical_inputs is not None:
                dataset = tf.data.Dataset.from_tensor_slices((horizontal_inputs, vertical_inputs, labels)).batch(
                    self.global_batch_size).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
            else:
                dataset = tf.data.Dataset.from_tensor_slices((horizontal_inputs, labels)).batch(
                    self.global_batch_size).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            return dist_dataset

    def forward(self, horizontal_inputs, vertical_inputs, labels):
        """
        Base forward inference function for tensorflow network.
        This method calculates model's forward inference value h and empirical loss
        :param input: server input data
        :return: inference value = intermediate vector h
        """
        with tf.GradientTape(persistent=True) as self.tape:
            if vertical_inputs is not None: predictions = self.model([horizontal_inputs, vertical_inputs], training=True)
            else: predictions = self.model(horizontal_inputs, training=True)

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
    def train_one_step(self, horizontal_inputs, vertical_inputs, labels):
        if vertical_inputs is not None: predictions, empirical_loss = self.forward(horizontal_inputs, vertical_inputs, labels)
        else: predictions, empirical_loss = self.forward(horizontal_inputs=horizontal_inputs, vertical_inputs=None, labels=labels)

        grads = self.backward(empirical_loss)
        self.update(grads)
        self.loss_metric.update_state(y_true=labels, y_pred=predictions)
        self.result_metric.update_state(y_true=labels, y_pred=predictions)
        return empirical_loss

    @tf.function
    def valid_one_step(self, horizontal_inputs, vertical_inputs, labels):
        if vertical_inputs is not None: predictions = self.model([horizontal_inputs, vertical_inputs], training=False)
        else: predictions = self.model(horizontal_inputs, training=False)
        self.loss_metric.update_state(y_true=labels, y_pred=predictions)
        self.result_metric.update_state(y_true=labels, y_pred=predictions)

    @tf.function
    def distributed_train_step(self, horizontal_inputs, vertical_inputs, labels):
        if vertical_inputs:
            per_replica_losses = self.strategy.run(
                self.train_one_step, args=(horizontal_inputs, vertical_inputs, labels)
            )
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        else:
            per_replica_losses = self.strategy.run(self.train_one_step, args=(horizontal_inputs, labels))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_valid_step(self, horizontal_inputs, vertical_inputs, labels):
        if vertical_inputs is not None: self.strategy.run(self.valid_one_step, args=(horizontal_inputs, vertical_inputs, labels,))
        else: self.strategy.run(self.valid_one_step, args=(horizontal_inputs, labels,))

    def train_one_epoch(self, train_dataset:tf.data.Dataset, vertical=True):
        if vertical==True:
            for step, (h_input_batch, v_input_batch, label_batch) in enumerate(train_dataset):
                if self.strategy is None: self.train_one_step(h_input_batch, v_input_batch, label_batch)
                else: self.distributed_train_step(h_input_batch, v_input_batch, label_batch)
                # mlflow.log_metric(
                #     f'{self.loss_metric.name}_train_step',
                #     np.round(float(step_loss), 4), step=self.current_iter
                # ) TODO step 단위 log 추가

                self.current_iter += 1
            self.current_epoch += 1

        else:
            for step, (h_input_batch, _, label_batch) in enumerate(train_dataset):
                if self.strategy is None: self.train_one_step(h_input_batch, None, label_batch)
                else: self.distributed_train_step(h_input_batch, None, label_batch)
                # mlflow.log_metric(
                #     f'{self.loss_metric.name}_train_step',
                #     np.round(float(step_loss), 4), step=self.current_iter
                # ) TODO step 단위 log 추가

                self.current_iter += 1
            self.current_epoch += 1

        train_loss = self.loss_metric.result()
        train_eval = self.result_metric.result()
        self.loss_metric.reset_states()
        self.result_metric.reset_states()

        mlflow.log_metric(
            f'{self.loss_metric.name}_train_epoch',
            np.round(float(train_loss), 4), step=self.current_epoch
        )
        mlflow.log_metric(
            f'{self.result_metric.name}_train_epoch',
            np.round(float(train_eval), 4), step=self.current_epoch
        )

        return train_loss, train_eval

    def valid_one_epoch(self, valid_dataset, vertical=True):

        if vertical==True:
            for step, (h_input_batch, v_input_batch, label_batch) in enumerate(valid_dataset):
                if self.strategy is None: self.valid_one_step(h_input_batch, v_input_batch, label_batch)
                else: self.distributed_valid_step(h_input_batch, v_input_batch, label_batch)
                # mlflow.log_metric(
                #     f'{self.loss_metric.name}_train_step',
                #     np.round(float(step_loss), 4), step=self.current_iter
                # ) TODO step 단위 log 추가

                self.current_iter += 1
            self.current_epoch += 1

        else:
            for step, (h_input_batch, _, label_batch) in enumerate(valid_dataset):
                if self.strategy is None: self.train_one_step(h_input_batch, label_batch)
                else: self.distributed_train_step(h_input_batch, label_batch)
                # mlflow.log_metric(
                #     f'{self.loss_metric.name}_train_step',
                #     np.round(float(step_loss), 4), step=self.current_iter
                # ) TODO step 단위 log 추가

        valid_loss = self.loss_metric.result()
        valid_eval = self.result_metric.result()
        self.loss_metric.reset_states()
        self.result_metric.reset_states()

        mlflow.log_metric(
            f'{self.loss_metric.name}_valid_epoch',
            np.round(float(valid_loss), 4), step=self.current_epoch
        )
        mlflow.log_metric(
            f'{self.result_metric.name}_valid_epoch',
            np.round(float(valid_eval), 4), step=self.current_epoch
        )

        return valid_loss, valid_eval

    def create_callbacks(self):
        raise NotImplementedError()
