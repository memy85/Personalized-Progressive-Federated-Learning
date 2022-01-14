import tensorflow as tf
from tensorflow.keras import losses, optimizers
from base.learn import NetworkLearningProcedure

class PersonalizedNetworkLearn(NetworkLearningProcedure):
    def __init__(self, network, config):
        super(PersonalizedNetworkLearn, self).__init__(network, config)
        self.network = None
        self.tape = None

        self.epochs: int = config.personalized.train.epochs
        self.loss_fn: losses.Loss = eval(config.personalized.train.loss_fn)
        self.optimize_fn: optimizers.Optimizer = eval(config.personalized.train.optimize_fn)
        self.learning_rate: float = config.personalized.train.learning_rate
        self.batch_size: int = config.personalized.train.batch_size
        self.buffer_size: int = config.personalized.train.buffer_size
        self.loss_metric: tf.keras.metrics.Metric = eval(config.personalized.train.loss_metric)
        self.evaluate_metric: tf.keras.metrics.Metric = eval(config.personalized.train.evaluate_metric)
        self.random_seed: int = config.random.random_seed

        self.result_state = None

    def learn(self, common_inputs, vertical_inputs, labels, valid_data, verbose, vertical=True):
        train_dataset = self.create_train_dataset(common_inputs, vertical_inputs, labels)

        self.result_state = {}
        self.result_state[f'train_{self.loss_metric().name}'] = []
        self.result_state[f'train_{self.evaluate_metric().name}'] = []
        self.result_state[f'valid_{self.loss_metric().name}'] = []
        self.result_state[f'valid_{self.evaluate_metric().name}'] = []

        for epoch in range(self.epochs):
            if verbose == 1: print("=====" * 10, f"epoch {epoch + 1}: ")
            train_loss, train_eval = self.train_one_epoch(train_dataset, vertical=vertical)

            self.result_state[f'train_{self.loss_metric().name}'].append(train_loss)
            self.result_state[f'train_{self.evaluate_metric().name}'].append(train_eval)

            if verbose == 1: print(f"train {self.loss_metric().name}: {train_loss}, "
                                   f"train {self.evaluate_metric().name}: {train_eval}")
            if valid_data is not None:
                if vertical==True: valid_loss_, valid_eval_ = self.validation(valid_data[0], valid_data[1],
                                                                              valid_data[1])
                else: valid_loss_, valid_eval_ = self.validation(valid_data[0], valid_data[1])

                self.result_state[f'valid_{self.loss_metric().name}'].append(valid_loss_)
                self.result_state[f'valid_{self.evaluate_metric().name}'].append(valid_eval_)
                if verbose == 1: print(f"valid {self.loss_metric().name}: {valid_loss_}, "
                                       f"valid {self.evaluate_metric().name}: {valid_eval_}")

    def create_train_dataset(self, common_inputs, vertical_inputs, labels):
        if vertical_inputs: train_dataset = tf.data.Dataset.from_tensor_slices(
            (common_inputs, vertical_inputs, labels)
        ).shuffle(buffer_size=self.buffer_size, seed=self.random_seed).batch(self.batch_size)
        else: train_dataset = tf.data.Dataset.from_tensor_slices((common_inputs, labels)).shuffle(
            buffer_size=self.buffer_size,
            seed=self.random_seed).batch(self.batch_size)

    def forward(self, common_inputs, vertical_inputs, labels):
        with tf.GradientTape(persistent=True) as self.tape:
            if vertical_inputs: predictions = self.network([common_inputs, vertical_inputs], labels)
            else: predictions = self.network(common_inputs, labels)
            empirical_loss = tf.reduce_mean(self.loss_fn()(labels, predictions))
        return predictions, empirical_loss

    def backward(self, empirical_loss):
        grads = self.tape.gradient(empirical_loss, self.network.trainable_variables)
        return grads

    def update(self, grads):
        self.optimize_fn(self.learning_rate).apply_gradients(
            zip(grads, self.network.trainable_variables)
        )

    def validation(self, common_inputs, vertical_inputs, labels):
        if vertical_inputs: predictions = self.network([common_inputs, vertical_inputs], training=False)
        valid_loss = tf.reduce_sum(self.loss_metric()(labels, predictions)).numpy()
        valid_eval = tf.reduce_sum(self.evaluate_metric()(labels, predictions)).numpy()
        return valid_loss, valid_eval

    def train_one_epoch(self, train_dataset:tf.data.Dataset, vertical: bool=True):
        loss_metric = self.loss_metric()
        evaluate_metric = self.evaluate_metric()
        if vertical==True:
            for step, (common_batch, vertical_batch, label_batch) in enumerate(train_dataset):
                predictions, empirical_loss = self.forward(common_inputs=common_batch, vertical_inputs=vertical_batch,
                                                           labels=label_batch)
                grads = self.backward(empirical_loss)
                self.update(grads)
                loss_metric.update_state(y_true=label_batch, y_pred=predictions)
                evaluate_metric.update_state(y_true=label_batch, y_pred=predictions)
            train_loss = loss_metric.result()
            train_eval = evaluate_metric.result()
        else:
            for step, (input_batch, label_batch) in enumerate(train_dataset):
                predictions, empirical_loss = self.forward(common_inputs=input_batch, vertical_inputs=None,
                                                           labels=label_batch)
                grads = self.backward(empirical_loss)
                self.update(grads)
                loss_metric.update_state(y_true=label_batch, y_pred=predictions)
                evaluate_metric.update_state(y_true=label_batch, y_pred=predictions)
            train_loss = loss_metric.result()
            train_eval = evaluate_metric.result()
        return train_loss, train_eval
