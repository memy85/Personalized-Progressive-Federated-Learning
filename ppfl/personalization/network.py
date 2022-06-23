from typing import Dict
from tensorflow.keras import losses, optimizers, metrics, models

from base.network import Network
from ppfl.personalization.progressive_layers import *
from ppfl.personalization.learn import PersonalizedNetworkLearn
from base.network import MLPNetwork
from base.learn import BaseNetworkLearn
from tensorflow.keras import models, layers, initializers

class PersonalizedProgressiveNetwork(Network):

    def __init__(self, cfg, federated_model, client_id: str, freeze: bool=True, vertical: bool=True, distribute=False):
        super(PersonalizedProgressiveNetwork, self).__init__(cfg)

        self.federated_model = federated_model

        self.client_id = client_id
        self.client_name = f'client-{client_id}'

        # vertical params
        self.vertical_input_size = cfg.vertical.vertical_inputs
        self.vertical_layers = cfg.vertical.layers
        self.vertical_units = cfg.vertical.units

        # progressive params
        self.progressive_layers = cfg.progressive.layers
        self.progressive_units = cfg.progressive.units

        # data params
        self.num_classes = cfg.data.num_classes

        # seed
        self.random_seed = eval(cfg.random.random_seed)

        self.strategy = None
        if distribute:
            self.strategy = tf.distribute.MirroredStrategy()
            with self.strategy.scope():
                self.network = self.create_network(freeze, vertical)
        else:
            self.network = self.create_network(freeze, vertical)

        self.procedure = PersonalizedNetworkLearn(self.network, cfg, self.strategy)

    def create_network(self, freeze=True, vertical=True):
        if vertical==True: self.network = self.PersonalizedVerticalNetwork(freeze)
        # else: self.model = self.PersonalizedCommonNetwork(federated_model, freeze)
        return self.network

    def learn(self, horizontal_inputs, vertical_inputs, labels, valid_data=None,
              verbose=1, project_path=None, save_path=None):

        self.procedure.learn(
            horizontal_inputs, vertical_inputs, labels, valid_data=None,
            verbose=verbose, project_path=project_path, save_path=save_path)

    def PersonalizedVerticalNetwork(self, freeze=True):
        h_net = self.HorizontalNet(freeze=freeze)
        v_net = self.VerticalNet()
        return self.PersonalizedNet(h_net, v_net)

    def PersonalizedNet(self, h_net, v_net):
        inputs = PersonalizedInput(
            units=self.progressive_units, activation='relu',
            c_input_shape=h_net.layers[0].output.shape[1],v_input_shape=v_net.layers[0].output.shape[1],
            name=f"progressive_dense_1", random_seed=self.random_seed
        )([h_net.layers[0].input, v_net.layers[0].input])

        dense = inputs
        for l in range(1, len(h_net.layers)-1):
            dense = PersonalizedDense(
                units=self.progressive_units, activation='relu', c_input_shape=h_net.layers[l].output.shape[1],
                v_input_shape=v_net.layers[l].output.shape[1],
                p_input_shape=dense.shape[1], name=f"progressive_dense_{l + 1}",
                random_seed=self.random_seed
            )([h_net.layers[l].output, v_net.layers[l].output, dense])
        
        outputs = layers.Dense(
            units=self.num_classes, activation='softmax',
            kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
            bias_initializer='zeros'
        )(dense)

        network = models.Model([h_net.layers[0].input, v_net.layers[0].input], outputs)
        for layer in network.layers:
            layer._name = f"{self.client_name}" + "_" + layer.name
        network._name = f"{self.client_name}-specific-personalized-model"
        
        return network

    def PersonalizedCommonNetwork(self, federated_model: models.Model, freeze=True):
        raise NotImplementedError

    def HorizontalNet(self, freeze=True):
        net = models.clone_model(self.federated_model)
        net.set_weights(self.federated_model.get_weights())

        if freeze==True:
            for layer in net.layers:
                layer.trainable = False

        return net

    def VerticalNet(self):
        inputs = layers.Input(shape=self.vertical_input_size, name="specific_vertical_input")
        dense = inputs

        for i in range(self.vertical_layers):
            dense = layers.Dense(
                units=self.vertical_units,
                activation='relu',
                kernel_initializer = initializers.glorot_uniform(seed=self.random_seed),
                bias_initializer='zeros',
                name = f'vertical_{i + 1}'
            )(dense)
        outputs = layers.Dense(
            1, activation="sigmoid", kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
            bias_initializer='zeros', name="vertical_classifier"
        )(dense)

        return models.Model(inputs, outputs, name=f"{self.client_name}_v_net")

if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from utils import gpu_utils
    from omegaconf import OmegaConf

    PROJECT_PATH = Path('.').absolute()
    gpu_utils.disable_tensorflow_debugging_logs()

    config = OmegaConf.load('./config/config.yaml')

    h_inputs_train = np.random.random((200, 5))
    h_inputs_valid = np.random.random((20, 5))

    v_inputs_train = np.random.random((200, 15))
    v_inputs_valid = np.random.random((20, 15))

    labels_train = np.random.binomial(1, 0.2, size=200)
    labels_valid = np.random.binomial(1, 0.2, size=20)

    lables_train = tf.keras.utils.to_categorical(labels_train, num_classes=2)
    lables_valid = tf.keras.utils.to_categorical(labels_valid, num_classes=2)


    mod = MLPNetwork(config, distribute=False)
    mod.model

    ppfl = PersonalizedProgressiveNetwork(config, mod.model, '1', freeze=True, vertical=True)
    # inference example
    ppfl.network([h_inputs_train, v_inputs_train])

    # train example
    ppfl.learn(
        h_inputs_train, v_inputs_train, lables_train,
        valid_data=[h_inputs_valid, v_inputs_valid, lables_valid],
        verbose=1, project_path=PROJECT_PATH, save_path=None
    )



