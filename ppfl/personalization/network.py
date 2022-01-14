from typing import Dict
from omegaconf import DictConfig
from base.network import Network
from learn import PersonalizedNetworkLearn
from tensorflow.keras import layers, models, initializers
from personalize_layers import *


class PersonalizedProgressiveNetwork(Network):
    def __init__(self, config: DictConfig, learn_module: PersonalizedNetworkLearn, federated_model: models.Model,
                 client_id: str, freeze=True, vertical=True):

        super(PersonalizedProgressiveNetwork, self).__init__(config, learn_module)
        self.client_id = client_id
        # vertical params
        self.vertical_input_size = config.personalized.vertical.client_vertical_input_size
        self.n_vertical_layers = config.personalized.vertical.layers # same with base network layers
        self.n_vertical_units = config.personalized.vertical.units
        # progressive params
        self.n_personalized_layers = config.personalized.network.layers # same with base network layers
        self.n_personalized_units = config.personalized.network.units
        self.num_classes: int = config.network.network.num_classes # TODO config network에?
        self.random_seed = config.random.random_seed

        self.network = self.create_network(federated_model, freeze, vertical)
        self.learn_module = learn_module(self.network, config)

        self.result_state: Dict = None

    def create_network(self, federated_model: models.Model, freeze=True, vertical=True):
        if vertical==True: self.network = self.PersonalizedVerticalNetwork(federated_model, freeze)
        else: self.network = self.PersonalizedCommonNetwork(federated_model, freeze)
        return self.network

    def learn(self, common_inputs, vertical_inputs, labels, valid_data, verbose, vertical=True):
        self.learn_module.learn(common_inputs, vertical_inputs, labels, valid_data, verbose, vertical=True)

    def PersonalizedVerticalNetwork(self, federated_model: models.Model, freeze=True):
        vertical_net = self.ClientSpecificVerticalNetwork()
        common_net = models.clone_model(federated_model)
        common_net.set_weights(federated_model.get_weights())

        if freeze==True:
            for layer in common_net.layers:
                layer.trainable = False

        p_inputs = PersonalizedInput(units=self.n_personalized_units, activation='relu',
                                    c_input_shape=common_net.layers[0].output.shape[1],
                                    v_input_shape=vertical_net.layers[0].output.shape[1],
                                    name=f"progressive_dense_1", random_seed=self.random_seed
                                    )([common_net.layers[0].input, vertical_net[0].input])

        p_dense = p_inputs
        for l in range(1, len(vertical_net.layers)-1):
            p_dense = PersonalizedDense(units=self.n_personalized_units, activation='relu',
                                       c_input_shape=common_net.layers[l].output.shape[1],
                                       v_input_shape=vertical_net.layers[l].output.shape[1],
                                       p_input_shape=p_dense.shape[1], name=f"progressive_dense_{l + 1}",
                                       random_seed=self.random_seed
                                       )([common_net.layers[l].output, vertical_net.layers[l].output, p_dense])
        outputs = layers.Dense(units=self.num_classes, activation='softmax',
                               kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                               bias_initializer='zeros',
                               )
        network = models.Model([common_net.layers[0].input, vertical_net.layers[0].input], outputs)
        for layer in network.layers:
            layer._name = f"{self.client_name}"+"_"+layer.name
        network._name = f"{self.client_name}-specific-personalized-model"
        return network

    def PersonalizedCommonNetwork(self, federated_model: models.Model, freeze=True):
        common_net = models.clone_model(federated_model)
        common_net.set_weights(federated_model.get_weights())

        if freeze==True:
            for layer in common_net.layers:
                layer.trainable = False

        p_inputs = PersonalizedInput(units=self.n_personalized_units, activation='relu',
                                    c_input_shape=common_net.layers[0].output.shape[1],
                                    v_input_shape=None,
                                    name=f"progressive_dense_1", random_seed=self.random_seed, vertical=False
                                    )(common_net.layers[0].input)

        p_dense = p_inputs
        for l in range(1, len(common_net.layers)-1):
            p_dense = PersonalizedDense(units=self.n_personalized_units, activation='relu',
                                       c_input_shape=common_net.layers[l].output.shape[1],
                                       v_input_shape=None,
                                       p_input_shape=p_dense.shape[1], name=f"progressive_dense_{l + 1}",
                                       random_seed=self.random_seed, vertical=False
                                       )([common_net.layers[l].output, p_dense])
        outputs = layers.Dense(units=self.num_classes, activation='softmax',
                               kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                               bias_initializer='zeros',
                               )
        network = models.Model(common_net.layers[0].input, outputs)
        for layer in network.layers:
            layer._name = f"{self.client_name}"+"_"+layer.name
        network._name = f"{self.client_name}-specific-personalized-model"
        return network

    def ClientSpecificVerticalNetwork(self):
        inputs = layers.Input(shape=self.vertical_input_size, name="specific_vertical_input")
        dense = inputs
        for i in range(self.n_vertical_layers):
            dense = layers.Dense(
                units=self.n_vertical_units,
                activation='relu',
                kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                name=f'specific_dense_{i + 1}'
            )(dense)
        outputs = layers.Dense(1, activation="sigmoid",
                               kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                               bias_initializer='zeros',
                               name="specific_classifier")(dense)
        network = models.Model(inputs, outputs, name=f"{self.client_name}_specific_vertical_model")
        return network

