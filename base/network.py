import sys
sys.path.append('../FederatedLearningBase')
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics, regularizers
from base.learn import NetworkLearningProcedure, BaseNetworkLearn

class Network(object):
    def __init__(self, config: DictConfig):
        """
        Network class that creates and trains the tensorflow 2.x model and manages its parameters
        """
        self.model: models.Model = None
        self.strategy: tf.distribute.MirroredStrategy = None
        
    def create_network(self):
        raise NotImplementedError
    
    def learn(self):
        raise NotImplementedError
        
class MLPNetwork(Network):
    def __init__(self, config: DictConfig, distribute=False):
        super().__init__(config)
        # network param
        self.task: str = config.network.task
        self.input_size: int = config.network.input_size
        self.n_layers: int = config.network.n_layers
        self.n_hidden_units: int = config.network.n_hidden_units
        self.num_classes: int = config.network.num_classes
        self.random_seed: int = eval(config.random.random_seed)

        self.strategy = None
        if distribute:
            self.strategy = tf.distribute.MirroredStrategy()
            with self.strategy.scope():
                self.model: models.Model = self.create_network()
        else:
            self.model: models.Model = self.create_network()
        self.learning_procedure = BaseNetworkLearn(self.model, config, self.strategy)

    def create_network(self):
        """
        creates the MLP network
        :return: model: models.Model
        """
        # create input layer
        input_layer = layers.Input(shape=self.input_size, name="input")
        # create intermediate layer
        dense = input_layer
        for i in range(self.n_layers):
            dense = layers.Dense(
                units=self.n_hidden_units,
                kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                bias_initializer='zeros',
                activation='relu',
                name='intermediate_dense_{}'.format(i + 1)
            )(dense)

        output_layer = layers.Dense(self.num_classes,
                                    kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                                    bias_initializer='zeros',
                                    activation="linear" if self.task == 'regression' else 'softmax',
                                    name="regressor" if self.task == 'regression' else 'classifier')(dense)
        self.model = models.Model(input_layer, output_layer)
        return self.model

    def learn(self, inputs, labels, valid_data, verbose=True):
        self.learning_procedure.learn(inputs, labels, valid_data, verbose=verbose)

class ResNet9(Network):
    def __init__(self, config: DictConfig, learn_module: BaseNetworkLearn=BaseNetworkLearn, distribute=False):
        super().__init__(config, learn_module)
        # network param
        self.input_shape: int = eval(config.network.input_shape)
        self.num_classes: int = config.network.num_classes
        self.random_seed: int = config.random.random_seed
        self.l2_decay: float = config.network.l2_decay
        self.pool_pad: bool = config.network.pool_pad
        # Learn module

        self.strategy = None
        if distribute:
            self.strategy = tf.distribute.MirroredStrategy()
            with self.strategy.scope():
                self.model: models.Model = self.create_network()
        else:
            self.model: models.Model = self.create_network()
        self.learn_module = learn_module(self.model, config, self.strategy)

    def create_network(self):
        def conv_block(in_channels, out_channels, pool=False, pool_no=2, pool_pad=self.pool_pad):
            block = [layers.Conv2D(out_channels, kernel_size=(3, 3), padding='same', use_bias=False, strides=(1, 1),
                                   kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                                   kernel_regularizer=regularizers.l2(self.l2_decay)),
                     layers.ReLU()]
            if pool:
                if pool_pad:
                    block.append(layers.MaxPooling2D(pool_size=(pool_no, pool_no), padding='same'))
                else:
                    block.append(layers.MaxPooling2D(pool_size=(pool_no, pool_no)))
            return models.Sequential(block)

        inputs = layers.Input(shape=self.input_shape)
        out = conv_block(self.input_shape[-1], 64)(inputs)
        out = conv_block(64, 128, pool=True, pool_no=2, pool_pad=self.pool_pad)(out)
        out = models.Sequential([conv_block(128, 128), conv_block(128, 128)])(out) + out
        out = conv_block(128, 256, pool=True, pool_pad=self.pool_pad)(out)
        out = conv_block(256, 512, pool=True, pool_no=2, pool_pad=self.pool_pad)(out)
        out = models.Sequential([conv_block(512, 512), conv_block(512, 512)])(out) + out
        out = models.Sequential([layers.MaxPooling2D(pool_size=4), layers.Flatten(),
                                 layers.Dense(self.num_classes, use_bias=False, activation='softmax')])(out)
        self.model = models.Model(inputs=inputs, outputs=out)
        return self.model

    def learn(self, inputs, labels, valid_data, verbose=True):
        self.learn_module.learn(inputs, labels, valid_data, verbose=verbose)

if __name__ == "__main__":
    # disable tensorflow debugging
    from utils import gpu_utils
    gpu_utils.disable_tensorflow_debugging_logs()
    from omegaconf import OmegaConf
    config = OmegaConf.load('./config/cnn_config.yaml')

    mod = MLPNetwork(config, BaseNetworkLearn, distribute=False)
    mod.model

    MLPNetwork.learn(inputs, labels, valid_data)



    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_inputs, test_inputs = train_images / 255.0, test_images / 255.0
    train_inputs = train_inputs.reshape(-1, 28, 28, 1)
    test_inputs = test_inputs.reshape(-1, 28, 28, 1)
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    res = ResNet9(config, BaseNetworkLearn, distribute=False)
    res.learn(train_inputs, train_labels, [test_inputs, test_labels])

