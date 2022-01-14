import gc
from typing import List, Dict

import keras.backend
import numpy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics
from base.network import *
from base.client import *
from base.learn import *
from personalization.network import *

class PersonalizedClient(Client):
    def __init__(self, client_id, inputs, labels):
        """
        Client class for Personalized Progressive Federated Learning
        This class for controlling individual client's network and information.
        Specifically, it creates and learns a client-specific personalized model.
        """
        super(PersonalizedClient, self).__init__(client_id, inputs, labels)
        self.personalized_net: PersonalizedProgressiveNetwork = None

    def create_personalized_network(self, config, network_module: PersonalizedProgressiveNetwork, learn_module: NetworkLearningProcedure,
                                    freeze=True, vertical=True,
                                    ):
        self.personalized_net = network_module(config, learn_module, self.client_net, self.client_id, freeze, vertical)

    def personalized_learn(self, vertical):
        self.personalized_net.learn()