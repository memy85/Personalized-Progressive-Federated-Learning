import tensorflow as tf
from tensorflow.keras import layers, activations, initializers

class PersonalizedInput(layers.Layer):
    def __init__(self, units, activation, c_input_shape, v_input_shape, random_seed=42, name=None, vertical=True):
        super(PersonalizedInput, self).__init__(name=name)
        self.vertical = vertical
        self.units = units
        self.activation = activations.get(activation)
        self.c_input_shape = c_input_shape
        self.v_input_shape = v_input_shape
        self.random_seed = random_seed
        self.c_lateral = self.add_weight(
            shape=(self.c_input_shape, self.units), initializer=initializers.glorot_uniform(seed=self.random_seed),
            trainable=True, name='c_lateral'
        )

        if self.vertical==True: self.v_lateral = self.add_weight(
            shape=(self.v_input_shape, self.units), initializer=initializers.glorot_uniform(seed=self.random_seed),
            trainable=True, name='v_lateral'
        )
        else: self.v_lateral = None

        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True, name='b')

    def get_config(self):
        config = super(PersonalizedInput, self).get_config().copy()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'c_input_shape': self.c_input_shape,
            'v_input_shape': self.v_input_shape,
            'random_seed': self.random_seed,
            'c_lateral': self.c_lateral,
            'v_lateral': self.v_lateral,
            'b': self.b
        })
        return config

    def call(self, inputs):
        if self.vertical == True:
            c_layer = inputs[0]
            v_layer = inputs[1]
            return self.activation(tf.matmul(c_layer, self.c_lateral) + tf.matmul(v_layer, self.v_lateral) + self.b)
        else:
            c_layer = inputs
            return self.activation(tf.matmul(c_layer, self.c_lateral) + self.b)

class PersonalizedDense(layers.Layer):

    def __init__(self, units, activation, c_input_shape, v_input_shape, p_input_shape, random_seed=42, vertical=True, name=None):
        super(PersonalizedDense, self).__init__(name=name)
        self.vertical = vertical
        self.units = units
        self.activation = activations.get(activation)
        self.c_input_shape = c_input_shape
        self.v_input_shape = v_input_shape
        self.p_input_shape = p_input_shape
        self.random_seed = random_seed
        self.c_lateral = self.add_weight(
            shape=(self.c_input_shape, self.units), initializer=initializers.glorot_uniform(seed=self.random_seed),
            trainable=True, name='c_lateral'
        )
        if self.vertical == True:
            self.v_lateral = self.add_weight(
                shape=(self.v_input_shape, self.units), initializer=initializers.glorot_uniform(seed=self.random_seed),
                trainable=True, name='v_lateral'
            )
        else:
            self.v_lateral = None
        self.p = self.add_weight(
            shape=(self.p_input_shape, self.units), initializer=initializers.glorot_uniform(seed=self.random_seed),
            trainable=True, name='p_lateral'
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True, name='b')

    def get_config(self):
        config = super(PersonalizedDense, self).get_config().copy()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'c_input_shape': self.c_input_shape,
            'v_input_shape': self.v_input_shape,
            'p_input_shape': self.p_input_shape,
            'random_seed': self.random_seed,
            'c_lateral': self.c_lateral,
            'v_lateral': self.v_lateral,
            'p': self.p,
            'b': self.b
        })
        return config

    def call(self, inputs):
        if self.vertical == True:
            c_layer = inputs[0]
            v_layer = inputs[1]
            p_layer = inputs[2]
            return self.activation(tf.matmul(c_layer, self.c_lateral) + tf.matmul(v_layer, self.v_lateral) + tf.matmul(p_layer, self.p) + self.b)
        else:
            c_layer = inputs[0]
            p_layer = inputs[1]
            return self.activation(tf.matmul(c_layer, self.c_lateral) + tf.matmul(p_layer, self.p) + self.b)
#
# if __name__ == "__main__":
#     input_layer = PersonalizedInput(
#         30, 'relu', 7, 10, None, vertical=True
#     )
#     dense_layer = PersonalizedDense(
#         30, 'relu', 7, 10, 10, None, vertical=True
#     )