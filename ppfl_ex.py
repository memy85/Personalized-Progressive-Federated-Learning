import hydra
import tensorflow as tf
from pathlib import Path
import numpy as np
from utils import gpu_utils
from omegaconf import OmegaConf, DictConfig
from base.network import MLPNetwork
from ppfl.personalization.network import PersonalizedProgressiveNetwork

PROJECT_PATH = Path('.').absolute()
gpu_utils.disable_tensorflow_debugging_logs()

def create_sample_data():
    h_inputs_train = np.random.random((200, 5))
    h_inputs_valid = np.random.random((20, 5))

    v_inputs_train = np.random.random((200, 15))
    v_inputs_valid = np.random.random((20, 15))

    labels_train = np.random.binomial(1, 0.2, size=200)
    labels_valid = np.random.binomial(1, 0.2, size=20)

    labels_train = tf.keras.utils.to_categorical(labels_train, num_classes=2)
    labels_valid = tf.keras.utils.to_categorical(labels_valid, num_classes=2)

    return h_inputs_train, h_inputs_valid, v_inputs_train, v_inputs_valid, labels_train, labels_valid

@hydra.main(config_path='./config', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    # cfg = OmegaConf.load('./cfg/cfg.yaml')
    h_inputs_train, h_inputs_valid, v_inputs_train, v_inputs_valid, labels_train, labels_valid \
        = create_sample_data()

    mod = MLPNetwork(cfg, distribute=False)
    mod.model

    ppfl = PersonalizedProgressiveNetwork(cfg, mod.model, '1', freeze=True, vertical=True)
    # inference example
    ppfl.network([h_inputs_train, v_inputs_train])

    # train example
    ppfl.learn(
        h_inputs_train, v_inputs_train, labels_train,
        valid_data=[h_inputs_valid, v_inputs_valid, labels_valid],
        verbose=1, project_path=PROJECT_PATH, save_path=None
    )

if __name__ == "__main__":
    main()