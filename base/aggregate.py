import numpy as np


class Aggregator(object):
    def __init__(self):
        pass

    def aggregate(self, n_sample, selected_client_n_k_list, selected_client_loss_list, selected_client_weight_list):
        raise NotImplementedError()


class FedAvg(Aggregator):
    def __init__(self):
        """
        Federated Averaging
        """
        super().__init__()

    def aggregate(self, n_sample, selected_client_n_k_list, selected_client_loss_list, selected_client_weight_list):
        scaled_term_list = [n_k_sample / n_sample for n_k_sample in selected_client_n_k_list]
        scaled_losses_list = [loss * scaled_term for loss, scaled_term in
                              zip(selected_client_loss_list, scaled_term_list)]
        scaled_weights_list = []
        for client_weight, scaled_term in zip(selected_client_weight_list, scaled_term_list):
            scaled_weights = [layer_weight * scaled_term for layer_weight in client_weight]
            scaled_weights_list.append(scaled_weights)

        weights_average = [np.zeros_like(w) for w in scaled_weights_list[0]]
        for layer_index in range(len(weights_average)):
            scaled_weights_layer_list = [client_scaled_weights[layer_index]
                                         for client_scaled_weights in scaled_weights_list]
            averaged_weights_layer = np.sum(scaled_weights_layer_list, axis=0)
            weights_average[layer_index] = averaged_weights_layer
        loss_average = sum(scaled_losses_list)
        return weights_average, loss_average
