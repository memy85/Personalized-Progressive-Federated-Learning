# Personalized Progressive Federated Learning with Client-Specific Vertical Features

This repository is an official Tensorflow 2 implementation of Personalized Progressive Federated Learning with Client-Specific Vertical Features

## Abstract


Federated learning has been collaboratively used to build a model without transmitting or exchanging raw data across the distributed clients. Specifically, personalized federated learning focuses on training personalized models to adapt to diverse data distributions among clients. On the other hand, horizontal federated learning enables clients to train a global model based on distributed samples of the same feature space, and vertical federated learning enables clients to train a global model based on distributed features of the same sample. However, conventional horizontal federated learning cannot leverage vertically partitioned features to increase model complexity, and vertical federated learning requires all clients to share a large number of overlapping sample-ids. In this paper, we propose the personalized progressive federated learning (PPFL) model, a multi-model personalized federated learning approach that allows the leveraging of clientspecific vertically partitioned features. In this approach, a personalized model that learns the client-specific vertical features, which can vary from client to client, is adopted after the federated learning model is trained using a common feature space for a number of individual clients. This architecture allows the model to be personalized to an individual client’s distribution while expanding its feature space. Furthermore, client-specific vertical features and their parameters are never transmitted, as they might not be utilized for federated learning due to the enhanced privacy issues. We empirically tested PPFL on a real-world public electronic health record dataset. The experimental results demonstrated that PPFL and its variants outperformed the other base models, by not only leveraging the common feature space from a number of clients, but also training individual clients and their feature spaces. Another important result is that the proposed model improved the robustness of the unseen data distribution.

The main contributions is that:
* We propose PPFL, which is a new approach for personalizing the federated learning model by leveraging client-specific vertical feature spaces. The vertical feature information, and their parameters, are never transmitted or exchanged.

* PPFL builds personalized models allow the learning of client-specific distributions from a globally learned FL model, by transmitting layer-wise knowledge to different network columns

* This approach allows (1) the FL model to improve its performance for individual clients, and (2) for robustness in estimating the non-trained distribution in the unseen datasets.

* PPFL was evaluated on the heterogeneously distributed ICU-Type of Physionet Challenge 2012 dataset.

## Environmental Setup

Please install packages from `requirements.txt` after creating your own environment with `python 3.7.x`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
