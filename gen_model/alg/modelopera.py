import numpy as np
import torch
from torch import softmax


def GPU_accuracy_target_user(network, t_loader, s_loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0
    total_cluster_accuracy = 0
    s_length = len(s_loader.dataset.tensors[1])

    confusion_matrix_all = 0

    network.eval()
    with torch.no_grad():
        for data in t_loader:
            x = data[0].float()
            y = data[1].long()
            y = y.cpu()
            if usedpredict == 'p':
                p, mean = network.predict(x)
                mean = mean.cpu()
                p = p.cpu()
            else:
                p = network.predict1(x)
                p = p.cpu()
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()

                # cluster in each temporal state first, then majority voting for one label in each cluster in each class
                class_list = y.numpy()
                class_unique_elements = np.unique(class_list)
                data2 = data[2].cpu()
                ts_list = data2.numpy().astype(int)
                ts_unique_elements = np.unique(ts_list)
                idx_list = t_loader.dataset.tensors[4].cpu().numpy()

                # Step 1: Initialize the confusion matrix
                confusion_matrix = np.zeros((class_unique_elements.shape[0], class_unique_elements.shape[0]))

                for a_class in range(len(class_unique_elements)):
                    for a_ts_label in range(len(ts_unique_elements)):
                        this_class_this_ts_idx_list = []
                        for index, class_name in enumerate(class_list):
                            if class_name == a_class and ts_list[index] == a_ts_label:
                                this_class_this_ts_idx_list.append(idx_list[index] - s_length)

                        if len(this_class_this_ts_idx_list) == 0:

                            print(str(a_class) + '_' + str(a_ts_label) + ': None list')
                        else:
                            this_class_this_ts_p = p[this_class_this_ts_idx_list]
                            this_class_this_ts_y = y[this_class_this_ts_idx_list]
                            this_class_this_ts_num = len(this_class_this_ts_idx_list)
                            this_class_this_ts_correct = (this_class_this_ts_p.argmax(1).eq(
                                this_class_this_ts_y).float()).sum().item() / this_class_this_ts_num

                            # calculate cluster accuracy
                            average_dis_per_class = torch.mean(this_class_this_ts_p, dim=0,
                                                               keepdim=True)
                            average_dis_per_class = torch.softmax(average_dis_per_class, dim=1)

                            # update confusion matrix
                            max_class = average_dis_per_class.argmax(1)
                            confusion_matrix[a_class][max_class.item()] += this_class_this_ts_num

                            total_cluster_accuracy += this_class_this_ts_num * (
                                average_dis_per_class.argmax(1).eq(a_class).float().numpy()[0])

                            # print(str(a_class) + '_' + str(a_ts_label) + ':' + str(this_class_this_ts_correct) + '___:' + str(average_dis_per_class) + '___:' + str(this_class_this_ts_num))

                confusion_matrix_all = confusion_matrix

            total += batch_weights.sum().item()

    network.train()

    return correct / total, total_cluster_accuracy / total, confusion_matrix_all, mean, y


def accuracy_target_user(network, loader, s_loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0
    total_cluster_accuracy = 0
    s_length = len(s_loader.dataset.tensors[1])

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].float()
            y = data[1].long()
            if usedpredict == 'p':
                p, mean = network.predict(x)
            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()

                class_list = y.numpy()
                class_unique_elements = np.unique(class_list)
                ts_list = data[2].numpy().astype(int)
                ts_unique_elements = np.unique(ts_list)
                idx_list = loader.dataset.tensors[4].numpy()

                for a_class in range(len(class_unique_elements)):
                    for a_ts_label in range(len(ts_unique_elements)):
                        this_class_this_ts_idx_list = []
                        for index, class_name in enumerate(class_list):
                            if class_name == a_class and ts_list[index] == a_ts_label:
                                this_class_this_ts_idx_list.append(idx_list[index] - s_length)

                        if len(this_class_this_ts_idx_list) == 0:
                            # print(str(a_class) + '_' + str(a_ts_label) + ': None list')
                        else:
                            this_class_this_ts_p = p[this_class_this_ts_idx_list]
                            this_class_this_ts_y = y[this_class_this_ts_idx_list]
                            this_class_this_ts_num = len(this_class_this_ts_idx_list)
                            this_class_this_ts_correct = (this_class_this_ts_p.argmax(1).eq(
                                this_class_this_ts_y).float()).sum().item() / this_class_this_ts_num

                            # calculate cluster accuracy
                            average_dis_per_class = torch.mean(this_class_this_ts_p, dim=0,
                                                               keepdim=True)
                            average_dis_per_class = torch.softmax(average_dis_per_class, dim=1)
                            total_cluster_accuracy += this_class_this_ts_num * (
                            average_dis_per_class.argmax(1).eq(a_class).float().numpy()[0])

                            # print(str(a_class) + '_' + str(a_ts_label) + ':' + str(this_class_this_ts_correct) + '___:' + str(average_dis_per_class) + '___:' + str(this_class_this_ts_num))
            total += batch_weights.sum().item()
    network.train()

    return correct / total, total_cluster_accuracy / total


def accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].float()
            y = data[1].long()
            if usedpredict == 'p':
                p, mean = network.predict(x)
            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()

            total += batch_weights.sum().item()
    network.train()

    return correct / total


def GPU_accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].float()
            y = data[1].long()
            y = y.cpu()
            if usedpredict == 'p':
                p, mean = network.predict(x)
                mean = mean.cpu()
                p = p.cpu()
            else:
                p = network.predict1(x)
                p = p.cpu()
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()

            total += batch_weights.sum().item()
    network.train()

    return correct / total, mean, y
