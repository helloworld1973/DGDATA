import time
from gen_model.alg.DeepGenTempRelaNet import DeepGenTempRelaNet, DeepGenTempRelaNet_Transformer
from gen_model.alg.linear_regreession import train_time_series_regression
from gen_model.alg.opt import *
from gen_model.alg import modelopera
from gen_model.utils.util import set_random_seed, log_and_print, print_row
from utils import draw_TSNE


def DeepGenTempRela_train(S_torch_loader, T_torch_loader, ST_torch_loader, global_epoch, local_epoch, num_classes,
                          num_temporal_states,
                          conv1_in_channels, conv1_out_channels, conv2_out_channels,
                          kernel_size_num, in_features_size, hidden_size, dis_hidden,
                          ReverseLayer_latent_domain_alpha, variance, alpha, beta, gamma, delta, epsilon,
                          lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta, file_name):
    set_random_seed(1234)

    best_valid_acc, target_acc = 0, 0

    algorithm = DeepGenTempRelaNet(conv1_in_channels, conv1_out_channels, conv2_out_channels, kernel_size_num,
                                   in_features_size,
                                   hidden_size, dis_hidden, num_classes, num_temporal_states,
                                   ReverseLayer_latent_domain_alpha, variance,
                                   alpha, beta, gamma, delta, epsilon)
    algorithm = algorithm
    algorithm.train()
    optt = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='DGTRN-temporal')
    opt = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                        nettype='DGTRN-final')
    opta = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='DGTRN-all')

    for round in range(global_epoch):
        print(f'\n========ROUND {round}========')
        log_and_print(f'\n========ROUND {round}========', filename=file_name)
        print('====Feature update====')
        log_and_print('====Feature update====', filename=file_name)
        loss_list = ['total', 'reconstruct', 'KL', 'classes', 'domains']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        print('====Latent domain characterization====')
        log_and_print('====Latent domain characterization====', filename=file_name)
        loss_list = ['total', 'reconstruct', 'KL', 'disc_classes', 'disc_domains', 'temporal']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_t(data, optt)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        algorithm.set_tlabel(ST_torch_loader, S_torch_loader, T_torch_loader)

        print('====Domain-invariant feature learning====')
        log_and_print('====Domain-invariant feature learning====', filename=file_name)

        loss_list = ['total', 'reconstruct', 'KL', 'source_classes', 'disc_domains', 'temporal']
        eval_dict = {'train': ['train_in'], 'valid': ['valid_in'], 'target': ['target_out'],
                     'target_cluster': ['target_out']}
        print_key = ['epoch']
        print_key.extend([item + '_loss' for item in loss_list])
        print_key.extend([item + '_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15, file_name=file_name)

        sss = time.time()
        for step in range(local_epoch):
            for ST_data in ST_torch_loader:
                for S_data in S_torch_loader:
                    step_vals = algorithm.update(ST_data, S_data, opt)

            results = {'epoch': step, }

            results['train_acc'] = modelopera.accuracy(
                algorithm, S_torch_loader, None)

            print('valid_acc___________________________________________________________________________________')
            log_and_print(
                'valid_acc___________________________________________________________________________________',
                filename=file_name)
            acc = modelopera.accuracy(algorithm, S_torch_loader, None)
            results['valid_acc'] = acc

            print('target_acc_#################################################################################')
            log_and_print(
                'target_acc_#################################################################################',
                filename=file_name)
            acc, cluster_acc = modelopera.accuracy_target_user(algorithm, T_torch_loader, S_torch_loader, None)
            results['target_acc'] = acc
            results['target_cluster_acc'] = cluster_acc

            for key in loss_list:
                results[key + '_loss'] = step_vals[key]
            if results['target_cluster_acc'] > best_valid_acc:
                best_valid_acc = results['target_cluster_acc']
                target_acc = results['target_cluster_acc']
            results['total_cost_time'] = time.time() - sss
            print_row([results[key] for key in print_key], colwidth=15, file_name=file_name)

    print(f'Target acc: {target_acc:.4f}')
    log_and_print(f'Target acc: {target_acc:.4f}', filename=file_name)

    return target_acc


def GPU_DeepGenTempRela_train(S_torch_loader, T_torch_loader, ST_torch_loader, global_epoch, local_epoch, num_classes,
                              num_temporal_states,
                              conv1_in_channels, conv1_out_channels, conv2_out_channels,
                              kernel_size_num, in_features_size, hidden_size, dis_hidden,
                              ReverseLayer_latent_domain_alpha, variance, alpha, beta, gamma, delta, epsilon,
                              lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta, file_name, device):
    set_random_seed(1234)

    best_valid_acc, target_acc = 0, 0

    algorithm = DeepGenTempRelaNet(conv1_in_channels, conv1_out_channels, conv2_out_channels, kernel_size_num,
                                   in_features_size,
                                   hidden_size, dis_hidden, num_classes, num_temporal_states,
                                   ReverseLayer_latent_domain_alpha, variance,
                                   alpha, beta, gamma, delta, epsilon)
    algorithm = algorithm.to(device)
    algorithm.train()
    optt = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='DGTRN-temporal')
    opt = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                        nettype='DGTRN-final')
    opta = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='DGTRN-all')

    for round in range(global_epoch):
        print(f'\n========ROUND {round}========')
        log_and_print(f'\n========ROUND {round}========', filename=file_name)
        print('====Feature update====')
        log_and_print('====Feature update====', filename=file_name)
        loss_list = ['total', 'reconstruct', 'KL', 'classes', 'domains']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        print('====Latent domain characterization====')
        log_and_print('====Latent domain characterization====', filename=file_name)
        loss_list = ['total', 'reconstruct', 'KL', 'disc_classes', 'disc_domains', 'temporal']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_t(data, optt)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        algorithm.GPU_set_tlabel(ST_torch_loader, S_torch_loader, T_torch_loader, device)

        print('====Domain-invariant feature learning====')
        log_and_print('====Domain-invariant feature learning====', filename=file_name)

        loss_list = ['total', 'reconstruct', 'KL', 'source_classes', 'disc_domains', 'temporal']
        eval_dict = {'train': ['train_in'], 'valid': ['valid_in'], 'target': ['target_out'],
                     'target_cluster': ['target_out']}
        print_key = ['epoch']
        print_key.extend([item + '_loss' for item in loss_list])
        print_key.extend([item + '_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15, file_name=file_name)

        sss = time.time()
        for step in range(local_epoch):
            for ST_data in ST_torch_loader:
                for S_data in S_torch_loader:
                    step_vals = algorithm.update(ST_data, S_data, opt)

            results = {'epoch': step, }

            results['train_acc'] = modelopera.GPU_accuracy(
                algorithm, S_torch_loader, None)

            print('valid_acc___________________________________________________________________________________')
            log_and_print(
                'valid_acc___________________________________________________________________________________',
                filename=file_name)
            acc = modelopera.GPU_accuracy(algorithm, S_torch_loader, None)
            results['valid_acc'] = acc

            print('target_acc_#################################################################################')
            log_and_print(
                'target_acc_#################################################################################',
                filename=file_name)
            acc, cluster_acc, cm = modelopera.GPU_accuracy_target_user(algorithm, T_torch_loader, S_torch_loader, None)
            results['target_acc'] = acc
            results['target_cluster_acc'] = cluster_acc
            results['cm'] = cm

            for key in loss_list:
                results[key + '_loss'] = step_vals[key]
            if results['target_cluster_acc'] > best_valid_acc:
                best_valid_acc = results['target_cluster_acc']
                target_acc = results['target_cluster_acc']
                best_cm = results['cm']
            results['total_cost_time'] = time.time() - sss
            print_row([results[key] for key in print_key], colwidth=15, file_name=file_name)

    print(f'Target acc: {target_acc:.4f}')
    print(best_cm)
    log_and_print(f'Target acc: {target_acc:.4f}', filename=file_name)

    return target_acc, best_cm


def GPU_DeepGenTempRela_with_AutoRegression_train(S_torch_loader, T_torch_loader, ST_torch_loader, global_epoch,
                                                  local_epoch, num_classes,
                                                  num_temporal_states,
                                                  conv1_in_channels, conv1_out_channels, conv2_out_channels,
                                                  kernel_size_num, in_features_size, hidden_size, dis_hidden,
                                                  ReverseLayer_latent_domain_alpha, variance, alpha, beta, gamma, delta,
                                                  epsilon,
                                                  lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                                                  file_name, device, temporal_lags):
    set_random_seed(1234)

    best_valid_acc, target_acc = 0, 0

    algorithm = DeepGenTempRelaNet(conv1_in_channels, conv1_out_channels, conv2_out_channels, kernel_size_num,
                                   in_features_size,
                                   hidden_size, dis_hidden, num_classes, num_temporal_states,
                                   ReverseLayer_latent_domain_alpha, variance,
                                   alpha, beta, gamma, delta, epsilon)
    algorithm = algorithm.to(device)
    algorithm.train()
    optt = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='DGTRN-temporal')
    opt = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                        nettype='DGTRN-final')
    opta = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='DGTRN-all')

    for round in range(global_epoch):
        S_mu = 0
        S_y = 0
        T_mu = 0
        T_y = 0
        print(f'\n========ROUND {round}========')
        log_and_print(f'\n========ROUND {round}========', filename=file_name)
        print('====Feature update====')
        log_and_print('====Feature update====', filename=file_name)
        loss_list = ['total', 'reconstruct', 'KL', 'classes', 'domains']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        print('====Auto Regression and get temporal coefficients====')
        log_and_print('====Auto Regression and get temporal coefficients====', filename=file_name)
        temporal_relation_weight = []
        features = []
        labels = []
        for param in algorithm.parameters():
            param.requires_grad = False
        for data in ST_torch_loader:
            all_x = data[0].float()
            all_c = data[1].long()
            labels = all_c
            features = algorithm.featurizer(all_x)

        # Create a dictionary to store features for each unique label
        features_dict = {}
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = labels == label
            features_dict[label.item()] = features[mask]
        model = train_time_series_regression(device, features_dict, window_size=temporal_lags, epochs=100, lr=0.01)
        # After training the model
        weights = model.scalar_weights.data.tolist()
        print("Coefficients:")
        log_and_print("Coefficients:", filename=file_name)
        for i, w in enumerate(weights, 1):
            print(f"Weight for t{i}: {w:.4f}")
            log_and_print(f"Weight for t{i}: {w:.4f}", filename=file_name)
            temporal_relation_weight.append(w)

        for param in algorithm.parameters():
            param.requires_grad = True

        print('====Latent domain characterization====')
        log_and_print('====Latent domain characterization====', filename=file_name)
        loss_list = ['total', 'reconstruct', 'KL', 'disc_classes', 'disc_domains', 'temporal']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_t(data, optt)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        algorithm.GPU_set_dlabel_temporal_relation(ST_torch_loader, S_torch_loader, T_torch_loader,
                                                   temporal_relation_weight, temporal_lags, device)

        print('====Domain-invariant feature learning====')
        log_and_print('====Domain-invariant feature learning====', filename=file_name)

        loss_list = ['total', 'reconstruct', 'KL', 'source_classes', 'disc_domains', 'temporal']
        eval_dict = {'train': ['train_in'], 'valid': ['valid_in'], 'target': ['target_out'],
                     'target_cluster': ['target_out']}
        print_key = ['epoch']
        print_key.extend([item + '_loss' for item in loss_list])
        print_key.extend([item + '_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15, file_name=file_name)

        sss = time.time()
        for step in range(local_epoch):
            for ST_data in ST_torch_loader:
                for S_data in S_torch_loader:
                    step_vals = algorithm.update(ST_data, S_data, opt)

            results = {'epoch': step, }

            results['train_acc'] = modelopera.GPU_accuracy(
                algorithm, S_torch_loader, None)

            print('valid_acc___________________________________________________________________________________')
            log_and_print(
                'valid_acc___________________________________________________________________________________',
                filename=file_name)
            acc, S_mu, S_y = modelopera.GPU_accuracy(algorithm, S_torch_loader, None)
            results['valid_acc'] = acc

            print('target_acc_#################################################################################')
            log_and_print(
                'target_acc_#################################################################################',
                filename=file_name)
            acc, cluster_acc, cm, T_mu, T_y = modelopera.GPU_accuracy_target_user(algorithm, T_torch_loader,
                                                                                  S_torch_loader, None)
            results['target_acc'] = acc
            results['target_cluster_acc'] = cluster_acc
            results['cm'] = cm

            for key in loss_list:
                results[key + '_loss'] = step_vals[key]
            if results['target_cluster_acc'] > best_valid_acc:
                best_valid_acc = results['target_cluster_acc']
                target_acc = results['target_cluster_acc']
                best_cm = results['cm']
            results['total_cost_time'] = time.time() - sss
            print_row([results[key] for key in print_key], colwidth=15, file_name=file_name)

        draw_TSNE(S_mu, S_y, T_mu, T_y, round)

    print(f'Target acc: {target_acc:.4f}')
    print(best_cm)
    log_and_print(f'Target acc: {target_acc:.4f}', filename=file_name)

    return target_acc, best_cm


def GPU_DeepGenTempRela_with_AutoRegression_transformer_train(S_torch_loader, T_torch_loader, ST_torch_loader,
                                                              global_epoch,
                                                              local_epoch, num_classes,
                                                              num_temporal_states,
                                                              embed_size, heads, num_layers, in_features_size,
                                                              hidden_size,
                                                              dis_hidden,
                                                              ReverseLayer_latent_domain_alpha, variance, alpha, beta,
                                                              gamma, delta,
                                                              epsilon,
                                                              lr_decay1, lr_decay2, lr, optim_Adam_weight_decay,
                                                              optim_Adam_beta,
                                                              file_name, device, temporal_lags):
    set_random_seed(1234)

    best_valid_acc, target_acc = 0, 0

    algorithm = DeepGenTempRelaNet_Transformer(embed_size, heads, num_layers, in_features_size,
                                               hidden_size, dis_hidden, num_classes, num_temporal_states,
                                               ReverseLayer_latent_domain_alpha, variance,
                                               alpha, beta, gamma, delta, epsilon)
    algorithm = algorithm.to(device)
    algorithm.train()
    optt = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='DGTRN-temporal')
    opt = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                        nettype='DGTRN-final')
    opta = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='DGTRN-all')

    for round in range(global_epoch):
        print(f'\n========ROUND {round}========')
        log_and_print(f'\n========ROUND {round}========', filename=file_name)
        print('====Feature update====')
        log_and_print('====Feature update====', filename=file_name)
        loss_list = ['total', 'reconstruct', 'KL', 'classes', 'domains']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        print('====Auto Regression and get temporal coefficients====')
        log_and_print('====Auto Regression and get temporal coefficients====', filename=file_name)
        temporal_relation_weight = []
        features = []
        labels = []
        for param in algorithm.parameters():
            param.requires_grad = False
        for data in ST_torch_loader:
            all_x = data[0].float()
            all_c = data[1].long()
            labels = all_c
            features = algorithm.featurizer(all_x)

        # Create a dictionary to store features for each unique label
        features_dict = {}
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = labels == label
            features_dict[label.item()] = features[mask]
        model = train_time_series_regression(device, features_dict, window_size=temporal_lags, epochs=100, lr=0.01)
        # After training the model
        weights = model.scalar_weights.data.tolist()
        print("Coefficients:")
        log_and_print("Coefficients:", filename=file_name)
        for i, w in enumerate(weights, 1):
            print(f"Weight for t{i}: {w:.4f}")
            log_and_print(f"Weight for t{i}: {w:.4f}", filename=file_name)
            temporal_relation_weight.append(w)

        for param in algorithm.parameters():
            param.requires_grad = True

        print('====Latent domain characterization====')
        log_and_print('====Latent domain characterization====', filename=file_name)
        loss_list = ['total', 'reconstruct', 'KL', 'disc_classes', 'disc_domains', 'temporal']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_t(data, optt)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        algorithm.GPU_set_dlabel_temporal_relation(ST_torch_loader, S_torch_loader, T_torch_loader,
                                                   temporal_relation_weight, temporal_lags, device)

        print('====Domain-invariant feature learning====')
        log_and_print('====Domain-invariant feature learning====', filename=file_name)

        loss_list = ['total', 'reconstruct', 'KL', 'source_classes', 'disc_domains', 'temporal']
        eval_dict = {'train': ['train_in'], 'valid': ['valid_in'], 'target': ['target_out'],
                     'target_cluster': ['target_out']}
        print_key = ['epoch']
        print_key.extend([item + '_loss' for item in loss_list])
        print_key.extend([item + '_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15, file_name=file_name)

        sss = time.time()
        for step in range(local_epoch):
            for ST_data in ST_torch_loader:
                for S_data in S_torch_loader:
                    step_vals = algorithm.update(ST_data, S_data, opt)

            results = {'epoch': step, }

            results['train_acc'] = modelopera.GPU_accuracy(
                algorithm, S_torch_loader, None)

            print('valid_acc___________________________________________________________________________________')
            log_and_print(
                'valid_acc___________________________________________________________________________________',
                filename=file_name)
            acc = modelopera.GPU_accuracy(algorithm, S_torch_loader, None)
            results['valid_acc'] = acc

            print('target_acc_#################################################################################')
            log_and_print(
                'target_acc_#################################################################################',
                filename=file_name)
            acc, cluster_acc = modelopera.GPU_accuracy_target_user(algorithm, T_torch_loader, S_torch_loader, None)
            results['target_acc'] = acc
            results['target_cluster_acc'] = cluster_acc

            for key in loss_list:
                results[key + '_loss'] = step_vals[key]
            if results['target_cluster_acc'] > best_valid_acc:
                best_valid_acc = results['target_cluster_acc']
                target_acc = results['target_cluster_acc']
            results['total_cost_time'] = time.time() - sss
            print_row([results[key] for key in print_key], colwidth=15, file_name=file_name)

    print(f'Target acc: {target_acc:.4f}')
    log_and_print(f'Target acc: {target_acc:.4f}', filename=file_name)

    return target_acc
