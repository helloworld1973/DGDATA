import math
import random
import numpy as np
import torch
from gtda.time_series import SlidingWindow
from gen_model.utils.util import log_and_print
from gen_model.train import GPU_DeepGenTempRela_train, GPU_DeepGenTempRela_with_AutoRegression_train
from utils import GPU_get_temporal_diff_train_data

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# OPPT_dataset
sensor_channels_required = ['IMU_RLA_ACC_X', 'IMU_RLA_ACC_Y', 'IMU_RLA_ACC_Z',
                            'IMU_RLA_GYRO_X', 'IMU_RLA_GYRO_Y', 'IMU_RLA_GYRO_Z']  # right lower arm
activity_list = ['Stand', 'Walk', 'Sit', 'Lie']
DATASET_NAME = 'OPPT'
activities_required = activity_list
source_user = 'S3'
target_user = 'S2'  # S1

Sampling_frequency = 30  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5


def sliding_window_seg(data_x, data_y):
    # same setting as M1, except for no feature extraction step
    sliding_bag = SlidingWindow(size=int(Sampling_frequency * Num_Seconds),
                                stride=int(Sampling_frequency * Num_Seconds * (1 - Window_Overlap_Rate)))
    X_bags = sliding_bag.fit_transform(data_x)
    Y_bags = sliding_bag.resample(data_y)  # last occur label
    Y_bags = Y_bags.tolist()

    return X_bags, Y_bags


S_data = []
S_label = []
T_data = []
T_label = []

for index, a_act in enumerate(activities_required):
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_X_features.npy', 'rb') as f:
        source_bags = np.load(f, allow_pickle=True)
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_Y_labels.npy', 'rb') as f:
        source_labels = np.load(f)
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_X_features.npy', 'rb') as f:
        target_bags = np.load(f, allow_pickle=True)
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_Y_labels.npy', 'rb') as f:
        target_labels = np.load(f)

    s_X_bags, s_Y_bags = sliding_window_seg(source_bags, source_labels)
    t_X_bags, t_Y_bags = sliding_window_seg(target_bags, target_labels)

    if index == 0:
        S_data = s_X_bags
        S_label = s_Y_bags
        T_data = t_X_bags
        T_label = t_Y_bags
    else:
        S_data = np.vstack((S_data, s_X_bags))
        S_label = S_label + s_Y_bags
        T_data = np.vstack((T_data, t_X_bags))
        T_label = T_label + t_Y_bags
print()
S_label = [int(x) for x in S_label]
T_label = [int(x) for x in T_label]
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# model training paras settings
num_D = 6
width = Sampling_frequency * Num_Seconds
Num_classes = 4
Epochs = 100
Local_epoch = 10
device = torch.device("cuda:2")
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# DGTSDA_temporal_diff model
Conv1_in_channels = num_D
Conv1_out_channels = 16
Conv2_out_channels = 32
Kernel_size_num = 9
In_features_size = Conv2_out_channels * math.floor(
    ((Num_Seconds * Sampling_frequency - Kernel_size_num + 1) / 2 - Kernel_size_num + 1) / 2)

Lr_decay1 = 1.0
Lr_decay2 = 1.0
Optim_Adam_weight_decay = 5e-4
Optim_Adam_beta = 0.5

Alpha = 1.0  # RECON_L
Beta = 10.0  # KLD_L
Delta = 1.0  # DOMAIN_L
Gamma = 30.0  # CLASS_L
Epsilon = 10.0  # TEMPORAL_L
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
file_name = str(DATASET_NAME) + '_' + str(source_user) + '_' + str(target_user) + '_with_autoregression_output.txt'
file_name_summary = str(DATASET_NAME) + '_' + str(source_user) + '_' + str(target_user) + '_with_autoregression_output_summary.txt'

for Lr_decay1 in [1.0, 0.8, 0.5]:  # 0.5 1.0
    for Lr_decay2 in [1.0, 0.8, 0.5]:
        for Optim_Adam_weight_decay in [5e-4, 5e-3, 5e-2, 5e-1]:
            for Optim_Adam_beta in [0.2, 0.5, 0.9]:
                for Hidden_size in [100, 80, 50]:
                    for Dis_hidden in [50, 30, 20]:
                        for ReverseLayer_latent_domain_alpha in [0.2, 0.15, 0.25, 0.3, 0.1, 0.35]:
                            for lr in [1e-3, 1e-4, 1e-2, 1e-5, 1e-1, 1e-6]:
                                for Variance in [1, 2, 0.7, 3, 0.4, 4, 5]:
                                    for Num_temporal_states in [2, 3, 4, 5, 6, 7]:
                                        for temporal_lags in [2, 3, 4, 5, 6, 7]:
                                            print('para_setting:' + str(Num_temporal_states) + '_' + str(
                                                Hidden_size) + '_' + str(Dis_hidden) + '_' + str(
                                                Lr_decay1) + '_' + str(Lr_decay2) + '_' + str(
                                                Optim_Adam_weight_decay) + '_' + str(Optim_Adam_beta) + '_' + str(
                                                temporal_lags) + '_' + str(Variance) + '_' + str(lr) + '_' + str(
                                                ReverseLayer_latent_domain_alpha))
                                            log_and_print(
                                                content='para_setting:' + str(Num_temporal_states) + '_' + str(
                                                    Hidden_size) + '_' + str(Dis_hidden) + '_' + str(
                                                    Lr_decay1) + '_' + str(Lr_decay2) + '_' + str(
                                                    Optim_Adam_weight_decay) + '_' + str(Optim_Adam_beta) + '_' + str(
                                                    temporal_lags) + '_' + str(Variance) + '_' + str(lr) + '_' + str(
                                                    ReverseLayer_latent_domain_alpha), filename=file_name)

                                            '''
                                            S_torch_loader, T_torch_loader, ST_torch_loader = get_temporal_diff_train_data(
                                                S_data, S_label, T_data, T_label,
                                                batch_size=10000, num_D=num_D,
                                                width=width,
                                                num_class=Num_classes)
                                            '''

                                            S_torch_loader, T_torch_loader, ST_torch_loader = GPU_get_temporal_diff_train_data(
                                                S_data, S_label, T_data, T_label,
                                                batch_size=10000, num_D=num_D,
                                                width=width,
                                                num_class=Num_classes, device=device)

                                            target_acc = GPU_DeepGenTempRela_with_AutoRegression_train(S_torch_loader,
                                                                                                       T_torch_loader,
                                                                                                       ST_torch_loader,
                                                                                                       global_epoch=Epochs,
                                                                                                       local_epoch=Local_epoch,
                                                                                                       num_classes=Num_classes,
                                                                                                       num_temporal_states=Num_temporal_states,
                                                                                                       conv1_in_channels=Conv1_in_channels,
                                                                                                       conv1_out_channels=Conv1_out_channels,
                                                                                                       conv2_out_channels=Conv2_out_channels,
                                                                                                       kernel_size_num=Kernel_size_num,
                                                                                                       in_features_size=In_features_size,
                                                                                                       hidden_size=Hidden_size,
                                                                                                       dis_hidden=Dis_hidden,
                                                                                                       ReverseLayer_latent_domain_alpha=ReverseLayer_latent_domain_alpha,
                                                                                                       variance=Variance,
                                                                                                       alpha=Alpha,
                                                                                                       beta=Beta,
                                                                                                       gamma=Gamma,
                                                                                                       delta=Delta,
                                                                                                       epsilon=Epsilon,
                                                                                                       lr_decay1=Lr_decay1,
                                                                                                       lr_decay2=Lr_decay2,
                                                                                                       lr=lr,
                                                                                                       optim_Adam_weight_decay=Optim_Adam_weight_decay,
                                                                                                       optim_Adam_beta=Optim_Adam_beta,
                                                                                                       file_name=file_name,
                                                                                                       device=device,
                                                                                                       temporal_lags=temporal_lags)

                                            print()
                                            log_and_print(
                                                content='para_setting:' + str(Num_temporal_states) + '_' + str(
                                                    Hidden_size) + '_' + str(Dis_hidden) + '_' + str(
                                                    Lr_decay1) + '_' + str(Lr_decay2) + '_' + str(
                                                    Optim_Adam_weight_decay) + '_' + str(Optim_Adam_beta) + '_' + str(
                                                    temporal_lags) + '_' + str(Variance) + '_' + str(lr) + '_' + str(
                                                    ReverseLayer_latent_domain_alpha), filename=file_name_summary)
                                            log_and_print(
                                                content='best target acc:' + str(target_acc),
                                                filename=file_name_summary)
                                            log_and_print(
                                                content='-------------------------------------------------',
                                                filename=file_name_summary)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
