import numpy as np
from read_dataset import read_OPPT_dataset

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# OPPT_dataset
sensor_channels_required = ['IMU_BACK_ACC_X', 'IMU_BACK_ACC_Y', 'IMU_BACK_ACC_Z',
                            'IMU_BACK_GYRO_X', 'IMU_BACK_GYRO_Y', 'IMU_BACK_GYRO_Z',  # back
                            'IMU_RLA_ACC_X', 'IMU_RLA_ACC_Y', 'IMU_RLA_ACC_Z',
                            'IMU_RLA_GYRO_X', 'IMU_RLA_GYRO_Y', 'IMU_RLA_GYRO_Z']  # right lower arm

sensor_channels_required = ['IMU_RLA_ACC_X', 'IMU_RLA_ACC_Y', 'IMU_RLA_ACC_Z',
                            'IMU_RLA_GYRO_X', 'IMU_RLA_GYRO_Y', 'IMU_RLA_GYRO_Z']  # right lower arm

activity_list = ['Stand', 'Walk', 'Sit', 'Lie']

# activity_list = ['Open Door 1', 'Open Door 2', 'Close Door 1', 'Close Door 2',
#                 'Open Fridge', 'Close Fridge', 'Open Dishwasher', 'Close Dishwasher',
#                 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2', 'Close Drawer 2',
#                 'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup',
#                 'Toggle Switch']


activities_required = activity_list
source_user = 'S1'
target_user = 'S2'  # S3
Sampling_frequency = 30  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5
DATASET_NAME = 'OPPT'
oppt_ds = read_OPPT_dataset.READ_OPPT_DATASET(source_user, target_user, bag_window_second=Num_Seconds,
                                              bag_overlap_rate=Window_Overlap_Rate,
                                              instances_window_second=0.1, instances_overlap_rate=0.5,
                                              sampling_frequency=Sampling_frequency)

source_required_X_bags, source_required_Y_bags, source_required_amount, _, _, \
target_required_X_bags, target_required_Y_bags, target_required_amount, _, _, source_data_x, source_data_y, target_data_x, target_data_y \
    = oppt_ds.generate_data_with_required_sensor_channels_and_activities(sensor_channels_required, activities_required)


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def norm_mean(raw_list):
    aaa = np.mean(raw_list, axis=0)
    bbb = (np.max(raw_list, axis=0))
    ccc = (np.min(raw_list, axis=0))
    ddd = raw_list - aaa
    eee = bbb - ccc
    return ddd / eee

source_samples = norm_mean(np.array(source_data_x))
target_samples = norm_mean(np.array(target_data_x))
print()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# data persistent
with open(DATASET_NAME + '_all_' + str(source_user) + '_X_features.npy', 'wb') as f:
    # np.save(f, np.array(source_samples))  # work for DSADS
    np.save(f, np.array(source_samples, dtype=object))  # work for OPPT PAMAP2
with open(DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy', 'wb') as f:
    np.save(f, np.array(source_data_y))
with open(DATASET_NAME + '_all_' + str(target_user) + '_X_features.npy', 'wb') as f:
    # np.save(f, np.array(target_samples))  # work for DSADS
    np.save(f, np.array(target_samples, dtype=object))  # work for OPPT PAMAP2
with open(DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy', 'wb') as f:
    np.save(f, np.array(target_data_y))
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

for index, a_act in enumerate(activities_required):
    s_indices = [i for i, x in enumerate(source_data_y) if x == index]
    t_indices = [i for i, x in enumerate(target_data_y) if x == index]
    with open(DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_X_features.npy', 'wb') as f:
        # np.save(f, np.array(source_samples))  # work for DSADS
        np.save(f, np.array(source_samples[s_indices], dtype=object))  # work for OPPT PAMAP2
    with open(DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_Y_labels.npy', 'wb') as f:
        np.save(f, np.array(source_data_y[s_indices]))
    with open(DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_X_features.npy', 'wb') as f:
        # np.save(f, np.array(target_samples))  # work for DSADS
        np.save(f, np.array(target_samples[t_indices], dtype=object))  # work for OPPT PAMAP2
    with open(DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_Y_labels.npy', 'wb') as f:
        np.save(f, np.array(target_data_y[t_indices]))

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
