import numpy as np
from read_dataset import read_DSADS_dataset

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# PAMAP2_dataset
activity_list = ['sitting', 'standing', 'lying_on_back', 'lying_on_right', 'ascending_stairs', 'descending_stairs',
                 'standing_in_an_elevator_still', 'moving_around_in_an_elevator',
                 'walking_in_a_parking_lot', 'walking_on_a_treadmill_in_flat',
                 'walking_on_a_treadmill_inclined_positions',
                 'running_on_a_treadmill_in_flat', 'exercising on a stepper', 'exercising on a cross trainer',
                 'cycling_on_an_exercise_bike_in_horizontal_positions',
                 'cycling_on_an_exercise_bike_in_vertical_positions',
                 'rowing', 'jumping', 'playing_basketball']
activities_required = activity_list
sensor_channels_required = ['RA_x_acc', 'RA_y_acc', 'RA_z_acc',
                            'RA_x_gyro', 'RA_y_gyro', 'RA_z_gyro']
source_user = '4'  # 2,4,7
target_user = '7'
Sampling_frequency = 25  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5
DATASET_NAME = 'DSADS'
dsads_ds = read_DSADS_dataset.READ_DSADS_DATASET(source_user, target_user, bag_window_second=Num_Seconds,
                                                 bag_overlap_rate=Window_Overlap_Rate,
                                                 instances_window_second=0.1, instances_overlap_rate=0.5,
                                                 sampling_frequency=Sampling_frequency)


source_samples, source_Y, target_samples, target_Y = \
    dsads_ds.generate_data_with_required_sensor_channels_and_activities(sensor_channels_required, activities_required)


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# feature extraction
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def norm_mean(raw_list):
    aaa = np.mean(raw_list, axis=0)
    bbb = (np.max(raw_list, axis=0))
    ccc = (np.min(raw_list, axis=0))
    ddd = raw_list - aaa
    eee = bbb - ccc
    return ddd / eee


source_samples = norm_mean(np.array(source_samples))
target_samples = norm_mean(np.array(target_samples))
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# data persistent
with open(DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_features.npy', 'wb') as f:
    # np.save(f, np.array(source_samples))  # work for DSADS
    np.save(f, np.array(source_samples, dtype=object))  # work for OPPT PAMAP2
with open(DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy', 'wb') as f:
    np.save(f, np.array(source_Y))
with open(DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
        Window_Overlap_Rate) + '_X_features.npy', 'wb') as f:
    # np.save(f, np.array(target_samples))  # work for DSADS
    np.save(f, np.array(target_samples, dtype=object))  # work for OPPT PAMAP2
with open(DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy', 'wb') as f:
    np.save(f, np.array(target_Y))
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

for index, a_act in enumerate(activities_required):
    s_indices = [i for i, x in enumerate(source_Y) if x == index]
    t_indices = [i for i, x in enumerate(target_Y) if x == index]
    with open(DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_X_features.npy', 'wb') as f:
        # np.save(f, np.array(source_samples))  # work for DSADS
        np.save(f, np.array(source_samples[s_indices], dtype=object))  # work for OPPT PAMAP2
    with open(DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_Y_labels.npy', 'wb') as f:
        np.save(f, np.array(source_Y)[s_indices])
    with open(DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_X_features.npy', 'wb') as f:
        # np.save(f, np.array(target_samples))  # work for DSADS
        np.save(f, np.array(target_samples[t_indices], dtype=object))  # work for OPPT PAMAP2
    with open(DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_Y_labels.npy', 'wb') as f:
        np.save(f, np.array(target_Y)[t_indices])
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
print()
