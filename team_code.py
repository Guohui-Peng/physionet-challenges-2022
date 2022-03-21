#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import tensorflow.keras as keras
import tensorflow as tf
import librosa
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import time, datetime

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')


    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    classes = ['Present', 'Unknown', 'Absent']
    num_classes = len(classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    labels = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract features.
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(num_classes, dtype=int)
        label = get_label(current_patient_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
        labels.append(current_labels)
    
    features = np.vstack(features)
    labels = np.vstack(labels)

    # Train the model.
    if verbose >= 1:
        print('Training model...')

    # Define parameters for random forest classifier.
    n_estimators = 10    # Number of trees in the forest.
    max_leaf_nodes = 100 # Maximum number of leaf nodes in each tree.
    random_state = 123   # Random state; set for reproducibility.

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    # Save the model.
    save_challenge_model(model_folder, classes, imputer, classifier)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    classes = model['classes']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    features = get_features(data, recordings)

    # Impute missing data.
    features = features.reshape(1, -1)
    features = imputer.transform(features)

    # Get classifier probabilities.
    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose label with higher probability.
    labels = np.zeros(len(classes), dtype=np.int_)
    idx = np.argmax(probabilities)
    labels[idx] = 1

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, classes, imputer, classifier):
    d = {'classes': classes, 'imputer': imputer, 'classifier': classifier}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)


# 直接三分类，使用CONV2D+MLP
class RESNET_MLP_C3:

    def __init__(self, output_directory, input_shape_a, input_shape_b, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape_a, input_shape_b, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape_a, input_shape_b, nb_classes):
        n_feature_maps = 64

        input_a = keras.layers.Input(input_shape_a)
        input_b = keras.layers.Input(input_shape_b)

        # BLOCK 1

        conv_x = keras.layers.Conv2D(filters=n_feature_maps, kernel_size=8, padding='same')(input_a)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv2D(filters=n_feature_maps, kernel_size=1, padding='same')(input_a)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling2D()(output_block_3)
        model_a = keras.models.Model(inputs=input_a, outputs=gap_layer)

        ds2 = keras.layers.Dense(54, activation='relu')(input_b)
        ds2 = keras.layers.Dense(4, activation='sigmoid')(ds2)
        model2 = keras.Model(inputs=input_b, outputs=ds2)

        combined = keras.layers.concatenate([model_a.output, model2.output])
        # output_layer = keras.layers.Dense(nb_classes, activation='softmax')(combined)
        output_layer = keras.layers.Dense(nb_classes, activation='sigmoid')(combined)
        
        
        model = keras.models.Model(inputs=[input_a, input_b], outputs=output_layer)

        # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
        #               metrics=['accuracy'])

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer = keras.optimizers.Adam(), 
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', dtype=None, threshold=0.5), 
                    keras.metrics.Recall(name='Recall'), keras.metrics.Precision(name='Precision'), 
                    keras.metrics.AUC(
                        num_thresholds=200,
                        curve="ROC",
                        summation_method="interpolation",
                        name="AUC",
                        dtype=None,
                        thresholds=None,
                        multi_label=True,
                        label_weights=None)
            ])
        
        # model.compile(loss='mse', optimizer = keras.optimizers.Adam(), 
        #     metrics=['accuracy', 'mse', keras.metrics.Recall(name='Recall'), keras.metrics.Precision(name='Precision'), 
        #             keras.metrics.AUC(
        #                 num_thresholds=200,
        #                 curve="ROC",
        #                 summation_method="interpolation",
        #                 name="AUC",
        #                 dtype=None,
        #                 thresholds=None,
        #                 multi_label=False,
        #                 label_weights=None)
        #     ])

        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        # file_path = self.output_directory + 'resnet_best_model.hdf5'

        # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        #                                                    save_best_only=True)

        # self.callbacks = [reduce_lr, model_checkpoint]

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=5, verbose=2, 
            # mode='max',
            min_delta=0.0001, cooldown=0, min_lr=1e-7
        )

        early_stop = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
        file_path = self.output_directory+'best_model.hdf5'
        log_dir = "/physionet/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_AUC', mode='max',
            save_best_only=True)

        self.callbacks = [reduce_lr, early_stop, model_checkpoint, tensorboard_callback]

        return model

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        if not tf.test.is_gpu_available:
            print('GPU is not available')
            # exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 8
        nb_epochs = 1500

        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        # start_time = time.time()
        if not x_val is None:
            hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:
            hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=self.verbose, callbacks=self.callbacks)

        # duration = time.time() - start_time

        self.model.save(self.output_directory + 'resnet_last_model.hdf5')

        # y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
        #                       return_df_metrics=False)

        # save predictions
        # np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        # y_pred = np.argmax(y_pred, axis=1)

        # df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        # return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        # if return_df_metrics:
        #     y_pred = np.argmax(y_pred, axis=1)
        #     # df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        #     return df_metrics
        # else:
        #     test_duration = time.time() - start_time
        #     # save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
        return y_pred


# Get the recordings of 2022
def get_training_data(data, recordings, padding=400, fs=4000):
    locations = get_locations(data)
    
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    # recording_features = np.zeros((num_recording_locations, 1), dtype=float)
    
    recording_features = list()
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        r = np.zeros((num_recording_locations, 1))
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    # recording_features[j] = [recordings[i]]
                    if r[j] == 0:
                        record = recordings[i] / float(tf.int16.max)
                        # MFCC
                        # n_nfcc常用13或24，或默认的20
                        # record = librosa.feature.mfcc(y=record,sr=fs,n_mfcc=24)
                        
                        # Downsample
                        # record = signal.resample_poly(record, 1000, fs)
                        
                        # Log-Mel Spectrogram特征
                        record = librosa.feature.melspectrogram(record, sr=1000, n_fft=1024, hop_length=512, n_mels=128)
                        record = librosa.power_to_db(record)
                        record = keras.preprocessing.sequence.pad_sequences(record, maxlen=padding, truncating='post',padding="post") 

                        recording_features.append(record)
                    r[j] = 1
    num_recording_features = len(recording_features)
    if num_recording_features < num_recording_locations:
        for i in range(num_recording_locations-num_recording_features):
            recording_features.append(np.zeros((128, padding)))
    return recording_features   


# 多分类，直接训练为三分类+MLP
def training_resnet_mlp(data_folder):
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)
    print(num_patient_files)

    PAD_LENGTH = 300

    # label分析
    classes = ['Present', 'Unknown', 'Absent']
    num_classes = len(classes)
    features = list()
    recordings = list()
    labels = list()

    for i in range(num_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract features.
                
        current_recording = get_training_data(current_patient_data, current_recordings, PAD_LENGTH)
        # r = np.asarray(current_recordings[0], dtype=np.float64)
        recording = current_recording
        recording = np.reshape(current_recording, (1, 128, PAD_LENGTH, 5))
        recordings.append(recording)
        # print(recording)
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(num_classes, dtype=int)
        label = get_label(current_patient_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
        labels.append(current_labels)
    # features = features.reshape(features.shape[0],features.shape[1],1)
    recordings = np.vstack(recordings)
    features = np.vstack(features)
    labels = np.vstack(labels)
    print(recordings.shape)
    print(features.shape)
    # print(features)
    # print(features[0])
    # recordings = np.reshape(recordings, (labels.shape[0], features.shape[1], features.shape[2], 5))
    
    # print(features)
    # print(recordings.shape)
    # print(labels)
    print(labels.shape)

    # random
    random_seed = 20
    random.seed(random_seed)
    random.shuffle(recordings)
    random.seed(random_seed)
    random.shuffle(features)
    random.seed(random_seed)
    random.shuffle(labels)

    # X = [recordings,features]
    # X_pd = pd.DataFrame(X)

    X_train, y_train, X_test, y_test = [recordings[:753],features[:753]], labels[:753],[recordings[753:],features[753:]], labels[753:]

    # X_train, X_test, X2_train, X2_test, y_train, y_test = train_test_split(recordings, features, labels, test_size=0.2, random_state=20)

    # print(len(X_train))
    # print(len(X_test))
    # print(len(X2_train))
    # print(len(X2_test))
    # print(len(y_train))
    # print(len(y_test))
    # print(X_train)
    
    model = RESNET_MLP_C3('model/resnet/', (128, PAD_LENGTH, 5), (26,), 3, verbose=True)
    model.fit(X_train, y_train, X_test, y_test)
    # model.fit([X_train, X2_train], y_train, [X_test, X2_test], y_test)
    