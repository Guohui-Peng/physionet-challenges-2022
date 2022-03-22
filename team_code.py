#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.keras as keras
import tensorflow as tf
import librosa
import time
import warnings
warnings.filterwarnings("ignore")

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):

    training_resnet_mlp(data_folder, model_folder, verbose)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):    
    my_model_folder = model_folder + '/best_model'    
    new_model = keras.models.load_model(my_model_folder)
    return new_model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    PAD_LENGTH = 256
    # Load features.
    current_recording = get_wav_data(data, recordings, PAD_LENGTH)            
    current_recording = np.reshape(current_recording, (1, 128, PAD_LENGTH, 5))    
   
    current_features = get_features(data, recordings)
    current_features = np.reshape(current_features, (1, len(current_features))) 
    
    pred = model.predict([current_recording, current_features])

    classes = ['Present', 'Unknown', 'Absent']
    # Get classifier probabilities.
    pred_softmax = tf.nn.softmax(pred[0])
    pred_softmax = pred_softmax.numpy()
    probabilities = pred_softmax

    # Choose label with higher probability.
    labels = np.zeros(len(classes), dtype=np.int_)
    idx = np.argmax(pred_softmax)
    labels[idx] = 1
    
    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

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
    features = np.nan_to_num(features)
    return np.asarray(features, dtype=np.float32)


# ResNet+MLP
class RESNET_MLP:

    def __init__(self, output_directory, input_shape_a, input_shape_b, nb_classes, verbose=1, build=True):
        if not output_directory.endswith('/'):
            output_directory = output_directory + '/'
        self.output_directory = output_directory
        self.verbose = verbose
        if build == True:
            self.model = self.build_model(input_shape_a, input_shape_b, nb_classes)
            if (verbose > 1):
                self.model.summary()
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
        output_layer = keras.layers.Dense(nb_classes, activation='sigmoid')(combined)
        
        model = keras.models.Model(inputs=[input_a, input_b], outputs=output_layer)

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

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=5, verbose=self.verbose, 
            min_delta=0.0001, cooldown=0, min_lr=1e-7
        )

        early_stop = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=self.verbose, patience=10)

        # Save the model.
        file_path = self.output_directory+'best_model'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', mode='min', save_weights_only=False,
            save_best_only=True)

        self.callbacks = [reduce_lr, early_stop, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        if not tf.test.is_gpu_available:
            print('GPU is not available')
            # exit()        
        batch_size = 8
        nb_epochs = 1500

        if not x_val is None:
            self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=self.verbose>1, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:
            self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=self.verbose>1, callbacks=self.callbacks)

        self.model.save(self.output_directory + 'last_model',include_optimizer=False)
        keras.backend.clear_session()

    def predict(self, x_test):
        model_path = self.output_directory + 'best_model'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        return y_pred


# Get the wav data
def get_wav_data(data, recordings, padding=400, fs=4000):
    locations = get_locations(data)
    
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    
    recording_features = list()
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        r = np.zeros((num_recording_locations, 1))
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:                    
                    if r[j] == 0:
                        record = recordings[i] / float(tf.int16.max)
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



# Training with RESNET+MLP
def training_resnet_mlp(data_folder, model_folder, verbose=1):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')
    
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)
    
    if num_patient_files==0:
        raise Exception('No data was provided.')

    PAD_LENGTH = 256

    # # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    classes = ['Present', 'Unknown', 'Absent']
    num_classes = len(classes)
    features = list()
    recordings = list()
    labels = list()    

    for i in range(num_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract audio data.
        current_recording = get_wav_data(current_patient_data, current_recordings, PAD_LENGTH)        
        recording = current_recording
        recording = np.reshape(current_recording, (1, 128, PAD_LENGTH, 5))
        recordings.append(recording)

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

    recordings = np.vstack(recordings)
    features = np.vstack(features)
    labels = np.vstack(labels)
    
    X_train, X2_train, y_train = recordings, features, labels
    
    model = RESNET_MLP(model_folder, (128, PAD_LENGTH, 5), (26,), 3, verbose=verbose)
    if verbose >= 1:
        print('Training model...')    
    model.fit([X_train, X2_train], y_train)
