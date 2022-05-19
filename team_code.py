#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, os, joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.keras as keras
import tensorflow as tf
import keras.backend as K
import librosa
import random

from sklearn.impute import SimpleImputer
# import warnings
# warnings.filterwarnings("ignore")

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################
PAD_LENGTH = 256
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

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    features = list()
    murmurs = list()
    outcomes = list()

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
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)

    features = np.vstack(features)
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)

    # Train the model.
    if verbose >= 1:
        print('Training model...')

    # Define parameters for random forest classifier.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 45   # Maximum number of leaf nodes in each tree.
    random_state   = 6789 # Random state; set for reproducibility.

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    murmur_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, murmurs)
    outcome_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes)

    # Save the model.
    save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier)

    if verbose >= 1:
        print('Done.')
# def train_challenge_model(data_folder, model_folder, verbose):

#     training_resnet_mlp(data_folder, model_folder, verbose)

#     if verbose >= 1:
#         print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)
# def load_challenge_model(model_folder, verbose):
#     my_model_folder = os.path.join(model_folder, 'best_model')
    
#     new_model = RESNET_MLP(input_shape=[(128,PAD_LENGTH,5),(26,)],nb_classes=3,verbose=verbose).build_model()
#     new_model.load_weights(my_model_folder).expect_partial()
#     return new_model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
# def run_challenge_model(model, data, recordings, verbose):    
#     # Load features.
#     current_recording = get_wav_data(data, recordings, PAD_LENGTH)
#     current_recording = np.asarray(current_recording, dtype=np.float32)
#     current_recording = np.reshape(current_recording, (1, 128, PAD_LENGTH, 5))    
   
#     current_features = get_features(data, recordings)
#     current_features = np.reshape(current_features, (1, len(current_features))) 
    
#     pred = model.predict([current_recording, current_features])

#     classes = ['Present', 'Unknown', 'Absent']
#     # Get classifier probabilities.
#     pred_softmax = tf.nn.softmax(pred[0])
#     pred_softmax = pred_softmax.numpy()
#     probabilities = pred_softmax

#     # Choose label with higher probability.
#     labels = np.zeros(len(classes), dtype=np.int_)
#     idx = np.argmax(pred_softmax)
#     labels[idx] = 1
    
#     return classes, labels, probabilities
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    imputer = model['imputer']
    murmur_classes = model['murmur_classes']
    murmur_classifier = model['murmur_classifier']
    outcome_classes = model['outcome_classes']
    outcome_classifier = model['outcome_classifier']

    # Load features.
    features = get_features(data, recordings)

    # Impute missing data.
    features = features.reshape(1, -1)
    features = imputer.transform(features)

    # Get classifier probabilities.
    murmur_probabilities = murmur_classifier.predict_proba(features)
    murmur_probabilities = np.asarray(murmur_probabilities, dtype=np.float32)[:, 0, 1]
    outcome_probabilities = outcome_classifier.predict_proba(features)
    outcome_probabilities = np.asarray(outcome_probabilities, dtype=np.float32)[:, 0, 1]

    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier):
    d = {'imputer': imputer, 'murmur_classes': murmur_classes, 'murmur_classifier': murmur_classifier, 'outcome_classes': outcome_classes, 'outcome_classifier': outcome_classifier}
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
                        record = librosa.feature.melspectrogram(y=record, sr=1000, n_fft=1024, hop_length=512, n_mels=128)
                        record = librosa.power_to_db(record)
                        record = keras.preprocessing.sequence.pad_sequences(record, maxlen=padding, truncating='post',padding="post",dtype=float)
                        recording_features.append(record)
                    r[j] = 1
    num_recording_features = len(recording_features)
    if num_recording_features < num_recording_locations:
        for i in range(num_recording_locations-num_recording_features):
            recording_features.append(np.zeros((128, padding)))
    return recording_features   

def get_murmur_locations(data):
    label = 'nan'
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            try:
                label = l.split(': ')[1]
            except:
                pass
    if label is None:
        raise ValueError('No label available. Is your code trying to load labels from the hidden data?')
    return label

# Get the wav data for training
def get_wav_data_training(data, recordings, padding=400, fs=4000):
    locations = get_locations(data)
    
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)

    murmur_locations = get_murmur_locations(data)
    has_murmur = not compare_strings(murmur_locations, 'nan')
    m_locs = murmur_locations.split('+')
    
    recording_features = list()
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        r = np.zeros((num_recording_locations, 1))
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    if r[j] == 0:
                        if has_murmur: 
                            if locations[i] in m_locs:
                                record = recordings[i] / float(tf.int16.max)
                                record = librosa.feature.melspectrogram(record, sr=1000, n_fft=1024, hop_length=512, n_mels=128)
                                record = librosa.power_to_db(record)
                                record = keras.preprocessing.sequence.pad_sequences(record, maxlen=padding, truncating='post',padding="post",dtype=float) 
                            else:
                                record = np.zeros((128, padding))
                        else:
                            record = recordings[i] / float(tf.int16.max)
                            record = librosa.feature.melspectrogram(record, sr=1000, n_fft=1024, hop_length=512, n_mels=128)
                            record = librosa.power_to_db(record)
                            record = keras.preprocessing.sequence.pad_sequences(record, maxlen=padding, truncating='post',padding="post",dtype=float) 
                        
                        recording_features.append(record)
                    r[j] = 1
    num_recording_features = len(recording_features)
    if num_recording_features < num_recording_locations:
        for i in range(num_recording_locations-num_recording_features):
            recording_features.append(np.zeros((128, padding)))
    return recording_features

# Load data for training or test
def get_data(classes, patient_files, pad_length, data_folder, get_training_func):
    num_patient_files = len(patient_files)
    num_classes = len(classes)
    features = list()
    recordings = list()
    labels = list()

    for i in range(num_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract features.
        current_recording = get_training_func(current_patient_data, current_recordings, pad_length)
        recording = current_recording
        recording = np.reshape(current_recording, (1, 128, pad_length, 5))
        recordings.append(recording)
        # print(recording)
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(num_classes, dtype=int)
        label = get_murmur(current_patient_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
        labels.append(current_labels)
    recordings = np.vstack(recordings)
    features = np.vstack(features)
    labels = np.vstack(labels)
    return recordings,features,labels


################################################################################
#
# Models
#
################################################################################
class RESNET_Block(keras.layers.Layer):
    """
    Custom ResNet Block
    """
    def __init__(self, filters=64, **kwargs):
        super(RESNET_Block,self).__init__(**kwargs)
        self.filters = filters
        self.conv_11 = keras.layers.Conv2D(filters=filters, kernel_size=8, padding='same')
        self.BN_11 = keras.layers.BatchNormalization()
        self.relu_11 = keras.layers.Activation("relu")

        self.conv_12 = keras.layers.Conv2D(filters=filters, kernel_size=5, padding='same')
        self.BN_12 = keras.layers.BatchNormalization()
        self.relu_12 = keras.layers.Activation("relu")

        self.conv_13 = keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')
        self.BN_13 = keras.layers.BatchNormalization()
        self.relu_13 = keras.layers.Activation("relu")

        self.conv_1e = keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same')
        self.BN_1e = keras.layers.BatchNormalization()
        self.add_1 = keras.layers.add

    def __call__(self, inputs):
        # Block
        x11 = self.conv_11(inputs)
        x11 = self.BN_11(x11)
        x11 = self.relu_11(x11)

        x12 = self.conv_12(x11)
        x12 = self.BN_12(x12)
        x12 = self.relu_12(x12)

        x13 = self.conv_13(x12)
        x13 = self.BN_13(x13)

        e1 = self.conv_1e(inputs)
        e1 = self.BN_1e(e1)

        b1 = self.add_1([e1, x13])
        b1 = self.relu_13(b1)
        return b1

    def get_config(self):
        config = super(RESNET_Block, self).get_config()
        config.update({"filters": self.filters})
        return config


class RESNET_C:
    """
    ResNet for challenge
    """
    def __init__(self, input_shape, verbose=1, filters=64):
        self.input_shape = input_shape
        self.filters = filters
        self.verbose = verbose
        
    def create_model(self):
        input_layer = keras.layers.Input(self.input_shape)

        block_1 = RESNET_Block(self.filters)(input_layer)
        block_2 = RESNET_Block(self.filters*2)(block_1)
        block_3 = RESNET_Block(self.filters*2)(block_2)

        output_layer = keras.layers.GlobalAveragePooling2D()(block_3)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model


# ResNet with MLP
class RESNET_MLP:
    def __init__(self, input_shape, nb_classes, verbose=2):
        self.input_shape_a, self.input_shape_b = input_shape
        self.nb_classes = nb_classes
        self.verbose = verbose

        # self.model = self.build_model()

    def build_model(self):
        input_a = keras.layers.Input(self.input_shape_a)
        resnet = RESNET_C(input_shape=self.input_shape_a, verbose=0).create_model()(input_a)
        model_a = keras.Model(inputs=input_a, outputs=resnet)

        input_b = keras.layers.Input(self.input_shape_b)
        mlp = keras.layers.Dense(54, activation='relu', name='MLP_Dense1')(input_b)
        mlp = keras.layers.Dense(4, activation='sigmoid', name='MLP_Dense2')(mlp)
        mlp = keras.layers.Dropout(0.3)(mlp)
        model_b = keras.Model(inputs=input_b, outputs=mlp)

        combined = keras.layers.concatenate([model_a.output, model_b.output])
        output_layer = keras.layers.Dense(self.nb_classes, activation='sigmoid', name='FC')(combined)

        model = keras.Model(inputs=[input_a,input_b], outputs=output_layer)

        model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer = keras.optimizers.Adam(), 
            metrics=[keras.metrics.BinaryAccuracy(name='accuracy', dtype=None, threshold=0.5), 
                    keras.metrics.Recall(name='Recall'), keras.metrics.Precision(name='Precision'),
                    keras.metrics.AUC(curve="ROC", name="AUROC"), 
                    keras.metrics.AUC(curve="PR", name="AUPRC")
            ])

        if self.verbose == 2:
            print(model.summary())
        
        return model


def training_resnet_mlp(data_folder, model_folder, verbose=1):
    """
    Training with RESNET+MLP
    """
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')
    
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)
    
    if num_patient_files==0:
        raise Exception('No data was provided.')

    # # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
    
    batch_size = 8
    nb_epochs = 1500

    classes = ['Present', 'Unknown', 'Absent']
    random_seed = 30
    random.seed(random_seed)
    random.shuffle(patient_files)

    # split data to train and test
    split_qty = int(num_patient_files * 0.9)
    patient_files_train, patient_files_validation = patient_files[:split_qty], patient_files[split_qty:]

    X_train1,X_train2,y_train = get_data(classes,patient_files=patient_files_train, pad_length=PAD_LENGTH, data_folder=data_folder, get_training_func=get_wav_data_training)
    X_train = [X_train1,X_train2]

    X_val1,X_val2,y_val = get_data(classes,patient_files=patient_files_validation, pad_length=PAD_LENGTH, data_folder=data_folder, get_training_func=get_wav_data)
    X_val = [X_val1,X_val2]
        
    model = RESNET_MLP(input_shape=[(128,PAD_LENGTH,5),(26,)], nb_classes=3, verbose=verbose).build_model()

    if verbose >= 1:
        print('Training model...')
  
    if not model_folder.endswith('/'):
        model_folder = model_folder + '/'
    
    best_model_path = os.path.join(model_folder,'best_model')
    last_model_path = os.path.join(model_folder,'last_model')

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_loss', mode='min', save_best_only=True,
        save_weights_only=True, verbose=0)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=verbose>=2, patience=10)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=20, restore_best_weights=True)        
    
    callbacks = [model_checkpoint, reduce_lr, early_stop]
    
    fit_verbose = 1 if verbose>=1 else 0
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_val, y_val),
                            verbose=fit_verbose, callbacks=callbacks)
    
    if (verbose >= 2):
        model.summary() 

    model.save_weights(last_model_path)
    