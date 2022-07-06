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
import keras.api._v2.keras as keras
import tensorflow as tf
import keras.backend as K
import librosa, random, shutil
# import random

from sklearn.impute import SimpleImputer
# import warnings
# warnings.filterwarnings("ignore")

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################
PAD_LENGTH = 128


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # batch_size = 64
    nb_epochs = 300

    split_path = 'split_data'
    split_data(data_folder, dest_folder=split_path)
    
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    # if num_patient_files==0:
    #     raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    # num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    # num_outcome_classes = len(outcome_classes)

    features = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)
        
        # Extract features.
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)
    
    features = np.vstack(features)

    imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)

    m_model_folder = os.path.join(model_folder, 'murmur')
    o_model_folder = os.path.join(model_folder, 'outcome')
    train_murmur(data_path=split_path, model_path=m_model_folder, verbose=verbose,nb_epochs=nb_epochs,batch_size=64,n_mels=128,pad_length=128,imputer=imputer)
    train_outcome(data_path=split_path, model_path=o_model_folder, murmur_model_path=m_model_folder, murmur_model_type='best_model', 
                verbose=verbose,nb_epochs=nb_epochs,batch_size=64,n_mels=128,pad_length=128,imputer=imputer)

    murmur_classifier = m_model_folder
    outcome_classifier = o_model_folder
    
    # murmur_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, murmurs)
    # outcome_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes)

    # Save the model.
    save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier)

    if verbose >= 1:
        print('Done.')


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(filename)
    murmur_model_path = model['murmur_classifier']
    outcome_model_path = model['outcome_classifier']

    murmur_fold_qty = 5
    murmur_models = list()
    for i in range(murmur_fold_qty):
        m_model = Team_Model(model_folder=murmur_model_path, filters=[32,32,32], verbose=verbose)
        murmur_model = m_model.create_resnet_mlp(input_shape=[(128,PAD_LENGTH,5),(26,)],nb_classes=3)
        m_path = os.path.join(murmur_model_path, str(i+1), 'best_model')
        murmur_model.load_weights(m_path).expect_partial()
        murmur_model.trainable = False        
        murmur_models.append(murmur_model)

    outcome_fold_qty = 5
    outcome_models = list()
    for i in range(outcome_fold_qty):
        o_model = Team_Model(model_folder=outcome_model_path, filters=[32,32,32], verbose=verbose)
        outcome_model = o_model.create_resnet_mlp_outcome(input_shape=[(128,PAD_LENGTH,5),(26,),(3*murmur_fold_qty,)],nb_classes=2)
        m_path = os.path.join(outcome_model_path, str(i+1), 'best_model')
        outcome_model.load_weights(m_path).expect_partial()
        outcome_model.trainable = False        
        outcome_models.append(outcome_model)

    model['murmur_models'] = murmur_models
    model['outcome_models'] = outcome_models
    
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    imputer = model['imputer']
    murmur_classes = model['murmur_classes']
    # murmur_classifier = model['murmur_classifier']
    murmur_models = model['murmur_models']
    outcome_classes = model['outcome_classes']
    # outcome_classifier = model['outcome_classifier']
    outcome_models = model['outcome_models']

    # Extract features.
    current_recording = get_wav_data(data, recordings, padding=PAD_LENGTH, n_fft=1024, hop_length=512, n_mels=128)    
    current_recording = np.reshape(current_recording, (1, 128, PAD_LENGTH, 5))    

    # Load features.
    features = get_features(data, recordings)

    # Impute missing data.
    features = features.reshape(1, -1)
    features = imputer.transform(features)

    # Murmur predict
    m_preds = list()
    for m in murmur_models:
        m_pred = m.predict([current_recording, features], verbose=0)
        m_probabilities = tf.nn.softmax(m_pred[0])
        # m_probabilities = m_probabilities.numpy()
        m_preds.append(m_probabilities)
    m_preds = np.asarray(m_preds)
    # print('m_preds: ', m_preds)
    m_pred_outcome = m_preds.reshape(1, -1)

    # Outcome predict
    o_preds = list()
    for m in outcome_models:
        o_pred = m.predict([current_recording, features, m_pred_outcome], verbose=0)
        o_probabilities = tf.nn.softmax(o_pred[0])
        o_preds.append(o_probabilities)
    o_preds = np.asarray(o_preds)
    # print('o_preds: ', o_preds)
   
    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    m_idx, murmur_probabilities = vote_selection(m_preds)
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    murmur_labels[m_idx] = 1
    print('murmur_labels: ', murmur_labels)

    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    o_idx, outcome_probabilities = outcome_vote_selection(o_preds)
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    outcome_labels[o_idx] = 1
    print('outcome_labels: ', outcome_labels)

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

def vote_selection(probabilities:list):
    idxs = np.argmax(probabilities, axis=1)
    idx = -1
    # Select the class that votes more than 2 times
    count_0 = np.count_nonzero(idxs == 0)
    # count_1 = np.count_nonzero(idxs == 1)
    count_2 = np.count_nonzero(idxs == 2)
    if count_2 >= 5:
        idx = 2
    elif count_0 >= 1:
        idx = 0
    else:
        idx = 1

    probs = np.sum(probabilities[idxs == idx], axis=0)            
    probs = tf.nn.softmax(probs)
    probs = probs.numpy()
    
    return idx, probs


def outcome_vote_selection(probabilities:list):
    # print('probabilities: ', probabilities)
    # preference_weight = np.array([0.55, 0.45])
    # prob_softmax = tf.nn.softmax(probabilities * preference_weight)
    # print('prob_softmax: ', prob_softmax)
    idxs = np.argmax(probabilities, axis=1)
    idx = -1
    # Selection
    count_0 = np.count_nonzero(idxs == 0)
    # count_1 = np.count_nonzero(idxs == 1)
    if count_0 >= 2:
        idx = 0
    else:
        idx = 1
        
    probs = np.sum(probabilities[idxs == idx], axis=0)            
    probs = tf.nn.softmax(probs)
    probs = probs.numpy()
    return idx, probs


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

################################################################################
# Added functions

def find_files(data_folder, patient_id):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if root.startswith(f'{patient_id}_'):
            filename = os.path.join(data_folder, f)
            filenames.append(filename)
    return filenames

def copy_files(data_folder, files, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    for file in files:
        shutil.copy(file,dst_path)
        patient_id = get_patient_id(load_patient_data(file))
        relevant_files = find_files(data_folder, patient_id=patient_id)
        if(len(relevant_files)>0):
            for f in relevant_files:
                shutil.copy(f,dst_path)

# Split training data to n folders
def split_data(data_folder, dest_folder='split_data', n=5, verbose=1):
    if verbose>=1:
        print('Spliting data...')

    patient_files = find_patient_files(data_folder)    
    num_patient_files = len(patient_files)
    num_files_per_path = num_patient_files // n

    random_seed = 20
    random.seed(random_seed)
    random.shuffle(patient_files)    
    
    dest_folder = os.path.join(dest_folder)
    if os.path.exists(dest_folder):
        # remove exists splited data folder
        shutil.rmtree(dest_folder)
    
    pos_s = 0
    pos_e = 0
    for i in range(n-1):
        pos_s = min(i*num_files_per_path, num_patient_files)        
        pos_e = min((i+1)*num_files_per_path, num_patient_files)
        
        d_path = os.path.join(dest_folder, str(i+1))
        copy_files(data_folder, patient_files[pos_s:pos_e], d_path)
    
    d_path = os.path.join(dest_folder, str(n))
    copy_files(data_folder, patient_files[pos_e:],d_path)

    if verbose>=1:
        print(f'Splited data to {n} folders.')
    return dest_folder

def band_filter(original_signal, order, fc1,fc2, fs):
    b, a = sp.signal.butter(N=order, Wn=[2*fc1/fs,2*fc2/fs], btype='bandpass')
    new_signal = sp.signal.lfilter(b, a, original_signal)
    return new_signal

# Get the wav data
def get_wav_data(data, recordings, padding=128, fs=4000, n_fft=1024, hop_length=512, n_mels=128):
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
                        # record = band_filter(record, 2, 25, 240, fs)
                        # record = librosa.resample(record, orig_sr=fs, target_sr=1000)
                        # record = librosa.feature.melspectrogram(y=record, sr=1000, n_fft=1024, hop_length=512, n_mels=128)
                        record = librosa.feature.melspectrogram(y=record, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
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
################################################################################

################################################################################
#
# Training
#
################################################################################

def get_data(data_folder, patient_files, verbose=1):
    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    num_patient_files = len(patient_files)

    features = list()
    recordings = list()
    murmurs = list()
    outcomes = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('Loading {}:   {}/{}...'.format(data_folder, i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)
        
        # Extract features.
        current_recording = get_wav_data(current_patient_data, current_recordings, padding=PAD_LENGTH, n_fft=1024, hop_length=512, n_mels=128)
        recording = current_recording
        recording = np.reshape(current_recording, (1, 128, PAD_LENGTH, 5))
        recordings.append(recording)

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

    recordings = np.vstack(recordings)
    features = np.vstack(features)
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)
    return recordings, features, murmurs, outcomes

def murmur_load_data(data_folders:list, verbose=1, imputer=None):
    X1,X2,y = [],[],[]
    for data_folder in data_folders:
        patient_files = find_patient_files(data_folder)
        X_train1,X_train2,y_train,_ = get_data(data_folder=data_folder, patient_files=patient_files, verbose=verbose)
        X1.append(X_train1)
        X2.append(X_train2)
        y.append(y_train)

    X1 = np.vstack(X1)
    X2 = np.vstack(X2)
    y = np.vstack(y)
    if imputer is not None:
        X2 = imputer.transform(X2)
    else:
        X2 = np.nan_to_num(X2)
    ds_x = tf.data.Dataset.from_tensor_slices((X1,X2))
    ds_y = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip((ds_x, ds_y))
    ds = ds.shuffle(y.shape[0] ,reshuffle_each_iteration=True)
    return ds

def outcome_load_data(data_folders:list, verbose=1, imputer=None, murmur_models=None):
    X1,X2,y = [],[],[]
    for data_folder in data_folders:
        patient_files = find_patient_files(data_folder)
        X_train1,X_train2,_,y_train = get_data(data_folder=data_folder, patient_files=patient_files, verbose=verbose)
        X1.append(X_train1)
        X2.append(X_train2)
        y.append(y_train)

    X1 = np.vstack(X1)
    X2 = np.vstack(X2)
    y = np.vstack(y)
    if imputer is not None:
        X2 = imputer.transform(X2)
    else:
        X2 = np.nan_to_num(X2)

    # Get Murmur model predicted data
    murmur_predicts = list()
    if murmur_models is not None:        
        for m in murmur_models:
            m_predict = m.predict((X1,X2),batch_size=16)
            m_predict = tf.nn.softmax(m_predict)
            # print(m_predict)
            murmur_predicts.append(m_predict)
        murmur_predicts = np.asarray(murmur_predicts)
        murmur_predicts = np.concatenate(murmur_predicts, axis=1)
        # print('murmur_predicts: ', murmur_predicts.shape)
    
    X3 = np.vstack(murmur_predicts)
    # print('murmur_features: ', X3)
    # print('murmur_features: ', X3.shape)

    ds_x = tf.data.Dataset.from_tensor_slices((X1,X2,X3))
    ds_y = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip((ds_x, ds_y))
    ds = ds.shuffle(y.shape[0] ,reshuffle_each_iteration=True)
    return ds

def train_murmur(model_path = 'resnet_mlp', data_path='split_data/', verbose = 2, nb_epochs = 200, batch_size = 64, n_mels = 128, pad_length=128, imputer=None):
    model_folder = os.path.join(model_path)
    # log_base_dir = os.path.join('/physionet/logs', log_path)
    PAD_LENGTH = pad_length

    num_folders = 5
    dest_folder = data_path

    for k in range(num_folders):
        training_folders = []
        for i in range(num_folders):
            if i == k:
                continue
            else:
                training_folders.append(os.path.join(dest_folder,str(i+1)))
        
        val_folders=[os.path.join(dest_folder,str(k+1))]
        model_folder_k = os.path.join(model_folder,str(k+1))
        
        t_model = Team_Model(model_folder=model_folder_k, filters=[32,32,32], verbose=verbose)
        pre_training_model_path = None
        model = t_model.create_resnet_mlp(input_shape=[(n_mels,PAD_LENGTH,5),(26,)],nb_classes=3, pre_training_model_path=pre_training_model_path)
        t_model.build_model(model)
        
        # load data
        train_ds = murmur_load_data(data_folders=training_folders,verbose=verbose, imputer=imputer)
        val_ds = murmur_load_data(data_folders=val_folders,verbose=verbose, imputer=imputer)        
        train_ds = train_ds.batch(batch_size).prefetch(2)
        val_ds = val_ds.batch(batch_size).prefetch(2)

        # Train the model.
        if verbose >= 1:
            print(f'Training murmur model {k+1}...')
        
        t_model.fit_(model, train_ds=train_ds, val_ds=val_ds,reduce_lr_patient=10,stop_patient=20, batch_size=batch_size,nb_epochs=nb_epochs,
                    reduce_monitor='loss',stop_monitor='loss',checkpoint_monitor='val_AUPRC', checkpoint_mode='max')
        del model
        del t_model
        # del t_data


def train_outcome(model_path = 'resnet_mlp', murmur_model_path='murmur',murmur_model_type='last_model', data_path='split_data/', verbose = 2, nb_epochs = 200, batch_size = 64, n_mels = 128, pad_length=128, imputer=None):
    model_folder = os.path.join(model_path)
    murmur_model_path =  os.path.join(murmur_model_path)
    PAD_LENGTH = pad_length

    num_folders = 5
    dest_folder = data_path

    murmur_fold_qty = 5
    murmur_models = list()
    for i in range(murmur_fold_qty):
        m_model = Team_Model(model_folder=murmur_model_path, filters=[32,32,32], verbose=verbose)
        murmur_model = m_model.create_resnet_mlp(input_shape=[(n_mels,PAD_LENGTH,5),(26,)],nb_classes=3)
        m_path = os.path.join(murmur_model_path, str(i+1), murmur_model_type)
        murmur_model.load_weights(m_path).expect_partial()
        murmur_model.trainable = False        
        murmur_models.append(murmur_model)

    for k in range(num_folders):
        training_folders = []
        for i in range(num_folders):
            if i == k:
                continue
            else:
                training_folders.append(os.path.join(dest_folder,str(i+1)))
        
        val_folders=[os.path.join(dest_folder,str(k+1))]
        model_folder_k = os.path.join(model_folder,str(k+1))
        
        t_model = Team_Model(model_folder=model_folder_k, filters=[32,32,32], verbose=verbose)        
        model = t_model.create_resnet_mlp_outcome(input_shape=[(n_mels,PAD_LENGTH,5),(26,),(3*num_folders,)],nb_classes=2)
        t_model.build_model(model)
        
        # load data
        train_ds = outcome_load_data(data_folders=training_folders,verbose=verbose, imputer=imputer, murmur_models=murmur_models)
        val_ds = outcome_load_data(data_folders=val_folders,verbose=verbose, imputer=imputer, murmur_models=murmur_models)        
        train_ds = train_ds.batch(batch_size).prefetch(2)
        val_ds = val_ds.batch(batch_size).prefetch(2)

        # Train the model.
        if verbose >= 1:
            print(f'Training outcome model {k+1}...')
        
        t_model.fit_(model, train_ds=train_ds, val_ds=val_ds,reduce_lr_patient=10,stop_patient=20, batch_size=batch_size,nb_epochs=nb_epochs,
                    reduce_monitor='loss',stop_monitor='loss',checkpoint_monitor='val_AUPRC', checkpoint_mode='max')
        del model
        del t_model


################################################################################
#
# Models
#
################################################################################
class RESNET_Block(keras.layers.Layer):
    """
    Custom ResNet Block
    """
    def __init__(self, filters=[64, 64, 64], kernels=[8, 5, 3], **kwargs):
        super(RESNET_Block,self).__init__(**kwargs)
        self.filters = filters
        self.kenels = kernels
        [kenel1, kenel2, kenel3] = kernels
        [filter1, filter2, filter3] = filters
        self.conv_11 = keras.layers.Conv2D(filters=filter1, kernel_size=kenel1, padding='same')
        self.BN_11 = keras.layers.BatchNormalization()
        self.relu_11 = keras.layers.Activation("relu")        

        self.conv_12 = keras.layers.Conv2D(filters=filter2, kernel_size=kenel2, padding='same')
        self.BN_12 = keras.layers.BatchNormalization()
        self.relu_12 = keras.layers.Activation("relu")

        self.conv_13 = keras.layers.Conv2D(filters=filter3, kernel_size=kenel3, padding='same')
        self.BN_13 = keras.layers.BatchNormalization()
        self.relu_13 = keras.layers.Activation("relu")

        self.conv_1e = keras.layers.Conv2D(filters=filter3, kernel_size=1, padding='same')
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
        config.update({"kenels": self.kenels})
        return config

# ResNet with MLP
class Team_Model:
    def __init__(self, verbose=2, filters=[64, 64, 64], model_folder='model'):
        self.verbose = verbose
        self.filters = filters
        self.best_model_path = os.path.join(model_folder, 'best_model')
        self.last_model_path = os.path.join(model_folder, 'last_model')      
    
    def create_resnet(self, input_shape, include_top=True, nb_classes=1, name='RESNET_C'):
        input_layer = keras.layers.Input(input_shape)
        block_1 = RESNET_Block(self.filters)(input_layer)
        block_2 = RESNET_Block([i*2 for i in self.filters])(block_1)
        block_3 = RESNET_Block([i*2 for i in self.filters])(block_2)

        output_layer = keras.layers.GlobalAveragePooling2D()(block_3)

        if include_top == True:
            output_layer = keras.layers.Dense(nb_classes, activation='sigmoid')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer, name=name)
        return model

    def create_resnet_mlp(self, input_shape, nb_classes:int, name='RESNET_MLP_C', pre_training_model_path=None):
        input_shape_a, input_shape_b = input_shape
        input_a = keras.layers.Input(input_shape_a)

        resnet_model = self.create_resnet(input_shape=input_shape_a, include_top=False)
        if pre_training_model_path is not None:
            resnet_model.load_weights(pre_training_model_path).expect_partial()
        
        resnet = resnet_model(input_a)
        model_a = keras.Model(inputs=input_a, outputs=resnet)

        input_b = keras.layers.Input(input_shape_b)
        mlp = keras.layers.Dense(54, activation='relu', name='MLP_Dense1')(input_b)
        mlp = keras.layers.Dense(4, activation='sigmoid', name='MLP_Dense2')(mlp)
        mlp = keras.layers.Dropout(0.3)(mlp)
        model_b = keras.Model(inputs=input_b, outputs=mlp)

        combined = keras.layers.concatenate([model_a.output, model_b.output])
        output_layer = keras.layers.Dense(nb_classes, activation='sigmoid', name='FC')(combined)

        model = keras.Model(inputs=[input_a,input_b], outputs=output_layer, name=name)
        
        return model

    def create_resnet_mlp_outcome(self, input_shape, nb_classes:int, name='RESNET_MLP_OUTCOME'):
        input_shape_a, input_shape_b, input_shape_c = input_shape
        input_a = keras.layers.Input(input_shape_a)

        resnet_model = self.create_resnet(input_shape=input_shape_a, include_top=False)
        
        resnet = resnet_model(input_a)
        model_a = keras.Model(inputs=input_a, outputs=resnet)

        input_b = keras.layers.Input(input_shape_b)
        mlp = keras.layers.Dense(54, activation='relu', name='MLP_Dense1')(input_b)
        mlp = keras.layers.Dense(4, activation='sigmoid', name='MLP_Dense2')(mlp)
        mlp = keras.layers.Dropout(0.3)(mlp)
        model_b = keras.Model(inputs=input_b, outputs=mlp)

        input_c = keras.layers.Input(input_shape_c)

        combined = keras.layers.concatenate([model_a.output, model_b.output, input_c])
        output_layer = keras.layers.Dense(nb_classes, activation='sigmoid', name='FC')(combined)

        model = keras.Model(inputs=[input_a,input_b,input_c], outputs=output_layer, name=name)
        
        return model

    def build_model(self, model: keras.Model):
        model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer = keras.optimizers.Adam(), 
            metrics=[keras.metrics.BinaryAccuracy(name='accuracy', dtype=None, threshold=0.5), 
                    keras.metrics.Recall(name='Recall'), keras.metrics.Precision(name='Precision'),
                    keras.metrics.AUC(curve="ROC", name="AUROC"), 
                    keras.metrics.AUC(curve="PR", name="AUPRC")
            ])

        if self.verbose == 2:
            print(model.summary())

        return model

    def fit_(self, model: keras.Model, train_ds:tf.data.Dataset, val_ds:tf.data.Dataset=None, batch_size = 8, 
                reduce_lr_patient=10, stop_patient=20, nb_epochs = 1500, steps_per_epoch = None,
                reduce_monitor='loss', reduce_mode='auto', stop_monitor='loss', stop_mode='min', 
                checkpoint_monitor='loss', checkpoint_mode='min'):
        # nb_epochs = 1500
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=self.best_model_path, monitor=checkpoint_monitor, mode=checkpoint_mode, save_best_only=True,
            save_weights_only=True, verbose=self.verbose)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=reduce_monitor, verbose=self.verbose>=2, patience=reduce_lr_patient, mode=reduce_mode
            , cooldown = 2, min_lr=1e-7
        )
        early_stop = keras.callbacks.EarlyStopping(monitor=stop_monitor, mode=stop_mode, verbose=self.verbose, patience=stop_patient, restore_best_weights=True)        
        
        callbacks = [model_checkpoint, reduce_lr, early_stop]

        fit_verbose = 2 if self.verbose>=1 else 0
        if not val_ds is None:
            hist = model.fit(train_ds, batch_size=batch_size, epochs=nb_epochs, steps_per_epoch = steps_per_epoch,
                              verbose=fit_verbose, validation_data=val_ds, callbacks=callbacks)
        else:
            hist = model.fit(train_ds, batch_size=batch_size, epochs=nb_epochs, steps_per_epoch = steps_per_epoch, 
                              verbose=fit_verbose, callbacks=callbacks)
                          
        if (self.verbose >= 2):
            model.summary()

        model.save_weights(self.last_model_path)

        keras.backend.clear_session()

    def load_best_weight(self, model:keras.Model):
        model.load_weights(self.best_model_path).expect_partial()
        return model

    def predict_(self, model:keras.Model, X):
        model.load_weights(self.best_model_path).expect_partial()
        pred = model.predict(X)
        return pred
    

if __name__ == '__main__':
    train_challenge_model('training_data', 'model', 1)
