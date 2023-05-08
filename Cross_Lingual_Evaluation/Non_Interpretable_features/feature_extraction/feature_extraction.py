# -*- coding: utf-8 -*-
"""
Feature extraction for non-interpretable approaches
(x-vector, TRILLsson, wav2vec2/HuBERT)

Input recordings assumed to be under rec_path.
Output feature files (.npy) will be saved under feat_path

@author: yiting
"""
import numpy as np
import numpy.matlib
import math
import os
import sys
from numpy import save
import pickle

import librosa
import torch
from speechbrain.pretrained import EncoderClassifier
import tensorflow as tf
import tensorflow_hub as hub
from transformers import Wav2Vec2ForSequenceClassification, HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# path to the directory that contains all input data sets 
rec_path = sys.argv[1] #'/export/b15/afavaro/Frontiers/'
# path to the directory to save the extracted features
feat_path = sys.argv[2] #'/export/b11/ytsai25/feats/'

def get_all_sub_segment_inds(x, fs=16e3, dur=10):
    """
        get the range of indices that can be used to run get_sub_segment()
        from the given audio signal
        
        - dur: number of seconds in a segment
    """
    N = x.shape[0] # number of samples in input signal
    N_seg = dur*fs # number of samples in a segment with the duration we want 
    ind_range = math.ceil(N/N_seg) # possible indices: 0:ind_range exclusive
    return ind_range

def get_sub_segment(x, fs=16e3, dur=10, index=0):
    """
        Get a segment of the input signal x
        
        - dur: number of seconds in a segment
        - index: index of the segment counted from the whole signal
    """
    # check if segment out of input range
    N = x.shape[0] # number of samples in input signal
    start_pt = int(index*dur*fs)
    end_pt = int(start_pt + dur*fs)
    if end_pt > N:
        end_pt = N
    
    # get segment
    seg = x[start_pt:end_pt]
    # zero padding at the end to dur if needed
    if seg.shape[0] < (dur*fs):
        pad_len = int((dur*fs)-seg.shape[0])
        seg = np.pad(seg, ((0,pad_len)), 'constant')

    return seg

# extract Paralinguistic speech embeddings
def trillsson_extraction(x,m):
    """
    get trillsson embeddings from one audio
    x: input audio (16khz)
    m = trillsson model
    """
    # normalize input
    x = x / (max(abs(x))) 

    # commented because OOM
#     ind_range = get_all_sub_segment_inds(x, fs=16e3, dur=10) # 10sec segments
#     audio_samples = np.zeros([ind_range, int(10*16e3)]) # [# of sub segments, time]
    
#     # get audios
#     for spec_ind in range(ind_range):
#         seg = get_sub_segment(x, fs=16e3, dur=10, index=spec_ind)
#         audio_samples[spec_ind,:] = seg
        
#     audio_samples = tf.convert_to_tensor(audio_samples, dtype=tf.float32)
#     embeddings = m(audio_samples)['embedding'] # num segments x 1024
#     embeddings = embeddings.numpy()

#     # average across embeddings of all sub-specs
#     features_tmp = np.mean(embeddings, axis=0) # (1024,)
#     features_tmp = features_tmp.reshape((1,1024)) # (1,1024)
    
    # divide into sub segments
    ind_range = get_all_sub_segment_inds(x, fs=16e3, dur=10) # 10sec segments
    embeddings = np.zeros(shape=(ind_range, 1024))
    for spec_ind in range(ind_range):
        seg = get_sub_segment(x, fs=16e3, dur=10, index=spec_ind)
        seg = tf.expand_dims(seg, 0) # -> tf.size [1, 160000]
        embedding = m(seg)['embedding'] # 1x1024
        embeddings[spec_ind,:] = embedding.numpy()

    # average across embeddings of all sub-specs
    features_tmp = np.mean(embeddings, axis=0) # (1024,)
    features_tmp = features_tmp.reshape((1,1024)) # (1,1024)

    return features_tmp

# x-vector extraction, given x sampled at 16kHz
def xvector_extraction(x,classifier):
    # normalize input
    x = x / (max(abs(x))) 
    x = torch.tensor(x[np.newaxis,]) # (459203,) -> torch.Size([1, 459203])

    # extract x-vectors using speechbrain
    embeddings = classifier.encode_batch(x)                        # torch.Size([1, 1, 512])
    features_tmp = embeddings.squeeze().numpy()
    features_tmp = np.reshape(features_tmp,(1, features_tmp.size)) # 1x512
    
#     # get mfccs and deltas
#     mfccs = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length) # n_mfcc x N
#     mfccs_delta = librosa.feature.delta(mfccs)               # n_mfcc x N
#     mfccs_delta2 = librosa.feature.delta(mfccs, order=2)     # n_mfcc x N

#     # Concatenate together as one feature vector
#     features_tmp = np.vstack((mfccs, mfccs_delta, mfccs_delta2)) # n_mfcc*3 x N
#     features_tmp = np.transpose(features_tmp)                    # N x n_mfcc*3
        
    return features_tmp

def wav2vec2_hubert_extraction(x,feature_extractor,model):
    """
    wav2vec2-base-superb-sid or hubert-base-superb-sid mean pooled hidden states extraction, 
    given x sampled at 16kHz
    """
    # normalize input
    x = x / (max(abs(x))) 
    
    # divide input into segments
    ind_range = get_all_sub_segment_inds(x, fs=16e3, dur=10) # 10sec segments
    embeddings = np.zeros(shape=(ind_range, 13, 768)) # 5 layers features x 768 dim
    for spec_ind in range(ind_range):
        seg = get_sub_segment(x, fs=16e3, dur=10, index=spec_ind)
        inputs = feature_extractor(seg, sampling_rate=16000, padding=True, return_tensors="pt")
        hidden_states = model(**inputs).hidden_states # tuple of 13 [1, frame#, 768] tensors
        for layer_num in range(13): # layer_num 0:12
            embeddings[spec_ind,layer_num,:] = hidden_states[layer_num].squeeze().mean(dim=0).detach().numpy() # [768,]

    # average across embeddings of all sub-specs
    hidden_states_list = []
    for layer_num in range(13): # layer_num 0:12
        hidden_layer_avg = np.mean(embeddings[:,layer_num,:], axis=0) # (768,)
        hidden_layer_avg = hidden_layer_avg.reshape((1,768)) # (1,768)
        hidden_states_list.append(hidden_layer_avg)

    return hidden_states_list

    # # if not dividing
    # inputs = feature_extractor(x, sampling_rate=16000, padding=True, return_tensors="pt")
    # hidden_states = model(**inputs).hidden_states
    # hidden_12 = hidden_states[-1].mean(dim=1).detach().numpy() # (1,768)
    # hidden_11 = hidden_states[11].mean(dim=1).detach().numpy()
    # hidden_10 = hidden_states[10].mean(dim=1).detach().numpy()
    # hidden_9 = hidden_states[9].mean(dim=1).detach().numpy()
    # hidden_7 = hidden_states[7].mean(dim=1).detach().numpy()
    # return hidden_12,hidden_11,hidden_10,hidden_9,hidden_7


# feature extraction for czech/german/gita/neurovoz dbs
def feature_extraction_db(sdir, task_inds, id_inds, trill=0):
    """
    Input
    - sdri: source directory of both PD and HC data
    - task_inds: list[int], which index or indices of the split filename (split by '_') indicates the task name
    - id_inds: list[int], which index or indices of the split filename (split by '_') indicates the subject id
    - trill: if using trillsson instead of x-vector (modified: 0-xvector,1-trillsson,2-wav2vec,2-hubert)
    
    czech eg. filename = PD_xx_task_Czech.wav, then task_inds = [2], id_inds = [1]
    
    Output
    - features: dict of { subjectID(str)  ->  dict of {task(str) -> xvector[np 1x512]} }
    - cats: dict of { subjectID(str)  ->  category(str) }, 
            category is PD or HC/CN/CTRL
    - tasks: set{str}, list of tasks
    """
    # dicts of all wav filenames in the selected folder
    wav_dicts = [f for f in list(os.scandir(sdir)) if f.name.endswith('.wav') ]
    
    # pretrained model
    if trill == 1:
        m = hub.KerasLayer('https://tfhub.dev/google/trillsson1/1') # select trillsson1~5
    elif trill == 0: # xvector
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    elif trill == 2: # wav2vec2
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
    elif trill == 3: # hubert
        model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-sid")

    print('Extracting features from all target train/test data...')
    # Store features and categories
    features = {}
    cats = {}
    tasks = set()
    for wav_dict in wav_dicts:
        # get info of the to-be-extracted file 
        filename = wav_dict.name 
        file_split = filename.split('_')
        # task
        task = [file_split[i] for i in task_inds] # get the list of str that indicate task name
        task = '_'.join(task) # list to str
        tasks.add(task)
        # id
        ID = [file_split[i] for i in id_inds] # get the list of str that indicate task name
        ID = '_'.join(ID) # list to str
        # category
        cat = file_split[0] # PD or HC/CN
        if ID not in cats:
            cats[ID] = cat
            
        # get audio data, record duration
        x, fs = librosa.load(sdir + "\\" + wav_dict.name,sr=16000) #testt 24000
        # get x-vector / other embeddings
        if trill == 0:
            features_tmp = xvector_extraction(x,classifier)
        elif trill == 1:
            features_tmp = trillsson_extraction(x,m)
        else:
            hidden_states_list = wav2vec2_hubert_extraction(x,feature_extractor,model)
            features_tmp = hidden_states_list[-1]

        # Store feature vector (of a particular task from a particular subject)
        if ID in features:
            features[ID][task] = features_tmp
        else:
            features[ID] = {}
            features[ID][task] = features_tmp
        
    return features, cats, tasks

# feature extraction for italian db
def feature_extraction_db_ita(sdir, task_files, id_inds, trill=0):
    """
    Input
    - sdri: source directory of both PD and HC data
    - task_files: list[str], a list of task (task folder) name
    - id_inds: list[int], which index or indices of the split filename (split by '_') indicates the subject id
    - trill: if using trillsson instead of x-vector (modified: 0-xvector,1-trillsson,2-wav2vec,2-hubert)
    
    eg. filename = CN_AGNESE_P_B1APGANRET55F170320171104.wav, then id_inds = [1,2]
    
    Output
    - features: dict of { subjectID(str)  ->  dict of {task(str) -> xvector[np 1x512]} }
    - cats: dict of { subjectID(str)  ->  category(str) }, 
            category is PD or HC/CN/CTRL
    - tasks: set{str}, list of tasks
    """
    print('Extracting features from all target train/test data...')
    
    # pretrained model
    if trill == 1:
        m = hub.KerasLayer('https://tfhub.dev/google/trillsson1/1') # select trillsson1~5
    elif trill == 0: # xvector
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    elif trill == 2: # wav2vec2
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
    elif trill == 3: # hubert
        model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-sid")

    # Store features and categories
    features = {}
    cats = {}
    tasks = set()
    
    for task in task_files:
        # dicts of all wav filenames in the selected folder
        wav_dicts = [f for f in list(os.scandir(sdir+'/'+task)) if f.name.endswith('.wav') ]

        for wav_dict in wav_dicts:
            # get info of the to-be-extracted file 
            filename = wav_dict.name 
            file_split = filename.split('_')
            # task
            tasks.add(task)
            # id
            ID = [file_split[i] for i in id_inds] # get the list of str that indicate task name
            ID = '_'.join(ID) # list to str
            # category
            if ID not in cats:
                cat = file_split[0] # PD or HC/CN
                cats[ID] = cat
                
            # get audio data, record duration
            x, fs = librosa.load(sdir+'/'+task+'/' + wav_dict.name,sr=16000) #testt 24000
            # get x-vector / other embeddings
            if trill == 0:
                features_tmp = xvector_extraction(x,classifier)
            elif trill == 1:
                features_tmp = trillsson_extraction(x,m)
            else:
                hidden_states_list = wav2vec2_hubert_extraction(x,feature_extractor,model)
                features_tmp = hidden_states_list[-1]

            # Store feature vector (of a particular task from a particular subject)
            if ID in features:
                features[ID][task] = features_tmp
            else:
                features[ID] = {}
                features[ID][task] = features_tmp
        
    return features, cats, tasks

# feature extraction for czech/german/gita/neurovoz/ dbs, based on feature_extraction_db, also
# return features as arrays of PD feats and HC feats, and store features at the same time 
def feature_extraction_db_extra(sdir, task_inds, id_inds, out_dir='feats/xvector/', db_name='neurovoz', trill=0, nls_labels={}):
    """
    Input 
    - sdri: source directory of both PD and HC data
    - task_inds: list[int], which index or indices of the split filename (split by '_') indicates the task name
    - id_inds: list[int], which index or indices of the split filename (split by '_') indicates the subject id
    - trill: if using trillsson instead of x-vector (modified: 0-xvector,1-trillsson,2-wav2vec,2-hubert)
    - out_dir: output directory to store features of each wav file, include part of file name
    - db_name: str of the db name, used for filename when saving all feats
    - nls_labels: dict { subjectID(str)  ->  category(str) }, required only for nls db
    
    (different way to check labels for db_name = nls)
    czech eg. filename = PD_xx_task_Czech.wav, then task_inds = [2], id_inds = [1]
    
    Output
    - features: dict of { subjectID(str)  ->  dict of {task(str) -> xvector[np 1x512]} }, or 1x1024 
            trillsson instead of xvector
    - cats: dict of { subjectID(str)  ->  category(str) }, 
            category is PD or HC/CN/CTRL
    - tasks: set{str}, list of tasks
    - features_PD: PD features, [#feats x feat dim np arrays]
    - features_HC: HC/CN features, [#feats x feat dim np arrays]
    """
    # dicts of all wav filenames in the selected folder
    wav_dicts = [f for f in list(os.scandir(sdir)) if f.name.endswith('.wav') ]
    
    # pretrained model
    if trill == 1:
        m = hub.KerasLayer('https://tfhub.dev/google/trillsson1/1') # select trillsson1~5
    elif trill == 0: # xvector
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    elif trill == 2: # wav2vec2
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
    elif trill == 3: # hubert
        model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-sid")
    print('Extracting features from all target train/test data...')

    # Store features and categories
    features = {}
    features_PD = np.array(())
    features_ctrl = np.array(())
    cats = {}
    tasks = set()
    for wav_dict in wav_dicts:
        # get info of the to-be-extracted file 
        filename = wav_dict.name 
        file_split = filename.split('_')
        # task
        task = [file_split[i] for i in task_inds] # get the list of str that indicate task name
        task = '_'.join(task) # list to str
        tasks.add(task)
        # id
        ID = [file_split[i] for i in id_inds] # get the list of str that indicate id name
        ID = '_'.join(ID) # list to str
        # category
        if db_name == 'nls':
            # if current wav file subject id not in Anna's label list, skip
            if ID not in nls_labels:
                cat = 'NA'
            else:
                cat = nls_labels[ID]
        else:
            cat = file_split[0] # PD or HC/CN
            
        if ID not in cats:
            cats[ID] = cat
            
        # get audio data, record duration
        x, fs = librosa.load(sdir + "/" + wav_dict.name,sr=16000) #testt 24000
        # get x-vector / other embeddings
        if trill == 0:
            features_tmp = xvector_extraction(x,classifier)
            # save individual feature vectors
            save(out_dir+filename[:-4]+'.npy', features_tmp) # exclude '.wav'
        elif trill == 1:
            features_tmp = trillsson_extraction(x,m)
            # save individual feature vectors
            save(out_dir+filename[:-4]+'.npy', features_tmp) # exclude '.wav'
        else:
            hidden_states_list = wav2vec2_hubert_extraction(x,feature_extractor,model)
            features_tmp = hidden_states_list[-1]
            # save individual feature vectors
            for layer_num in range(13):
                save(out_dir+'hidden'+str(layer_num) +'/'+filename[:-4]+'.npy', hidden_states_list[layer_num]) # exclude '.wav'
            # save(out_dir+'hidden12/'+filename[:-4]+'.npy', hidden_12) # exclude '.wav'
            # save(out_dir+'hidden11/'+filename[:-4]+'.npy', hidden_11) # exclude '.wav'
            # save(out_dir+'hidden10/'+filename[:-4]+'.npy', hidden_10) # exclude '.wav'
            # save(out_dir+'hidden9/'+filename[:-4]+'.npy', hidden_9) # exclude '.wav'
            # save(out_dir+'hidden7/'+filename[:-4]+'.npy', hidden_7) # exclude '.wav'
        
        # Store feature vector (of a particular task from a particular subject)
        if cat.startswith('PD'):
            features_PD = np.vstack((features_PD, features_tmp)) if features_PD.size else features_tmp
        elif cat.startswith('HC') or cat.startswith('CN') or cat.startswith('CTRL'):
            features_ctrl = np.vstack((features_ctrl, features_tmp)) if features_ctrl.size else features_tmp
        else:
            continue # skip if current wav label is not PD/healthy
            
        if ID in features:
            features[ID][task] = features_tmp
        else:
            features[ID] = {}
            features[ID][task] = features_tmp
             
    # save features vars 
    with open(out_dir+db_name+'_features.pkl', 'wb') as file:
        pickle.dump(features, file)
    with open(out_dir+db_name+'_cats.pkl', 'wb') as file:
        pickle.dump(cats, file)
    with open(out_dir+db_name+'_tasks.pkl', 'wb') as file:
        pickle.dump(tasks, file)
    with open(out_dir+db_name+'_features_PD.pkl', 'wb') as file:
        pickle.dump(features_PD, file)
    with open(out_dir+db_name+'_features_ctrl.pkl', 'wb') as file:
        pickle.dump(features_ctrl, file)
        
    return features, cats, tasks, features_PD, features_ctrl


# feature extraction for italian db, based on feature_extraction_db_ita, also
# return features as arrays of PD feats and HC feats, and store features at the same time 
def feature_extraction_db_ita_extra(sdir, task_files, id_inds, out_dir='feats/xvector/', db_name='italian', trill=0):
    """
    Input
    - sdri: source directory of both PD and HC data
    - task_files: list[str], a list of task (task folder) name
    - id_inds: list[int], which index or indices of the split filename (split by '_') indicates the subject id
    - trill: if using trillsson instead of x-vector (modified: 0-xvector,1-trillsson,2-wav2vec,2-hubert)
    - out_dir: output directory to store features of each wav file, include part of file name
    - db_name: str of the db name, used for filename when saving all feats
    
    eg. filename = CN_AGNESE_P_B1APGANRET55F170320171104.wav, then id_inds = [1,2]
    
    Output
    - features: dict of { subjectID(str)  ->  dict of {task(str) -> xvector[np 1x512]} }
    - cats: dict of { subjectID(str)  ->  category(str) }, 
            category is PD or HC/CN/CTRL
    - tasks: set{str}, list of tasks
    - features_PD: PD features, [#feats x feat dim np arrays]
    - features_HC: HC/CN features, [#feats x feat dim np arrays]
    """
    print('Extracting features from all target train/test data...')
    
    # pretrained model
    if trill == 1:
        m = hub.KerasLayer('https://tfhub.dev/google/trillsson1/1') # select trillsson1~5
    elif trill == 0: # xvector
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    elif trill == 2: # wav2vec2
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
    elif trill == 3: # hubert
        model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-sid")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-sid")

    # Store features and categories
    features = {}
    features_PD = np.array(())
    features_ctrl = np.array(())
    cats = {}
    tasks = set()
    
    for task in task_files:
        # dicts of all wav filenames in the selected folder
        wav_dicts = [f for f in list(os.scandir(sdir+'/'+task)) if f.name.endswith('.wav') ]

        for wav_dict in wav_dicts:
            # get info of the to-be-extracted file 
            filename = wav_dict.name 
            file_split = filename.split('_')
            # task
            tasks.add(task)
            # id
            ID = [file_split[i] for i in id_inds] # get the list of str that indicate id name
            ID = ''.join(ID) # list to str
            # category
            if ID not in cats:
                cat = file_split[0] # PD or HC/CN
                cats[ID] = cat
                
            # get audio data, record duration
            x, fs = librosa.load(sdir+'/'+task+'/' + wav_dict.name,sr=16000) #testt 24000
            # get x-vector / other embeddings
            if trill == 0:
                features_tmp = xvector_extraction(x,classifier)
                # save individual feature vectors
                save(out_dir+filename[:-4]+'.npy', features_tmp) # exclude '.wav'
            elif trill == 1:
                features_tmp = trillsson_extraction(x,m)
                # save individual feature vectors
                save(out_dir+filename[:-4]+'.npy', features_tmp) # exclude '.wav'
            else:
                hidden_states_list = wav2vec2_hubert_extraction(x,feature_extractor,model)
                features_tmp = hidden_states_list[-1]
                # save individual feature vectors
                for layer_num in range(13):
                    save(out_dir+'hidden'+str(layer_num) +'/'+filename[:-4]+'.npy', hidden_states_list[layer_num]) # exclude '.wav'
                # save(out_dir+'hidden12/'+filename[:-4]+'.npy', hidden_12) # exclude '.wav'
                # save(out_dir+'hidden11/'+filename[:-4]+'.npy', hidden_11) # exclude '.wav'
                # save(out_dir+'hidden10/'+filename[:-4]+'.npy', hidden_10) # exclude '.wav'
                # save(out_dir+'hidden9/'+filename[:-4]+'.npy', hidden_9) # exclude '.wav'
                # save(out_dir+'hidden7/'+filename[:-4]+'.npy', hidden_7) # exclude '.wav'

            # Store feature vector (of a particular task from a particular subject)
            if cat.startswith('PD'):
                features_PD = np.vstack((features_PD, features_tmp)) if features_PD.size else features_tmp
            else:
                features_ctrl = np.vstack((features_ctrl, features_tmp)) if features_ctrl.size else features_tmp
            
            if ID in features:
                features[ID][task] = features_tmp
            else:
                features[ID] = {}
                features[ID][task] = features_tmp
        
    # save features vars 
    with open(out_dir+db_name+'_features.pkl', 'wb') as file:
        pickle.dump(features, file)
    with open(out_dir+db_name+'_cats.pkl', 'wb') as file:
        pickle.dump(cats, file)
    with open(out_dir+db_name+'_tasks.pkl', 'wb') as file:
        pickle.dump(tasks, file)
    with open(out_dir+db_name+'_features_PD.pkl', 'wb') as file:
        pickle.dump(features_PD, file)
    with open(out_dir+db_name+'_features_ctrl.pkl', 'wb') as file:
        pickle.dump(features_ctrl, file)
        
    return features, cats, tasks, features_PD, features_ctrl


# load NLS labels, dict of {id -> label}, where label is CTRL / PD / ...
with open("nls_id_label_dict.pkl", "rb") as input_file:
    nls_labels = pickle.load(input_file)
    

# extract wav2vec2 ---
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_ita_extra(sdir=rec_path+'Italian_PD', task_files=['B1','B2','FBR1','PR1'], id_inds=[1,2], out_dir=feat_path+'wav2vec2/', db_name='italian', trill=2)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Czech_PD/All_16k', task_inds=[2], id_inds=[1], out_dir=feat_path+'wav2vec2/', db_name='Czech', trill=2)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'German_PD/All', task_inds=[2], id_inds=[1], out_dir=feat_path+'wav2vec2/', db_name='German', trill=2)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'GITA_NEW_TASKS/All_Recordings_Correct_Naming', task_inds=[2], id_inds=[1], out_dir=feat_path+'wav2vec2/', db_name='GITA', trill=2)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Neurovoz_data/Neurovoz_rec', task_inds=[1], id_inds=[-1], out_dir=feat_path+'wav2vec2/', db_name='neurovoz', trill=2)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'NLS/NLS_RESAMPLED', task_inds=[3], id_inds=[0,1], out_dir=feat_path+'wav2vec2/', db_name='nls', trill=2, nls_labels=nls_labels)
# new NLS and italian RP (concatenated readpassage) features
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'NLS/RPs_concatenated', task_inds=[2], id_inds=[0,1], out_dir=feat_path+'wav2vec2/', db_name='nlsCon', trill=2, nls_labels=nls_labels)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Italian_PD/RPs_concatenated', task_inds=[3], id_inds=[1,2], out_dir=feat_path+'wav2vec2/', db_name='italianCon', trill=2)

# # extract hubert ---
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_ita_extra(sdir=rec_path+'Italian_PD', task_files=['B1','B2','FBR1','PR1'], id_inds=[1,2], out_dir=feat_path+'hubert/', db_name='italian', trill=3)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Czech_PD/All_16k', task_inds=[2], id_inds=[1], out_dir=feat_path+'hubert/', db_name='Czech', trill=3)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'German_PD/All', task_inds=[2], id_inds=[1], out_dir=feat_path+'hubert/', db_name='German', trill=3)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'GITA_NEW_TASKS/All_Recordings_Correct_Naming', task_inds=[2], id_inds=[1], out_dir=feat_path+'hubert/', db_name='GITA', trill=3)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Neurovoz_data/Neurovoz_rec', task_inds=[1], id_inds=[-1], out_dir=feat_path+'hubert/', db_name='neurovoz', trill=3)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'NLS/NLS_RESAMPLED', task_inds=[3], id_inds=[0,1], out_dir=feat_path+'hubert/', db_name='nls', trill=3, nls_labels=nls_labels)
# new NLS and italian RP (concatenated readpassage) features
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'NLS/RPs_concatenated', task_inds=[2], id_inds=[0,1], out_dir=feat_path+'hubert/', db_name='nlsCon', trill=3, nls_labels=nls_labels)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Italian_PD/RPs_concatenated', task_inds=[3], id_inds=[1,2], out_dir=feat_path+'hubert/', db_name='italianCon', trill=3)

# # extract x-vectors ---
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_ita_extra(sdir=rec_path+'Italian_PD', task_files=['B1','B2','FBR1','PR1'], id_inds=[1,2], out_dir=feat_path+'xvector/', db_name='italian', trill=0)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Czech_PD/All_16k', task_inds=[2], id_inds=[1], out_dir=feat_path+'xvector/', db_name='Czech', trill=0)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'German_PD/All', task_inds=[2], id_inds=[1], out_dir=feat_path+'xvector/', db_name='German', trill=0)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'GITA_NEW_TASKS/All_Recordings_Correct_Naming', task_inds=[2], id_inds=[1], out_dir=feat_path+'xvector/', db_name='GITA', trill=0)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Neurovoz_data/Neurovoz_rec', task_inds=[1], id_inds=[-1], out_dir=feat_path+'xvector/', db_name='neurovoz', trill=0)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'NLS/NLS_RESAMPLED', task_inds=[3], id_inds=[0,1], out_dir=feat_path+'xvector/', db_name='nls', trill=0, nls_labels=nls_labels)
# new NLS and italian RP (concatenated readpassage) features
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'NLS/RPs_concatenated', task_inds=[2], id_inds=[0,1], out_dir=feat_path+'xvector/', db_name='nlsCon', trill=0, nls_labels=nls_labels)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Italian_PD/RPs_concatenated', task_inds=[3], id_inds=[1,2], out_dir=feat_path+'xvector/', db_name='italianCon', trill=0)

# # extract trillsson representations ---
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_ita_extra(sdir=rec_path+'Italian_PD', task_files=['B1','B2','FBR1','PR1'], id_inds=[1,2], out_dir=feat_path+'trill/', db_name='italian', trill=1)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Czech_PD/All_16k', task_inds=[2], id_inds=[1], out_dir=feat_path+'trill/', db_name='Czech', trill=1)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'German_PD/All', task_inds=[2], id_inds=[1], out_dir=feat_path+'trill/', db_name='German', trill=1)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'GITA_NEW_TASKS/All_Recordings_Correct_Naming', task_inds=[2], id_inds=[1], out_dir=feat_path+'trill/', db_name='GITA', trill=1)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Neurovoz_data/Neurovoz_rec', task_inds=[1], id_inds=[-1], out_dir=feat_path+'trill/', db_name='neurovoz', trill=1)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'NLS/NLS_RESAMPLED', task_inds=[3], id_inds=[0,1], out_dir=feat_path+'trill/', db_name='nls', trill=1, nls_labels=nls_labels)
# new NLS and italian RP (concatenated readpassage) features
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'NLS/RPs_concatenated', task_inds=[2], id_inds=[0,1], out_dir=feat_path+'trill/', db_name='nlsCon', trill=1, nls_labels=nls_labels)
features, cats, tasks, feats_PD, feats_CTRL = feature_extraction_db_extra(sdir=rec_path+'Italian_PD/RPs_concatenated', task_inds=[3], id_inds=[1,2], out_dir=feat_path+'trill/', db_name='italianCon', trill=1)



