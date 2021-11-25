# Creation of MFCC files to be used as input to machine learning model

import pandas as pd
import os
import librosa
import numpy as np
import sys

folder_name = sys.argv[1]
df = pd.DataFrame()
mfccs = []
classification = []
for file_name in os.listdir(folder_name):
    print(file_name)
    classification.append(file_name.split('_')[0])
    X, sample_rate = librosa.load(folder_name+file_name, res_type='kaiser_fast') 
    _mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
    mfccs.append(_mfccs)

pd.DataFrame(mfccs).to_csv(sys.argv[2])
target = pd.get_dummies(pd.DataFrame(classification))
target.to_csv(sys.argv[3])
