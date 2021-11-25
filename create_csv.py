# Creation of MFCC files to be used as input to machine learning model

import pandas as pd
import os
import librosa
import numpy as np
import sys

folder_name = sys.argv[1]
df = pd.DataFrame()
mfccs = []
for file_name in os.listdir(folder_name):
    print(file_name)
    X, sample_rate = librosa.load(folder_name+file_name, res_type='kaiser_fast') 
    _mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
    mfccs.append(_mfccs)

pd.DataFrame(mfccs).to_csv(sys.argv[2])