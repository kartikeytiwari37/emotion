import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import pandas as pd
import librosa
from keras import regularizers
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
graph=tf.compat.v1.get_default_graph()

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
r={4: 'female_sad', 6: 'male_calm', 9: 'male_sad', 2: 'female_fearful', 7: 'male_fearful', 0: 'female_angry', 8: 'male_happy', 1: 'female_calm', 5: 'male_angry', 3: 'female_happy'}

audio='naya/Recording.wav'

def predict(a):
    data, sampling_rate = librosa.load(a)
    X, sample_rate = librosa.load(a, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    with graph.as_default():
        livepreds = loaded_model.predict(twodim, batch_size=32,verbose=1)
        livepreds1=livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
    return r[liveabc[0]]
        
    
    

s=predict(audio)
print(s)