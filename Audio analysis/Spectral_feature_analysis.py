import librosa as lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras.models import Sequential
import warnings,pathlib,csv,os

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
data = '../Data/genres/'

def get_img_data(genres):
    cmap = plt.get_cmap('inferno')
    plt.figure(figsize=(8,8))
    
    for g in genres:
     pathlib.Path(f'img_data/{g}').mkdir(parents=True,exist_ok=True)
     for filename in os.listdir(data+g):
        songname = os.path.join(data,g,filename)
        x, sr = lib.load(songname,mono=True,duration=5)
        plt.specgram(x,NFFT=2048,noverlap=128,Fs=2,Fc=0,cmap=cmap,mode='default',sides='default',scale='dB')
        plt.axis('off')
        plt.savefig(f'img_data/{g}/{filename[:-4]}.png')
        plt.clf()

header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1,21):
    header += f' mfcc{i}'
header+=' label'
header = header.split()
file = open('dataset.csv','w',newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True,exist_ok=True)
    for filename in os.listdir(data+g):
        songname = os.path.join(data,g,filename)
        x, sr = lib.load(songname,mono=True,duration=5)
        #rmse = lib.feature.rmse(x)
        chroma_stft = lib.feature.chroma_stft(x,sr=sr)
        spec_cent = lib.feature.spectral_centroid(x,sr=sr)
        spec_bw = lib.feature.spectral_bandwidth(x, sr=sr)
        rolloff = lib.feature.spectral_rolloff(x, sr=sr)
        zcr = lib.feature.zero_crossing_rate(x)
        mfcc = lib.feature.mfcc(y=x, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('dataset.csv','a',newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

data = pd.read_csv('dataset.csv')
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:,-1]

encoder = LabelEncoder()
Y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:,:-1]))

X_train,X_test,y_train,y_test = train_test_split(X,Y,train_size=0.8)

model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.evaluate(X_test,y_test)