import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
from cfg import Config
import ntpath
import easygui

config = Config()

fn = easygui.fileopenbox('F:\Musik')

with open(config.p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)


df = pd.read_csv(config.model_audio_date_path)
classes = list(np.unique(df.label))

y_pred = False
fn_prob = False


localPath = os.path.join(fn)
if(fn.endswith('.wav')):
    newAudio = AudioSegment.from_wav(localPath)
else:
    newAudio = AudioSegment.from_mp3(localPath)
newAudio = newAudio.set_frame_rate(config.frame_rate)
newAudio = newAudio.set_channels(1)
newAudio = newAudio.set_sample_width(2)
tempPath = 'C:/Users/robin/AppData/Local/Temp/'+ntpath.basename(fn)
newAudio.export(tempPath, format="wav")


rate, wav = wavfile.read(tempPath)
y_prob = False
os.remove(tempPath)

for i in tqdm(range(0, wav.shape[0]-config.step, config.step)):
    sample = wav[i:i+config.step]
    x = mfcc(sample, rate, numcep=config.nfeat,
             nfilt=config.nfilt, nfft=config.nfft)
    x = (x - config.min) / (config.max - config.min)

    if config.mode == 'conv':
        x = x.reshape(1, x.shape[1], x.shape[0], 1)
    elif config.mode == 'time':
        x = np.expand_dims(x, axis=0)
    y_hat = model.predict(x)
    y_prob = (y_hat)
    y_pred = (np.argmax(y_hat))

fn_prob = np.mean(y_prob, axis=0).flatten()
y_pred = classes[int(y_pred)]

print("Predict: {} ({}%)".format(y_pred, round(fn_prob[0]*100, 2)))
