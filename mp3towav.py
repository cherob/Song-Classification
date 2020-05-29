
import os
from pydub import AudioSegment
import pandas as pd
import numpy as np
import os
import shutil
import glob
import re
from tqdm import tqdm

audioFolder = os.path.join('audio')
audio_files = glob.glob(os.path.join(audioFolder, '*', '*'), recursive=True)

print("refactor files")

for file in tqdm(audio_files):

    fileSize = os.path.getsize(file)
    genre = os.path.basename(os.path.split(os.path.normpath(file))[-2])
    name = os.path.basename(os.path.split(
        os.path.normpath(file))[-1]).replace(' ', '_')
    # newFilename = hex(i) + ".wav"

    newPath = os.path.join(audioFolder, genre, name)

    if(file.endswith('.wav')):
        newAudio = AudioSegment.from_wav(file)
    else:
        newAudio = AudioSegment.from_mp3(file)

    newAudio = newAudio.set_channels(1)
    newAudio = newAudio.set_sample_width(2)
    # print("{} ==>> {}".format(audioFolder+'/'+genre+'/'+file, newPath))
    # Exports to a wav file in the current path.

    dirPath = os.path.split(newPath)[0]
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    if (file.endswith('.mp3')):
        os.unlink(file)

    newAudio.export(newPath.replace('.mp3', '.wav'), format="wav")

print('FINISH!!')
