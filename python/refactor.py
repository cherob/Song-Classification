import os
from pydub import AudioSegment
import pandas as pd
import numpy as np
import os
import shutil
import glob
import re
from tqdm import tqdm
from cfg import Config

config = Config()

newFolder = config.refactored_audio_dir
oldFolder = config.trimmed_audio_dir

print("delete old files")
for file in tqdm(os.listdir(newFolder)):
    file_path = os.path.join(newFolder, file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        # elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

data = []
i = 0


largestSize = 0
audio_files = glob.glob(os.path.join(oldFolder, '*', '*'), recursive=True)

print("find largest files")
for file in tqdm(audio_files):
    genre = os.path.basename(os.path.split(os.path.normpath(file))[-2])
    name = os.path.basename(os.path.split(
        os.path.normpath(file))[-1]).replace(' ', '_')
    newSize = os.path.getsize(file)
    if newSize > largestSize:
        largestSize = newSize


print("refactor files")
for file in tqdm(audio_files):
    fileSize = os.path.getsize(file)
    # print(fileSize, largestSize)
    if(fileSize == largestSize):
        genre = os.path.basename(os.path.split(os.path.normpath(file))[-2])
        name = os.path.basename(os.path.split(
            os.path.normpath(file))[-1]).replace(' ', '_')
        # newFilename = hex(i) + ".wav"
        i = i + 1

        newPath = os.path.join(newFolder, genre, name)

        newAudio = AudioSegment.from_wav(file)
        newAudio.set_frame_rate(config.frame_rate)
        newAudio = newAudio.set_channels(1)
        newAudio = newAudio.set_sample_width(2)
        # print("{} ==>> {}".format(oldFolder+'/'+genre+'/'+file, newPath))
        # Exports to a wav file in the current path.

        dirPath = os.path.split(newPath)[0]
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        newAudio.export(newPath.replace('.mp3', '.wav'), format="wav")

print('FINISH!! {} FILES'.format(i))
