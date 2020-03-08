import os
from pydub import AudioSegment
import pandas as pd
import glob
from tqdm import tqdm
from cfg import Config


def requirements(audio):
    lms = len(audio)
    ls = lms/1000
    lm = ls/60

    if(lm > 6):
        return False

    if(ls < (config.audio_length + config.audio_startpoint)):
        return False
    return True



config = Config()

newFolder = config.trimmed_audio_dir
oldFolder = config.raw_audio_dir

print("delete old files...")
for file in os.listdir(newFolder):
    file_path = os.path.join(newFolder, file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        # elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


start = config.audio_startpoint
size = config.audio_length

t1 = start
t2 = start+size

t1 = t1 * 1000  # Works in milliseconds
t2 = t2 * 1000

data = []
i = 0

audio_files = glob.glob(os.path.join(oldFolder, '*', '*'), recursive=True)


dataset = {
    "l": 0.00,  # long
    "s": 1000*60*10*0.01,  # short
    "c": 0,  # count
    "a": 0.00  # avarage
}

print("calculate files")
for file in tqdm(audio_files):
    genre = os.path.basename(os.path.split(os.path.normpath(file))[-2])
    name = os.path.basename(os.path.split(
        os.path.normpath(file))[-1]).replace(' ', '_')

    if(file.endswith('.wav')):
        newAudio = AudioSegment.from_wav(file)
    else:
        newAudio = AudioSegment.from_mp3(file)

    dataset["c"] = dataset["c"] + 1
    dataset["a"] = dataset["a"] + len(newAudio)

    if(len(newAudio) > dataset["l"]):
        dataset["l"] = len(newAudio)

    if(len(newAudio) < dataset["s"]):
        dataset["s"] = len(newAudio)

    if(requirements(newAudio)):
        newPath = os.path.join(newFolder, genre, name)
        # print("{} ==>> {}".format(file, newPath))
        i = i + 1
        data.append([name, genre])
        # Exports to a wav file in the current path.

        dirPath = os.path.split(newPath)[0]
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        newAudio.export(file, format="wav")


dataset["a"] = dataset["a"]/dataset["c"]


print(dataset)

