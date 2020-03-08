import os
from pydub import AudioSegment
import pandas as pd
import glob
from tqdm import tqdm
from cfg import Config

config = Config()


def requirements(audio):
    lms = len(audio)
    ls = lms/1000
    lm = ls/60

    if(lm > 6):
        return False

    if(ls < (config.audio_length + config.audio_startpoint)):
        return False
    return True


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

print("trimming files")
for file in tqdm(audio_files):
    genre = os.path.basename(os.path.split(os.path.normpath(file))[-2])
    name = os.path.basename(os.path.split(
        os.path.normpath(file))[-1]).replace(' ', '_')

    if(file.endswith('.wav')):
        newAudio = AudioSegment.from_wav(file)
    else:
        newAudio = AudioSegment.from_mp3(file)

    if(requirements(newAudio)):
        newPath = os.path.join(newFolder, genre, name)
        # print("{} ==>> {}".format(file, newPath))
        newAudio = newAudio[t1:t2]
        newAudio = newAudio.set_channels(1)
        newAudio = newAudio.set_frame_rate(config.frame_rate)
        newAudio = newAudio.set_sample_width(2)

        newFilename = hex(i) + ".wav"
        i = i + 1
        data.append([name, genre])
        # Exports to a wav file in the current path.

        dirPath = os.path.split(newPath)[0]
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        newAudio.export(newPath, format="wav")

df = pd.DataFrame(data=data, columns=['fname', 'label'])
df.to_csv(config.trimmed_audio_date_path, index=False)
