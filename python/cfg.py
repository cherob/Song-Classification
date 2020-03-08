import os


class Config:
    def __init__(self,
                 dots=False):  # (False)
        self.mode = 'conv'
        self.nfilt = 26  # (int)
        self.nfeat = 13  # (int)
        self.nfft = 512  # (int)
        self.cat = 3  # (bool, int)
        self.frame_rate = 16000  # (int)
        self.audio_startpoint = 10  # (int)
        self.audio_length = 120  # (int)
        self.validation_data_mult = 0.01  # (int)
        self.use_random_in_feat = False  # (bool)
        self.use_random_in_val_feat = True  # (bool)
        self.sample_length = 1.5  # (int)

        self.epochs = 5  # (int)
        self.batch_size = 32 # (int)
        self.use_checkpoints = False  # (bool)
        self.use_evaluate = False  # (bool)
        self.calls = False  # (bool, int)

        self.max_data = False
        self.max_tack_samples = int((self.audio_length) / self.sample_length)
        self.step = int(self.frame_rate*self.sample_length)
        self.id = 171654

        self.temp_audio_dir = os.path.join('audio', '.temp')
        self.raw_audio_dir = os.path.join('audio', 'raw')
        self.trimmed_audio_date_path = os.path.join('data', 'trimmed.csv')
        self.trimmed_audio_dir = os.path.join('audio', 'trimmed')
        self.refactored_audio_date_path = os.path.join(
            'data', 'refactored.csv')
        self.refactored_audio_dir = os.path.join('audio', 'refactored')
        self.predictions_date_path = os.path.join(
            'data', 'predictions'+str(self.id)+'.csv')
        self.eda_audio_date_path = os.path.join('data', 'edaed.csv')
        self.eda_audio_audio_dir = os.path.join('audio', 'edaed')
        self.model_audio_date_path = os.path.join(
            'data', 'model'+str(self.id)+'.csv')

        self.model_path = os.path.join('models', str(self.id)+'.model')
        self.p_path = os.path.join('config', str(self.id)+'.p')

        self.p_date_dir_path = os.path.join('config')
        self.model_date_dir_path = os.path.join('models')

        self.stats_img_path = os.path.join('images', 'stats.png')
        self.acc_img_path = os.path.join('images', 'acc.png')
        self.loss_img_path = os.path.join('images', 'loss.png')

        self.img_dir = os.path.join('images')
        self.max_class_files = False  # (bool, int)
