"""
    Audio Data Augmentation 

    Features:
        1. Resampling augmentation (downsample -> upsampling)
        2. Speed and Pitch Augmentation
        3. noise Augmentation
"""
import torch
import torchaudio
import os
import soundfile as sf
import numpy as np
import random 
import yaml
from utils import HyperparameterParser

# configurations
device = "cuda" if torch.cuda.is_available() else "cpu"
config_path = "../configs/trainer-config.yaml"

# Reading the configuration files
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    cfg = HyperparameterParser(config)


class AudioAugmentation:
    """
        Audio Augmentation :: Creating the synthetic data from the augmenting the audio.
    """
    def __init__(self):
        self.noise_file_dir = cfg.audio_augment.noise_files
        self.files = os.listdir(self.noise_file_dir)
        self.augment_sampling_rate = cfg.audio_augment.augment_sr
        self.original_sr = cfg.audio_augment.original_sr
        self.speed_factor = cfg.audio_augment.speed_factor
        self.fixed_rate =  cfg.audio_augment.time_stretch_factor
        self.max_min_snr = cfg.audio_augment.snr_list
        self.dis_enable_additive_noisy = [True, False, True, False, False]

        # speed layer
        self.speed_layer = torchaudio.transforms.SpeedPerturbation(orig_freq=self.original_sr, factors=self.speed_factor)
        if(self.fixed_rate):
            self.time_stretch = torchaudio.transforms.TimeStretch(fixed_rate = self.fixed_rate)

        self.additiveNoise = torchaudio.transforms.AddNoise()


    def augmentSampling(self, audio):
        ''' AugmentSampling

            original audio 16Khz -> 8khz -> 16khz
            Args:
                audio (np.ndarray) : An numpy array for an audio.

            Returns:
                np.ndarry : audio resampled to 16khz

        '''
        audio_8hz = torchaudio.functional.resample(audio, orig_freq = self.original_sr, new_freq = self.augment_sampling_rate)
        audio_16hz = torchaudio.functional.resample(audio_8hz, orig_freq=self.augment_sampling_rate, new_freq=self.original_sr)
        return audio_16hz
    

    def _select_random_noise_data(self):
        """
            Select a random noise file from the noise directory.
            Returns:
                np.ndarray
        """
        fold = random.randint(0, len(self.files)-1)
        audio_path = os.path.join(self.noise_file_dir, self.files[fold])
        audio,sr = torchaudio.load(audio_path)
        return audio
    

    def _pad_noise_data(self,original_audio):
        """
            Additing the noise dataset to the original audio data.

            Args:
                original_audio (np.ndarray) : an audio from the training data

            Returns:
                additive noise (np.ndarray)
        """
        noise_data = self._select_random_noise_data()
        len_noise = noise_data.shape[1]
        len_original_audio = original_audio.shape[1]
        index_range = len_noise - len_original_audio
        index = random.randint(0,index_range)
        noise_snippet = noise_data[:, index:index+len_original_audio]
        return noise_snippet


    def add_noisy_data_to_source_audio(self, original_audio):
        """
            Additive noise.
            Noise augmentation will be select based on the probability.

            Anyway for addtive noisy we are adding based on snr values. 
        """

        # noise_data = self._select_random_noise_data()
        # original_audio = original_audio.to(device)
        noise_data = self._pad_noise_data(original_audio) + 0.00001
        snr = torch.randint(low=self.max_min_snr[0], high=self.max_min_snr[1], size=(1,))
        
        add_noise = self.dis_enble_additive_noisy[torch.randint(low=0, high= len(self.dis_enble_additive_noisy), size=(1,1))[0]]
        if(add_noise):
            noise_wav = self.additiveNoise(waveform = original_audio, noise = noise_data, snr=snr)
            return noise_wav
        else:
            return original_audio

    def speed_augmentation(self, wav_array):
        try:
            array_ = self.speed_layer(wav_array)
        except:
            array_ = [wav_array]
        return array_