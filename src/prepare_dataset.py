"""
    Preparing the Iteratable dataset for streaming training data.
"""


import os
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import datasets
import random
import torchaudio
from datasets import DatasetDict, load_dataset, Features
from transformers import (
                          AutoFeatureExtractor,
                          AutoTokenizer
                        )
import torch
from data_augmentation import AudioAugmentation
warnings.filterwarnings('ignore')


# Augmentation using the TorchAudio dataset.
class CustomeDataset:
    """ Creating the Custome Dataset

        Args:
            tokenizer (AutoTokenizer) : initializing the tokenizer
            feature_extractor (AutoFeatureExtractor) : a prerained feature extractor
            language (str) : target language
            task (str) : specifying the whishper task for finetuning ( here in this case is transcribe)
            audio_column_name (str) : the column name in the csv file which constis of the audio paths.
            text_column_name (str) : the column name in the csv file which consists of the text paths.
            streaming (bool) : The datset need to read in iteratively or at a time
            forward_attention_mask (bool) : this is used which when have the spec Augmentation
            min_input_length (int): minimum length of the input. in number of sample points according to the duration
            max_input_length (int): maximum length of the input, its in number of sample points according to the duration
            min_label_tokens (int) : filtering or ignoring the samples which are less this will be ignored
            max_label_tokens (int) : If the number of characters were more than this will be ignore from the training
            training (bool) : True if we are preparing dataset for training or False if we are preparing dataset for Evaluation
            
        Returns:
            vectorized_dataset (Datasets) : An iterative or non-iterative dataset.
    """

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 feature_extractor: AutoFeatureExtractor,
                 language:str,
                 task:str,
                 audio_column_name:str,
                 text_column_name:str,
                 streaming:bool,
                 forward_attention_mask:bool,
                 min_input_length:int,
                 max_input_length:int,
                 min_label_tokens:int = 6,
                 max_label_tokens:int = 448,
                 training:bool = True
                 ):
        
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.audio_column_name = audio_column_name
        self.text_column_name = text_column_name
        self.streaming = streaming
        self.forward_attention_mask = forward_attention_mask
        self.min_input_length = min_input_length
        self.max_input_length = max_input_length
        self.raw_dataset = None
        self.tokenizer.set_prefix_tokens(language=language, task=task)
        self.model_input_names = self.feature_extractor.model_input_names[0]
        self.features = Features({'audio': datasets.Value(dtype='string', id=None), 'transcript': datasets.Value(dtype='string', id=None)})
        self.audio_augment = AudioAugmentation()
        self.min_label_tokens = min_label_tokens
        self.max_label_tokens = max_label_tokens
        self.training = training


    def augmentation(self, original_audio):
        original_audio = torch.Tensor(original_audio).unsqueeze(0)
        noisy_audio = self.audio_augment.add_noisy_data_to_source_audio(original_audio)
        # speed_audio = self.audio_augment.speed_augmentation(noisy_audio)[0]
        return noisy_audio[0].numpy()

    
    def get_features(self, batch):
        sample = batch[self.audio_column_name]
        if(self.training):
            array_ = self.augmentation(sample["array"])
        else:
            array_ = sample["array"]

        inputs = self.feature_extractor(array_, sampling_rate = sample["sampling_rate"], return_attention_mask = self.forward_attention_mask)
        batch[self.model_input_names] = inputs.get(self.model_input_names)[0]
        batch["input_length"] = len(sample["array"])
        if(self.forward_attention_mask):
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        input_str = batch[self.text_column_name]
        batch["labels"] = self.tokenizer(input_str).input_ids
        return batch
    

    def is_audio_in_length_range(self, length):
        return length > self.min_input_length and length < self.max_input_length
    
    def filter_labels(self,labels):
        return self.min_label_tokens < len(labels) < self.max_label_tokens
        


    def load_dataset_from_csv(self, metadata_path):
        raw_ds = load_dataset("csv", data_files=metadata_path,streaming=self.streaming,features=self.features, split="train")
        raw_dataset = raw_ds.cast_column(self.audio_column_name, datasets.features.Audio(sampling_rate=self.feature_extractor.sampling_rate))
        raw_dataset = raw_dataset.map(self.get_features)
        raw_dataset = raw_dataset.filter(self.is_audio_in_length_range, input_columns = ["input_length"])
        return raw_dataset
    
    def clean_cache_files(self):
        self.raw_dataset.cleanup_cache_files()




