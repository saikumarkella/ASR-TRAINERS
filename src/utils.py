from typing import Any
import os
import warnings
from dataclasses import dataclass
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    AdamW,
    set_seed,
)
from glob import glob
from datasets import load_dataset
from typing import Any, List, Dict, Union
from datetime import datetime

class HyperparameterParser:

    def __init__(self, dict_):
        for k,v in dict_.items():
            if(type(v)==dict):
                v = HyperparameterParser(v)
            setattr(self, k, v)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
    
        return batch
    

def load_parquet(dir_path):
    """Loading Dataset from the parquet cache files which were got
        from the Preprocessing step.

        Args:
            config (Config class) : A configuration class for defining all the parametes / attributes.

        Return:
            Vectorized dataset
    """
    train_files = glob(f"{dir_path}/processed_data/train_*")
    test_files = glob(f"{dir_path}/processed_data/eval_*")
    vectorized_dataset = load_dataset("parquet", data_files={"train":train_files, "test":test_files})
    return vectorized_dataset


def load_configurations(Config, Training_args):
    """ Load the configurations and returns parsers & training args.

        Args:
            Config (Configurations) : Getting the configurations 
            Training_args (Configs) : Training Arguments.
    """
    parser = HfArgumentParser((Config, Training_args))
    cfg, training_args = parser.parse_args_into_dataclasses()
    return cfg, training_args



def load_models(cfg):
    """Loading Feature Extractor / tokenizer / model
        
        Args:
            cfg (Config) : defined configurations
        Returns:
            config, feature_extractor, tokenizer, model
    """
    config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(cfg.model_name_or_path)
    return config,feature_extractor, tokenizer, model



def resume_from_latest_checkpoint(accelerator, checkpoint_dir):
    """Resum the training state from the latest checkpoint
    """
    # get the latest checkpoint
    f = [dir_ for dir_ in os.path.scandir(checkpoint_dir) if dir_.is_dir()]
    f.sort(key=os.path.getctime)
    latest_checkpoint_dir = f
    
    path_to_checkpoint_dir = os.path.join(checkpoint_dir, latest_checkpoint_dir)
    try:
        accelerator.load_state(path_to_checkpoint_dir)
        accelerator.print(f"Model from path :: {path_to_checkpoint_dir}  --- LOADED")
    except:
        accelerator.print(f"Model Loading FAILED")
        exit(0)


def restore_checkpoint(accelerator, checkpoint_path: str):
    """Restoring the model with the given checkpoint
        Args:
            accelerator (Accelerator) : object for accelerating the model
            checkpoint_path (str) : A path to the checkpoint

        Return:
            It will loads the model.
    """
    try:
        accelerator.load_state(checkpoint_path)
        accelerator.print(f"Model from path :: {checkpoint_path} --- LOADED")
    except:
        accelerator.print(f"Model Loading FAILED")
        exit(0)


def logging_stdout_templet_1(mode, steps, total_steps, global_steps, params_dict):
    '''Display logs in terminal outs
        
        Args:
            mode : training / evaluation
            steps : current steps
            total_steps : total steps
            global_steps : global steps
            params dict : logging dictonary
    '''
    print("\n")
    if(mode == "training"):
        header = f"| {mode.upper()} --- steps {steps}/{total_steps} --- {datetime.now()} --- gobal_steps {global_steps}"
        print(header)
        for i,v in params_dict.items():
            print(f"\t> {i}  :  {v}")
    else:
        header = f"| {mode.upper()} --- {datetime.now()} --- gobal_steps {global_steps}"
        print(header)
        for i,v in params_dict.items():
            print(f"\t> {i}  :  {v}")