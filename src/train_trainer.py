""" Training whisper large using Seq2Seq Trainer and Trainer Arguments.
"""
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import evaluate
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from prepare_dataset import CustomeDataset
import pandas as pd
import gc
import jiwer
from datasets import set_caching_enabled
from utils import DataCollatorSpeechSeq2SeqWithPadding, HyperparameterParser
import yaml

# Configurations
warnings.filterwarnings("ignore")
set_caching_enabled(False)
logger = logging.getLogger(__name__)
CONFIG_PATH = '../configs/trainer-config.yaml'

def get_filtered_metadataset(data_args):
    """
        Filtering Training and Evaluation Dataset.
            > Getting the subset of training (based on fraction).
            > Getting the subset of evaluation (based on fraction).

        Args:
            data_args: data configurations

        Returns:
            - total training samples (int)
            - training_dataset_path (str)
            - evaluation_dataset_path (str)
    """
    # reading dataset 
    tr_df = pd.read_csv(data_args.train_metadata_path)
    te_df = pd.read_csv(data_args.eval_metadata_path)

    # get the sample of the dataset
    tr_df = tr_df.sample(frac=data_args.max_train_samples)
    te_df = te_df.sample(frac=data_args.max_eval_samples)

    # saving the training and evaluation dataset samples.
    tr_path = data_args.train_metadata_path.replace(".csv", "_sub.csv").replace("processed", "final")
    te_eval_path = data_args.eval_metadata_path.replace(".csv", "_sub.csv").replace("processed", "final")

    # save the dataframes in the respective dataset paths.
    tr_df.to_csv(tr_path, index=False)
    te_df.to_csv(te_eval_path, index=False)

    # display details of the dataset.
    print("\n")
    print(f"|> Training dataset at : {tr_path} and Total samples : {len(tr_df)}")
    print(f"|> Evaluation dataset at :: {te_eval_path} and Total samples : {len(te_df)}")

    total_training_samples = len(tr_df)
    del tr_df, te_df
    gc.collect()
    return total_training_samples, tr_path, te_eval_path


def get_confgiurations():
    """
        Getting Configurations from Yaml file.
    """
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        cfg = HyperparameterParser(config)
    return cfg


def main():
    cfg = get_confgiurations()
    parser = HfArgumentParser(Seq2SeqTrainingArguments)
    training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # overwrite the training arguments
    training_args.output_dir = "output directory"
    training_args.do_train = True
    training_args.per_device_train_batch_size = cfg.train_args.train_batch_size
    training_args.per_device_eval_batch_size = cfg.train_args.eval_batch_size
    training_args.gradient_accumulation_steps = cfg.train_args.gradient_accumulation_steps
    training_args.learning_rate = cfg.train_args.learning_rate
    training_args.weight_decay = cfg.train_args.weight_decay
    training_args.num_train_epochs = cfg.train_args.num_epochs
    training_args.warmup_steps = cfg.train_args.num_warmup_steps
    training_args.logging_steps = cfg.train_args.log_steps
    training_args.evaluation_strategy = cfg.train_args.evaluation_strategy
    training_args.predict_with_generate = cfg.train_args.predict_with_generate
    training_args.generation_max_length = cfg.train_args.generation_max_length
    training_args.eval_steps = cfg.train_args.eval_steps
    training_args.metric_for_best_model = cfg.train_args.metric_for_best_model
    training_args.save_steps = cfg.train_args.save_checkpoint_steps
    training_args.save_total_limit = cfg.train_args.save_total_limit
    training_args.save_only_model = cfg.train_args.save_only_model
    training_args.fp16 = cfg.train_args.fp16
    training_args.gradient_checkpointing = cfg.train_args.gradient_checkpointing
    training_args.overwrite_output_dir = cfg.train_args.overwrite_output_dir



    # loading the training and evaluation dataset
    total_train_samples, train_path, eval_path = get_filtered_metadataset(cfg.data)
    training_args.max_steps = int((total_train_samples//training_args.per_device_train_batch_size)* training_args.num_train_epochs)
    cfg.data.train_metadata_path = train_path
    cfg.data.eval_metadata_path = eval_path

    #---------------------------------------------------------------------------
    #       setting up dataloaders, feature extractors, tokenizers and model 
    #---------------------------------------------------------------------------
    
    # Initializing configurations
    config = AutoConfig.from_pretrained(cfg.model_args.model_name_or_path)
    config.update({"forced_decoder_ids": cfg.model_args.forced_decoder_ids, "suppress_tokens": cfg.model_args.suppress_tokens})
    config.update({"apply_spec_augment": cfg.model_args.apply_spec_augment, "mask_time_prob": cfg.model_args.mask_time_prob, "mask_feature_prob": cfg.model_args.mask_feature_prob})

    # initializing the feature Extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_args.model_name_or_path)

    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_args.model_name_or_path)

    # initialize the model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(cfg.model_args.model_name_or_path, config=config)


    if cfg.model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if cfg.model_args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if cfg.model_args.language is not None:
        tokenizer.set_prefix_tokens(language=cfg.model_args.language, task=cfg.model_args.task)


    #------------------ saving the models --------------------------
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    processor = AutoProcessor.from_pretrained(training_args.output_dir)  

    #----------------- Computing metric -----------------------------
    def compute_metrics(pred):
        pred_ids = pred.predictions
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        cer = jiwer.cer(label_str, pred_str)
        return {"cer": cer}
   
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=cfg.model_args.apply_spec_augment,
    )


    #---------------------------------- Getting datasets ----------------------
    custom_dataset_train = CustomeDataset(
                                        tokenizer=tokenizer,
                                        feature_extractor=feature_extractor,
                                        language=cfg.model_args.language,
                                        task=cfg.model_args.task,
                                        audio_column_name= cfg.data.audio_column_name,
                                        text_column_name= cfg.data.text_column_name,
                                        streaming=True,
                                        forward_attention_mask=cfg.model_args.apply_spec_augment,
                                        min_input_length=int(cfg.data.min_duration_in_seconds * feature_extractor.sampling_rate),
                                        max_input_length=int(cfg.data.max_duration_in_seconds * feature_extractor.sampling_rate)
                                        )
    
    custom_dataset_eval = CustomeDataset(
                                        tokenizer=tokenizer,
                                        feature_extractor=feature_extractor,
                                        language=cfg.model_args.language,
                                        task=cfg.model_args.task,
                                        audio_column_name= cfg.data.audio_column_name,
                                        text_column_name= cfg.data.text_column_name,
                                        streaming=True,
                                        forward_attention_mask=cfg.model_args.apply_spec_augment,
                                        min_input_length=int(cfg.data.min_duration_in_seconds * feature_extractor.sampling_rate),
                                        max_input_length=int(cfg.data.max_duration_in_seconds * feature_extractor.sampling_rate),
                                        training=False
                                        )
    

    vectorized_dataset_train = custom_dataset_train.load_dataset_from_csv(cfg.data.train_metadata_path)
    vectorized_dataset_eval = custom_dataset_eval.load_dataset_from_csv(cfg.data.eval_metadata_path)

    #---------------------------------------------------
    #           SEQUENCE TO SEQUENCE TRAINER
    #---------------------------------------------------
    trainer = Seq2SeqTrainer(
                            model=model,
                            args=training_args,
                            train_dataset=vectorized_dataset_train,
                            eval_dataset=vectorized_dataset_eval,
                            tokenizer=feature_extractor,
                            data_collator=data_collator,
                            compute_metrics=compute_metrics,
                        )
    
    train_result = trainer.train()
    trainer.save_model()
    
    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)
    trainer.save_state()



if __name__ == "__main__":
    main()