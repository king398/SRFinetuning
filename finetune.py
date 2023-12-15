from datasets import load_dataset, IterableDatasetDict, concatenate_datasets
from datasets import Audio
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
import torch
import evaluate

from dataclasses import dataclass
from typing import Any, Dict, List, Union

model = "openai/whisper-large-v3"
common_voice = IterableDatasetDict()
from datasets import interleave_datasets, load_dataset


def load_streaming_dataset(dataset_name, dataset_config_name, split, **kwargs):
    if "+" in split:
        # load multiple splits separated by the `+` symbol *with* streaming mode
        dataset_splits = [load_dataset(dataset_name, dataset_config_name, split=split_name, streaming=True, **kwargs)
                          for split_name in split.split("+")]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(dataset_name, dataset_config_name, split=split, streaming=True, **kwargs)
        return dataset


common_voice["train"] = load_streaming_dataset("mozilla-foundation/common_voice_11_0", "zh-CN",
                                               split="train+validation",
                                               token=True, )
common_voice["test"] = load_streaming_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="test", token=True,
                                              )
#

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

tokenizer = WhisperTokenizer.from_pretrained(model, language="Chinese", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])


class CFG:
    num_devices = torch.cuda.device_count()
    batch_size = 2 * num_devices
    batch_size_per_device = batch_size // num_devices
    epochs = 0.001


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"], truncation=True, padding="max_length",
                                max_length=448).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset).with_format("torch")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-large-v3-chinese",  # change to a repo name of your choice
    per_device_train_batch_size=CFG.batch_size_per_device,
    learning_rate=1.25e-6,
    warmup_steps=500,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=CFG.batch_size_per_device,
    predict_with_generate=True,
    generation_max_length=448,
    logging_steps=25,
    report_to=["wandb"],
    push_to_hub=False,
    save_strategy="steps",
    dataloader_pin_memory=True,
    eval_steps=40000 // CFG.batch_size,
    save_steps=40000 // CFG.batch_size,
    max_steps=int(40000 * CFG.epochs // CFG.batch_size),
    save_safetensors=True,
    save_total_limit=1,

)
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(

    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


def launch():
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        trainer.train()
    model.push_to_hub("whisper-large-v3-chinese")
    # upload the model to the hub


launch()
