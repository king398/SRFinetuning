import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset, IterableDatasetDict, concatenate_datasets, interleave_datasets
from datasets import Audio
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, \
    get_linear_schedule_with_warmup
import torch
import evaluate
from accelerate import Accelerator
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import Dataset

accelerate = Accelerator(log_with=['wandb'], mixed_precision="fp16")
accelerate.init_trackers(project_name="SR-Finetuning")
model = "openai/whisper-large-v3"
common_voice = IterableDatasetDict()
# Initialize the model and optimizer
tokenizer = WhisperTokenizer.from_pretrained(model, language="Chinese", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = WhisperForConditionalGeneration.from_pretrained(model)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
optimizer = torch.optim.AdamW(model.parameters(), lr=1.25e-6)


class CFG:
    num_devices = torch.cuda.device_count()
    batch_size = 1
    batch_size_per_device = batch_size // 2
    epochs = 2
    num_workers = os.cpu_count() // 2


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", )

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

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN",
                                     split="train+validation",
                                     token=True, num_proc=8)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="test", token=True,
                                    num_proc=8)

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


class WhisperDataset(Dataset):
    def __init__(self, dataset: IterableDatasetDict):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def prepare_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = \
            feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"], truncation=True, max_length=448, padding="max_length").input_ids
        return batch

    def __getitem__(self, idx):
        return self.prepare_dataset(self.dataset[idx])


# Prepare DataLoader for training and evaluation
train_dataloader = DataLoader(WhisperDataset(common_voice["train"]), batch_size=CFG.batch_size,
                              collate_fn=data_collator, pin_memory=True, num_workers=CFG.num_workers)
eval_dataloader = DataLoader(WhisperDataset(common_voice["test"]), batch_size=CFG.batch_size, collate_fn=data_collator,
                             pin_memory=True, num_workers=CFG.num_workers)
total_steps = len(train_dataloader) * CFG.epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=500,
                                            num_training_steps=total_steps)

model, train_dataloader, eval_dataloader = accelerate.prepare(model, train_dataloader, eval_dataloader)
metric = evaluate.load("wer")


def compute_metrics(pred, labels):
    pred_ids = pred.logits.argmax(-1)

    # replace -100 with the pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "pred_str": pred_str, "label_str": label_str}


# Custom training loop
for epoch in range(CFG.epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}",
                      disable=not accelerate.is_local_main_process):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        metrics = compute_metrics(outputs, batch['labels'])
        accelerate.log({"lr": optimizer.param_groups[0]['lr'], "train_loss": loss.item(),
                        "train_wer": metrics["wer"], "pred_str_train": metrics["pred_str"],
                        "label_str_train": metrics["label_str"]})
        #accelerate.print("WER: ", compute_metrics(outputs, batch['labels'])["wer"])

    accelerate.print(f"Average training loss for epoch {epoch}: {total_loss / len(train_dataloader)}")

    # Evaluation loop
    model.eval()
    total_eval_loss = 0
    average_wer = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}",
                          disable=not accelerate.is_local_main_process):
            outputs = model(**batch)
            total_eval_loss += outputs.loss.item()
            wer = compute_metrics(outputs, batch['labels'])
            average_wer += wer["wer"] / len(eval_dataloader)

            accelerate.log({"eval_loss": outputs.loss.item(), "eval_wer": wer["wer"], "pred_str_eval": wer["pred_str"],
                            "label_str_eval": wer["label_str"]})
    accelerate.print(f"Average validation WER For epoch {epoch}: {average_wer,}")
    accelerate.print(f"Average evaluation loss For epoch {epoch}: {total_eval_loss / len(eval_dataloader)}")

# Save the model
model = accelerate.unwrap_model(model)
if accelerate.is_local_main_process:
    print("Saving model")
    model.push_to_hub("whisper-large-v3-chinese-finetune", use_safetensors=True, )

    tokenizer.push_to_hub("whisper-large-v3-chinese-finetune", )
accelerate.end_training()
# Optionally, push to the hub
# model.push_to_hub("whisper-large-v3-chinese")
