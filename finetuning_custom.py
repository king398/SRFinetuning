from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from accelerate import Accelerator
from bitsandbytes.optim import AdamW8bit
from datasets import Audio
from datasets import load_dataset, IterableDatasetDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, \
    get_linear_schedule_with_warmup

accelerate = Accelerator(log_with=['wandb'], mixed_precision="fp16", gradient_accumulation_steps=8)
accelerate.init_trackers(project_name="SR-Finetuning")
model = "openai/whisper-large-v3"
common_voice = IterableDatasetDict()
# Initialize the model and optimizer
tokenizer = WhisperTokenizer.from_pretrained(model, language="Chinese", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = WhisperForConditionalGeneration.from_pretrained(model, )
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.gradient_checkpointing_enable()
model.use_cache = False
model.generation_config.language = "zh"
optimizer = AdamW8bit(model.parameters(), lr=1e-6)


class CFG:
    num_devices = torch.cuda.device_count()
    batch_size = 2
    batch_size_per_device = batch_size // torch.cuda.device_count()
    epochs = 5
    num_workers = 8


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

common_voice["train"] = load_dataset("Mithilss/L2TS",
                                     split="train",
                                     token=True, )
common_voice["test"] = load_dataset("Mithilss/L2TS", split="test", token=True,
                                    )

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
# shuffle the dataset
common_voice["train"] = common_voice["train"].shuffle(seed=42)


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
        batch["labels"] = tokenizer(batch["transcript"], truncation=True, max_length=448,
                                    padding="max_length").input_ids
        return batch

    def __getitem__(self, idx):
        return self.prepare_dataset(self.dataset[idx])


# Prepare DataLoader for training and evaluation
train_dataloader = DataLoader(WhisperDataset(common_voice["train"]), batch_size=CFG.batch_size,
                              collate_fn=data_collator, pin_memory=True, num_workers=CFG.num_workers, shuffle=True)
eval_dataloader = DataLoader(WhisperDataset(common_voice["test"]), batch_size=CFG.batch_size * 16,
                             collate_fn=data_collator,
                             pin_memory=True, num_workers=CFG.num_workers)
total_steps = len(train_dataloader) * CFG.epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=(len(train_dataloader) // torch.cuda.device_count()),
                                            num_training_steps=total_steps)

model, train_dataloader, eval_dataloader = accelerate.prepare(model, train_dataloader, eval_dataloader)
metric = evaluate.load("cer")


def compute_metrics(pred, labels):
    pred_ids = pred.argmax(-1)

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

    for i, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}",
                                   disable=not accelerate.is_local_main_process)):
        optimizer.zero_grad()
        with  torch.backends.cuda.sdp_kernel(enable_flash=True,
                                             enable_math=True,
                                             enable_mem_efficient=True) and accelerate.accumulate(
            model):
            outputs = model(**batch)

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        accelerate.log({"lr": optimizer.param_groups[0]['lr'], "train_loss": loss.item()})
        model.eval()
    val_loss = 0
    predictions = []
    labels = []

    for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}",
                      disable=not accelerate.is_local_main_process):
        with torch.no_grad() and torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True):
            outputs = model.generate(**batch)
            outputs = processor.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(outputs)
            labels.extend(processor.batch_decode(batch["labels"], skip_special_tokens=True))
    predictions = [predictions[i] for i in range(len(predictions)) if len(labels[i]) > 0]
    labels = [labels[i] for i in range(len(labels)) if len(labels[i]) > 0]

    cer = metric.compute(predictions=predictions, references=labels)
    accelerate.log({"cer": cer})
    accelerate.print(f"Epoch {epoch} CER: {cer}")
    model = accelerate.unwrap_model(model)
    accelerate.print(
        f"Average training loss for epoch {epoch}: {total_loss / len(train_dataloader)}")
    if accelerate.is_local_main_process:
        model.push_to_hub(f"whisper-large-v3-chinese-finetune-epoch-{epoch}-custom-dataset", safe_serialization=True)
        processor.push_to_hub(f"whisper-large-v3-chinese-finetune-epoch-{epoch}-custom-dataset", )
    accelerate.wait_for_everyone()

# Save the model
accelerate.end_training()
