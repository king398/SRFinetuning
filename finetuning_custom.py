from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from bitsandbytes.optim import AdamW8bit
from datasets import Audio
from datasets import load_dataset, IterableDatasetDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, \
    get_linear_schedule_with_warmup, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer, \
    Wav2Vec2ForCTC
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

accelerate = Accelerator(log_with=['wandb'], mixed_precision="bf16", gradient_accumulation_steps=2)
accelerate.init_trackers(project_name="SR-Finetuning-custom")
model = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
common_voice = IterableDatasetDict()
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model, language="Chinese", task="transcribe")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = Wav2Vec2ForCTC.from_pretrained(model, )
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.gradient_checkpointing_enable()
model.use_cache = False
#model.generation_config.language = "zh"
optimizer = AdamW8bit(model.parameters(), lr=1e-5)


normalizer = BasicTextNormalizer()


class CFG:
    num_devices = torch.cuda.device_count()
    batch_size = 8
    batch_size_per_device = batch_size // torch.cuda.device_count()
    epochs = 5
    num_workers = 8


import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor = processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch




data_collator = DataCollatorCTCWithPadding()



common_voice["train"] = load_dataset("Mithilss/L2TS",
                                     split="train",
                                     token=True, )
common_voice["test"] = load_dataset("Mithilss/L2TS", split="test", token=True,
                                    )

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
# shuffle the dataset
common_voice["train"] = common_voice["train"].shuffle(seed=42)


class WhisperDataset(Dataset):
    def __init__(self, dataset: IterableDatasetDict, augmentation=None):
        self.dataset = dataset
        self.augmentation = augmentation

    def __len__(self):
        return len(self.dataset)

    def prepare_dataset(self, batch):
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_values"] = \
            feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch['transcript'] = normalizer(batch['transcript'])
        # encode target text to label ids
        batch["labels"] = tokenizer(batch["transcript"], truncation=True, max_length=256,
                                    padding="max_length").input_ids
        return batch

    def __getitem__(self, idx):
        return self.prepare_dataset(self.dataset[idx])


# Prepare DataLoader for training and evaluation
train_dataloader = DataLoader(WhisperDataset(common_voice["train"]),
                              batch_size=CFG.batch_size,
                              collate_fn=data_collator, pin_memory=True, num_workers=CFG.num_workers, shuffle=True)
eval_dataloader = DataLoader(WhisperDataset(common_voice["test"]), batch_size=16,
                             collate_fn=data_collator,
                             pin_memory=True, num_workers=CFG.num_workers)
total_steps = len(train_dataloader) * CFG.epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

model, train_dataloader, eval_dataloader = accelerate.prepare(model, train_dataloader, eval_dataloader)
metric = evaluate.load("cer")

# Custom training loop
for epoch in range(CFG.epochs):

    best_cer = np.inf
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

    cer = metric.compute(predictions=predictions, references=labels) * 100
    accelerate.log({"cer": cer})
    accelerate.print(f"Epoch {epoch} CER: {cer}")
    model = accelerate.unwrap_model(model)
    accelerate.print(
        f"Average training loss for epoch {epoch}: {total_loss / len(train_dataloader)}")
    accelerate.wait_for_everyone()
if accelerate.is_main_process:
    model.push_to_hub(f"whisper-large-v3-chinese-finetune-custom-dataset-augmentations", safe_serialization=True)
    processor.push_to_hub(f"whisper-large-v3-chinese-finetune-custom-dataset-augmentations", )

# Save the model
accelerate.end_training()
