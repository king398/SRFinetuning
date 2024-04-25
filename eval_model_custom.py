import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import evaluate
from tqdm import tqdm
from datasets import load_dataset

device = "cuda:1" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model_id = "BELLE-2/Belle-whisper-large-v3-zh"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype,
)
model.use_cache = False

model.to(device)

processor = AutoProcessor.from_pretrained("BELLE-2/Belle-whisper-large-v3-zh")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"task": "transcribe", "language": "zh"},

)
dataset = load_dataset("Mithilss/L2TS")['test']
# split  the dataset into train and test
def iterate_data(dataset):
    for i, item in tqdm(enumerate(dataset)):
        yield item["audio"]



metric = evaluate.load("cer")
predictions = []
labels = dataset['transcript']
for out in pipe(iterate_data(dataset), batch_size=32):
    predictions.append(out["text"])
predictions = [predictions[i] for i in range(len(predictions)) if len(labels[i]) > 0]
labels = [labels[i] for i in range(len(labels)) if len(labels[i]) > 0]

metric.add_batch(predictions=predictions, references=labels, )
print(f" CER: {metric.compute() * 100}")
#  CER: 153.96737056354152