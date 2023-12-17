import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import evaluate
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "Mithilss/whisper-large-v3-chinese-finetune-4-epochs-1e-4-lr-epoch-0"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True, use_flash_attention_2=True, low_cpu_mem_usage=True
)
model.use_cache = False

model.to(device)

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=448,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=False,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "chinese", "task": "transcribe"},
)

dataset = iter(load_dataset("mozilla-foundation/common_voice_13_0", "zh-CN", split="test", streaming=True))
eval_samples = 2000

metric = evaluate.load("cer")
for i in tqdm(range(eval_samples)):
    sample = next(dataset)
    result = pipe(sample["audio"])
    metric.add_batch(predictions=[result['text']], references=[sample["sentence"]], )

print(f" Fine tuned CER: {metric.compute() * 100}")
