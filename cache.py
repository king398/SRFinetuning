from datasets import load_dataset

dataset_name = "mozilla-foundation/common_voice_13_0"
split_name = "zh-CN"

dataset = load_dataset(dataset_name, split_name)
