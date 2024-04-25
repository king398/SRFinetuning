import glob
import re
import datasets
import os

audios = glob.glob("/mnt/e/Processed Data for L2TS/audio/*")
transcripts = glob.glob("/mnt/e/Processed Data for L2TS/Text/*")
# remove the processed and DONE from audio files
audios_new = [i.replace("processed", "") for i in audios]
audios_new = [i.replace("DONE", "") for i in audios_new]

audio_transcript = {}
for i, audio in enumerate(audios):
    audio_name = audio.split("/")[-1].split(".")[0]
    for transcript in transcripts:
        transcript_name = transcript.split("/")[-1].split(".")[0]
        if re.match(f"^{re.escape(audio_name)}", transcript_name):
            print(audio, transcript)
            # load the transcript
            with open(transcript, 'r') as f:
                transcript = f.read()
            audio_transcript[audios_new[i]] = transcript

            break
keys_to_remove = []
for key in audio_transcript.keys():
    if not os.path.exists(key):
        print(f"Removing {key} from the dataset")
        keys_to_remove.append(key)

for key in keys_to_remove:
    del audio_transcript[key]
# create a huggingface dataset
dataset = datasets.Dataset.from_dict(
    {"audio": audio_transcript.keys(), "transcript": audio_transcript.values()}).cast_column("audio", datasets.Audio())
# push this as a private dataset to the hub
dataset.push_to_hub("L2TS", private=True)
