import os
import glob

files = glob.glob("/mnt/e/Processed Data for L2TS/audio/*")
"""for i in files:
    with open(i, 'r') as f:
        file = f.read()
    new_path = i.split("_")[:2]
    new_path = f"{new_path[0]}_{new_path[1]}.txt"
    print(new_path)
    break
"""
for i in files:
    new_path = i.split(".")[0]
    new_path = new_path.split("_")[:2]
    if len(new_path) == 1:
        new_path_j = new_path[0].split("/")[-1].replace(" ", "_")
        # insert a _ between the spaces
        new_path = f"/mnt/e/Processed Data for L2TS/audio/{new_path_j}.mp3"
    else:
        new_path = f"{new_path[0]}_{new_path[1]}.mp3"
    # rename the file
    os.rename(i, new_path)
    break
