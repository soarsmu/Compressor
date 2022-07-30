import json
import random
import numpy as np


# with open("train.jsonl") as f:
#     for line in f:
#         data.append(json.loads(line.strip()))

# random.shuffle(data)

# print(len(data))
# label_data = data[:int(len(data)/2)]
# unlabel_data = data[int(len(data)/2):]

# print(len(label_data))
# print(len(unlabel_data))

# with open("label_train.jsonl","w") as f:
#     for d in label_data:
#         f.write(json.dumps(d)+"\n")

# with open("unlabel_train.jsonl","w") as f:
#     for d in unlabel_data:
#         f.write(json.dumps(d)+"\n")

mix_data = []
data = []
with open("unlabel_train.jsonl") as f:
    for line in f:
        data.append(json.loads(line.strip()))

preds = np.load("preds_unlabel_train.npy").astype(int).tolist()

for pred, d in zip(preds, data):
    d["target"] = -1
    d["pred"] = pred
    mix_data.append(d)

# data = []
# with open("label_train.jsonl") as f:
#     for line in f:
#         data.append(json.loads(line.strip()))

# preds = np.load("preds_label_train.npy").astype(int).tolist()

# for pred, d in zip(preds, data):
#     d["pred"] = pred
#     mix_data.append(d)

# random.shuffle(mix_data)
# print(len(mix_data))

# with open("mixed_train.jsonl","w") as f:
#     for d in mix_data:
#         f.write(json.dumps(d)+"\n")

# mix_data = []
# data = []
# with open("label_train.jsonl") as f:
#     for line in f:
#         data.append(json.loads(line.strip()))

# preds = np.load("preds_label_train.npy").astype(int).tolist()
# print(len(preds), len(data))
# for pred, d in zip(preds, data):
#     d["pred"] = pred
#     mix_data.append(d)

random.shuffle(mix_data)
print(len(mix_data))

with open("unlabel_train.jsonl","w") as f:
    for d in mix_data:
        f.write(json.dumps(d)+"\n")