import json
import random
import numpy as np

# def main():
#     url_to_code={}

#     with open('./data.jsonl') as f:
#         for line in f:
#             line=line.strip()
#             js=json.loads(line)
#             url_to_code[js['idx']]=js['func']

#     data_0 = []
#     data_1 = []

#     with open("./train_sampled.txt") as f:
#         for line in f:
#             line=line.strip()
#             url1, url2, label = line.split('\t')
#             if url1 not in url_to_code or url2 not in url_to_code:
#                 continue
#             if label=='0':
#                 label=0
#                 data_0.append((url1, url2, label))
#             else:
#                 label=1
#                 data_1.append((url1, url2, label))

#         random.shuffle(data_0)
#         random.shuffle(data_1)

#         label_data = data_0[:int(len(data_0)/2)] + data_1[:int(len(data_1)/2)+1]
#         unlabel_data = data_0[int(len(data_0)/2):] + data_1[int(len(data_1)/2)+1:]

#         random.shuffle(label_data)
#         random.shuffle(unlabel_data)
        
#     with open("./unlabel.txt", "w") as wf:
#         for d in unlabel_data:
#             wf.write(d[0]+"\t"+d[1]+"\t"+str(d[2])+'\n')
#     print(len(unlabel_data))

#     with open("./label.txt", "w") as wf:
#         for d in label_data:
#             wf.write(d[0]+"\t"+d[1]+"\t"+str(d[2])+'\n')
#     print(len(label_data))


# if __name__ == "__main__":
#     main()

# preds = np.load("preds_unlabel.npy").astype(int).tolist()
# l = []
# data = []
# with open("./unlabel_train.txt") as f:
#    for line in f:
#         line = line.strip()
#         url1, url2, label, p = line.split('\t')
#         data.append((url1, url2, label))
#         l.append(int(label))

# with open("./unlabel_train.txt", "w") as wf:
#     for d, p in zip(data, preds):
#         wf.write(d[0]+"\t"+d[1]+"\t"+str(-1)+"\t"+str(p)+'\n')
# # print(np.mean(np.array(preds)==np.array(l)))

# data_0 = []
# data_1 = []
# with open("./test.txt") as f:
#    for line in f:
#         line = line.strip()
#         url1, url2, label = line.split('\t')
#         if label=='0':
#             label=0
#             data_0.append((url1, url2, label))
#         else:
#             label=1
#             data_1.append((url1, url2, label))

# random.shuffle(data_0)
# random.shuffle(data_1)
# t = []

# # while len(t) < 50:
# #     d = random.sample(data_1, 1)[0]
# #     q = d[0]
# #     a = d[1]
# #     s = [(q, a, d[2])]
# #     samples = random.sample(data_1, 100)
# #     for sample in samples:
# #         if not (q, sample[0], 1) in data_1 or not (sample[0], q, 1) in data_1:
# #             if not (q, sample[0], 0) in s:
# #                 s.append((q, sample[0], 0))
# #     if len(s) > 50:
# #         t.append(s[:50])
# #     print(len(s))


# while len(t) < 100:
#     d = random.sample(data_1, 1)[0]
#     q = d[0]
#     a = d[1]
#     s = [(q, a, d[2])]
#     # samples = random.sample(data_1, 100)
#     for i in data_0:
#         if i[0] == q:
#             s.append((q, i[1], 0))
#         elif i[1] == q:
#             s.append((q, i[0], 0))
#     if len(s) > 8:
#         t.append(s[:8])
#     # print(len(s))
# print(len(t))
# with open("./test_mrr.txt", "w") as wf:
#     for p in t:
#         for d in p:
#             wf.write(d[0]+"\t"+d[1]+"\t"+str(d[2])+'\n')


# preds = np.load("preds_unlabel.npy").astype(int).tolist()
# l = []
# data = []
# with open("./unlabel_train.txt") as f:
#    for line in f:
#         line = line.strip()
#         url1, url2, label, p = line.split('\t')
#         data.append((url1, url2, label, p))
#         l.append(int(p))

# with open("./unlabel_train.txt", "w") as wf:
#     for d, p in zip(data, preds):
#         wf.write(d[0]+"\t"+d[1]+"\t"+str(-1)+"\t"+str(p)+'\n')

print(len(np.load("preds_unlabel.npy").tolist()))
