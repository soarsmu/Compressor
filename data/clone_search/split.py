import json
import random

def main():
    url_to_code={}

    with open('./data.jsonl') as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            url_to_code[js['idx']]=js['func']

    data_0 = []
    data_1 = []

    with open("./train_sampled.txt") as f:
        for line in f:
            line=line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label=='0':
                label=0
                data_0.append((url1, url2, label))
            else:
                label=1
                data_1.append((url1, url2, label))

        random.shuffle(data_0)
        random.shuffle(data_1)

        label_data = data_0[:int(len(data_0)/2)] + data_1[:int(len(data_1)/2)+1]
        unlabel_data = data_0[int(len(data_0)/2):] + data_1[int(len(data_1)/2)+1:]

        random.shuffle(label_data)
        random.shuffle(unlabel_data)
        
    with open("./unlabel.txt", "w") as wf:
        for d in unlabel_data:
            wf.write(d[0]+"\t"+d[1]+"\t"+str(d[2])+'\n')
    print(len(unlabel_data))

    with open("./label.txt", "w") as wf:
        for d in label_data:
            wf.write(d[0]+"\t"+d[1]+"\t"+str(d[2])+'\n')
    print(len(label_data))


if __name__ == "__main__":
    main()