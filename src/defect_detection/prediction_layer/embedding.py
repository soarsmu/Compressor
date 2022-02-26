import json
import gensim
from tqdm import tqdm
from transformers import RobertaTokenizer

class Dataset:
    def __init__(self, fname):
        self.fname = fname
        self.epoch = 0
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    def __iter__(self):
        data = []
        print(f'Epoch: {self.epoch}')
        with open(self.fname) as f:
            for line in f:
                data.append(" ".join(json.loads(line.strip())["func"].split()))
        
        for d in tqdm(data):
            d = self.tokenizer.tokenize(d)[:400]
            yield d

        self.epoch += 1


def train(data, vector_size=300, window=10, min_count=5, workers=10, sg=1, negative=5, epochs=10, max_vocab_size=5000, output_file=None):
    model = gensim.models.Word2Vec(data, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg, negative=negative, epochs=epochs, max_vocab_size=max_vocab_size)
    model.save(output_file)


if __name__ == '__main__':
    sents = Dataset("../../../data/defect_detection/train.jsonl")
    train(sents, vector_size=200, window=10, workers=10,
          epochs=5, output_file='w2v_200d.bin')