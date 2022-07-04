import numpy as np
import torch
from torch.utils.data import Dataset
np.random.seed(23)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count):

        self.negative_words = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Total Words Read So far: " + str(int(self.token_count / 1000000)) + "M words.")
        w_id = 1# -1 reserved for starting tag of sentence.
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = w_id
            self.id2word[w_id] = w
            self.word_frequency[w_id] = c
            w_id += 1
        self.word2id["<SPACE>"] = 0
        self.id2word[0] = "<SPACE>"
        self.word_frequency[0] = 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(0.0001 / f) + (0.0001 / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = np.sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for w_id, c in enumerate(count):
            self.negative_words += [w_id] * int(c)
        self.negative_words = np.array(self.negative_words)
        np.random.shuffle(self.negative_words)

    def getNegatives(self, target, size):
        response = self.negative_words[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negative_words)
        if len(response) != size:
            return np.concatenate((response, self.negative_words[0:self.negpos]))
        return [word_id if word_id != target else response[k-1] for k, word_id in enumerate(response)]


# -----------------------------------------------------------------------------------------------------------------

class Word2vecDataLoader(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            # Check if Line
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()
            if len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = self.window_size//2 #np.random.randint(1, self.window_size)
                    cbow_data = []
                    for i, v in enumerate(word_ids):
                        tmp_lst = []
                        for u in word_ids[max(i - boundary, 0):i + boundary + 1]:# as python handles max indexing case so no need to take min
                            if u!=v:
                                tmp_lst.append(u)
                        if len(tmp_lst) < 2*boundary:
                            tmp_lst += [0]*(2*boundary - len(tmp_lst))
                        cbow_data.append((tmp_lst, v, self.data.getNegatives(v, 5)))
                    return cbow_data

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""
class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        # self.u_embeddings(pos_u)
        emb_u = self.u_embeddings(pos_u)
        emb_u = torch.mean(emb_u, axis=1)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for w_id, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[w_id]))
                f.write('%s %s\n' % (w, e))

class CBOWTrainer:
    def __init__(self, input_file, output_file, emb_dimension=350, batch_size=128, window_size=9, iterations=25,
                 initial_lr=0.001, min_count=3):

        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataLoader(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=25, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)

if __name__ == '__main__':
    w2v = CBOWTrainer(input_file="../models/train_data.txt", output_file="../models/word_embeddings.txt")
    w2v.train()
