from torch.utils.data import Dataset

import numpy as np
import torch


class hypernymy_dataset(Dataset):
    def __init__(self, DATA_DIR, HYPONYMS_FILENAME, HYPERNYMS_FILENAME, VOCAB_FILENAME, num_negs=5):
        self.hyponyms = [line.strip().split("\t")[0] for line in self.read_file(DATA_DIR, HYPONYMS_FILENAME)]
        self.hypernyms = [line.strip().split("\t") for line in self.read_file(DATA_DIR, HYPERNYMS_FILENAME)]
        self.vocab = list(set([line.strip().split("\t")[0] for line in self.read_file(DATA_DIR, VOCAB_FILENAME)]))
        self.all_hypernyms = self.merge_hypernyms()
        self.num_negs = num_negs
        self.word_id_map = dict()
        self.id_word_map = dict()
        self.data_size = len(self.hyponyms)

        self.generate_mappings()
        # print(self.hyponyms[1443])
        # print(self.hypernyms[1443])
        # print(self.get_negative_samples(1443))
        # print(self.vocab[1443])

    def read_file(self, DATA_DIR, FILENAME):
        f = open(DATA_DIR + FILENAME)
        lines = f.readlines()
        f.close()
        modified_lines = list()
        for line in lines:
            words = line.strip().split("\t")
            modified_line = ""
            for word in words:
                underscored_word = "_".join(word.split(" ")).lower()
                modified_line += underscored_word + "\t"
            modified_lines.append(modified_line)
        return modified_lines

    def merge_hypernyms(self):
        all_hypernyms = list()
        for hypers in self.hypernyms:
            all_hypernyms += hypers
        return list(set(all_hypernyms))

    def generate_mappings(self):
        for ndx, word in enumerate(self.vocab):
            self.word_id_map[word] = ndx
            self.id_word_map[ndx] = word

    def get_negative_samples(self, ndx):
        positives = self.hypernyms[ndx]
        negatives = list()
        count = 0
        while count < self.num_negs*len(positives):
            rand_neg = self.all_hypernyms[np.random.randint(0, len(self.all_hypernyms))]
            if rand_neg not in positives and rand_neg not in negatives:
                negatives.append(rand_neg)
                count += 1
        return negatives

    @staticmethod
    def collate(batches):
        u = [u for b in batches for u, _, _ in b if len(b) > 0]
        v = [v for b in batches for _, v, _ in b if len(b) > 0]
        neg = [neg_v for b in batches for _, _, neg_v in b if len(b) > 0]
        return torch.LongTensor(u), torch.LongTensor(v), torch.LongTensor(neg)

    def __getitem__(self, ndx):
        hyponym = self.hyponyms[ndx]
        hypernyms = self.hypernyms[ndx]
        negative_samples = self.get_negative_samples(ndx)
        items = list()
        start, end = 0, self.num_negs
        for ndx, hypernym in enumerate(hypernyms):
            negatives = negative_samples[start:end]
            start = end
            end += self.num_negs
            items.append((self.word_id_map[hyponym],
                          self.word_id_map[hypernym],
                          [self.word_id_map[neg] for neg in negatives]))
        return items

    def __len__(self):
        return self.data_size


if __name__ == "__main__":
    DATA_DIR = "../../data/" # "./DATA_SemEval2018-Task9/"
    ENGLISH_HYPONYMS_TRAIN_FILE = "training/data/1A.english.training.data.txt"
    MEDICAL_HYPONYMS_TRAIN_FILE = "training/data/2A.medical.training.data.txt"
    MUSIC_HYPONYMS_TRAIN_FILE = "training/data/2B.music.training.data.txt"

    ENGLISH_HYPERNYMS_TRAIN_FILE = "training/gold/1A.english.training.gold.txt"
    MEDICAL_HYPERNYMS_TRAIN_FILE = "training/gold/2A.medical.training.gold.txt"
    MUSIC_HYPERNYMS_TRAIN_FILE = "training/gold/2B.music.training.gold.txt"

    ENGLISH_VOCAB_FILENAME = "vocabulary/english_merged_vocab.txt"
    MEDICAL_VOCAB_FILENAME = "vocabulary/medical_merged_vocab.txt"
    MUSIC_VOCAB_FILENAME = "vocabulary/music_merged_vocab.txt"

    sg_dataset = hypernymy_dataset(DATA_DIR, MUSIC_HYPONYMS_TRAIN_FILE, MUSIC_HYPERNYMS_TRAIN_FILE, MUSIC_VOCAB_FILENAME)
    print(len(sg_dataset))
    # print(sg_dataset[np.random.randint(0, 1500)])
