import pickle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import hypernymy_dataset
from model import SkipGramModel


class SkipGram_Embeddings_generator:
    def __init__(self, DATA_DIR, HYPONYMS_FILENAME, HYPERNYMS_FILENAME, VOCAB_FILENAME, OUTFILE, iterations=10, embedding_len=300):

        self.dataset = hypernymy_dataset(DATA_DIR, HYPONYMS_FILENAME, HYPERNYMS_FILENAME, VOCAB_FILENAME)
        self.dataloader = DataLoader(self.dataset, batch_size=32,
                                     shuffle=True, num_workers=10, collate_fn=self.dataset.collate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_size = len(self.dataset.vocab)
        self.embedding_len = embedding_len
        self.word2vec_sg_model = SkipGramModel(self.emb_size, self.embedding_len).to(self.device)
        self.OUTFILE_name = OUTFILE
        self.iterations = iterations
        self.loss_logs = list()

    def train(self):
        print("TRAINING EMBEDDINGS")
        for iteration in tqdm(range(self.iterations)):
            optimizer = optim.SparseAdam(self.word2vec_sg_model.parameters(), lr=0.00001)
            running_loss = 0.0
            for i, sample_batched in enumerate(self.dataloader):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)
                    optimizer.zero_grad()
                    loss = self.word2vec_sg_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    running_loss = running_loss * 0.9 + loss.item() * 0.1
            self.loss_logs.append(running_loss)
        self.word2vec_sg_model.save_embedding(self.dataset.id_word_map, self.OUTFILE_name+"_sg_emb.txt")
        f = open("{}_model.pkl".format(self.OUTFILE_name), "wb")
        pickle.dump(self.word2vec_sg_model, f)
        f.close()
        self.generate_loss_plots()

    def generate_loss_plots(self):
        x = list(range(len(self.loss_logs)))
        y = self.loss_logs[:]
        plt.plot(x, y)
        plt.xlabel("Epochs")
        plt.ylabel("Running loss (SkipGram)")
        plt.title("Training loss trend for {} embeddings".format(self.OUTFILE_name))
        plt.savefig(self.OUTFILE_name + "_sg_emb_loss.png")

    def write_logs(self):
        f = open(self.OUTFILE_name+"_sg_emb_loss.pkl", "wb")
        pickle.dump(self.loss_logs, f)
        f.close()
        print("CREATED {} FILE".format(self.OUTFILE_name+"_sg_emb_loss.pkl"))


if __name__ == '__main__':
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

    w2v = SkipGram_Embeddings_generator(DATA_DIR,
                                        MEDICAL_HYPONYMS_TRAIN_FILE,
                                        MEDICAL_HYPERNYMS_TRAIN_FILE,
                                        MEDICAL_VOCAB_FILENAME,
                                        OUTFILE="medical2",
                                        iterations=150)
    w2v.train()
    w2v.write_logs()
