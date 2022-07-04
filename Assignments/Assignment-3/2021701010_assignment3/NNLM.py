# import torchtext
import torch
# from typing import Tuple
import matplotlib.pyplot as plt

from vocabBuilder import build_vocab  # yield_tokens,
# import math
import time
from nltk.util import ngrams
from tqdm import tqdm
from NNLM_Model import NNLM
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
import pickle
import io
import argparse

# ===========================================================================================================================

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--model_number', type = str, required=True)
# Parse the argument
args = parser.parse_args()

# ===========================================================================================================================
# Mention Device...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.lang == "ENGLISH":
    path = "./data/europarl-corpus/"
    # Construct tokenizer
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    # Define FilePaths...
    train_filepaths = path+"train.europarl"
    # Build Vocabulary...
    en_vocab = build_vocab(train_filepaths, en_tokenizer)
    # Save the Vocabulary
    with open(path+"en_vocab.pkl", "wb") as f:
        pickle.dump(en_vocab, f)
    PAD_IDX = en_vocab['<pad>']
    BOS_IDX = en_vocab['<bos>']
    EOS_IDX = en_vocab['<eos>']
    VOCAB_SIZE = len(en_vocab)
elif args.lang == "FRENCH":
    path = "./data/news-crawl-corpus/"
    # Construct tokenizer
    fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
    # Define FilePaths...
    train_filepaths = path+"train.news"
    # Build Vocabulary...
    fr_vocab = build_vocab(train_filepaths, fr_tokenizer)
    # Save the Vocabulary
    with open(path+"fr_vocab.pkl", "wb") as f:
        pickle.dump(fr_vocab, f)
    PAD_IDX = fr_vocab['<pad>']
    BOS_IDX = fr_vocab['<bos>']
    EOS_IDX = fr_vocab['<eos>']
    VOCAB_SIZE = len(fr_vocab)
else:
    raise ValueError("Please ENter Correct Language Name...")

# ===========================================================================================================================

def ngrams_data_process(filepath, n, tokenizer, vocab):
    raw_lang_iter = iter(io.open(filepath, encoding="utf8"))
    data = []  # ; ctr = 0
    assert n > 1, "Atleast BiGrams Possible"
    for raw_sent in raw_lang_iter:
        # ADDing HERE STARTING (n-1) AND ENDING TAGS...
        token_lst = tokenizer(raw_sent)
        if len(token_lst) >= n:
            tokens_lst = [BOS_IDX]+[vocab[token]
                                    for token in token_lst]  # +[EOS_IDX] [BOS_IDX]*abs(n-2)+
        else:
            continue
        grams_lst = list(ngrams(tokens_lst, n))
        for gram in grams_lst:
            data.append(list(gram))
        # gram_tensor = torch.tensor(gram, dtype=torch.long)
        # ctr += 1
    print(f"Total {n} Grams in {filepath.split('/')[-1]} = ", len(data))
    return data


n_gram_n = 5
if args.lang == "ENGLISH":
    train_data = ngrams_data_process(
        path+"train.europarl", n_gram_n, en_tokenizer, en_vocab)
    val_data = ngrams_data_process(
        path+"dev.europarl", n_gram_n, en_tokenizer, en_vocab)
    test_data = ngrams_data_process(
        path+"test.europarl", n_gram_n, en_tokenizer, en_vocab)
elif args.lang == "FRENCH":
    train_data = ngrams_data_process(
        path+"train.news", n_gram_n, fr_tokenizer, fr_vocab)
    val_data = ngrams_data_process(
        path+"dev.news", n_gram_n, fr_tokenizer, fr_vocab)
    test_data = ngrams_data_process(
        path+"test.news", n_gram_n, fr_tokenizer, fr_vocab)

# ===========================================================================================================================


def generate_batch(data_batch):
    input_ngram, target = [], []
    for tensorGram in data_batch:
        # n-1 words from tuple as input and rest one for target--> Indexing Starts from One
        input_ngram.append(tensorGram[:-1])
        target.append(tensorGram[-1])
    # No need for paading as all TensorGram are of identical length...
    return torch.tensor(input_ngram, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# ===========================================================================================================================


# Batch Size...
BATCH_SIZE = 256

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
val_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                      shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)

# ===========================================================================================================================

def train(model: nn.Module, iterator: torch.utils.data.DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, clip: float):
    """Train the NNLM model"""
    model.train()

    epoch_loss = epoch_acc = 0

    for _, (src, trg) in enumerate(tqdm(iterator)):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        model_output, _ = model(src)
        # Reshaping the Target from (batch_size, 1) to (batch_Size,) ---> Squeezing
        trg = trg.view(-1)
        loss = criterion(model_output, trg)
        loss.backward()

        # BAtch Accuracy
        epoch_acc += 100*(model_output.argmax(dim=1) == trg).float().mean()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        # print(f"Epoch Loss = {loss.item()}.\n\n")
    return epoch_loss / len(iterator), (epoch_acc/len(iterator)).cpu()


# ===========================================================================================================================

def evaluate(model: nn.Module, iterator: torch.utils.data.DataLoader, criterion: nn.Module):

    model.eval()

    epoch_loss = epoch_acc = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(tqdm(iterator)):
            src, trg = src.to(device), trg.to(device)

            output, _ = model(src)
            trg = trg.view(-1)
            loss = criterion(output, trg)

            # Calculate Barch Accuracy
            epoch_acc += 100*(output.argmax(dim=1) == trg).float().mean()

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), (epoch_acc/len(iterator)).cpu()


# ===========================================================================================================================
# Define Hyper-Parameters of MOdel
EMB_DIM = 512
HID_DIM = 256
NUM_STACKS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001

model = NNLM(VOCAB_SIZE=VOCAB_SIZE, emb_dim=EMB_DIM, enc_hid_dim=HID_DIM,
              num_stacks=NUM_STACKS, dropout=DROPOUT).to(device)

#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()

# To initialize weights manually
def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Fine-Tune the Model
# model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1, last_epoch=-1, verbose=False)
#ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
print(f'The model has {count_parameters(model):,} trainable parameters')

# ===========================================================================================================================


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 100
EARLY_STOP = 0
CLIP = 1
ctr = 0
PATIENCE = 10

best_valid_loss = float('inf')
train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst = [], [], [], []
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss, val_acc = evaluate(model, val_iter, criterion)

    # Scheduler Step
    scheduler.step()  # valid_loss when other schedulers are there

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(
        f'\nEpoch: {epoch+1:02}/{N_EPOCHS} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.9f} | Train Accuracy: {train_acc:7.9f}')
    print(f'\t Val. Loss: {valid_loss:.9f} | Val Accuracy: {val_acc:7.9f}')
    train_loss_lst.append(train_loss)
    train_acc_lst.append(train_acc)
    val_loss_lst.append(valid_loss)
    val_acc_lst.append(val_acc)
    EARLY_STOP += 1
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model, f'./models/NNLM_{args.model_number}.pth')
        print("Improved! Model Saved...\n\n")
        ctr = 0
    else:
        ctr += 1
        print("\n\n")
    if ctr > PATIENCE:
        break
test_loss, test_acc = evaluate(model, test_iter, criterion)

# | Test PPL: {math.exp(test_loss):7.3f} |
print(f'| Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:7.3f}')

# Plot the model...
plt.figure(figsize=(20, 20))
plt.subplot(2, 1, 1)
plt.plot([k for k in range(1, EARLY_STOP+1)],
         train_loss_lst, 'r', label="Train Loss")
plt.plot([k for k in range(1, EARLY_STOP+1)],
         val_loss_lst, 'b', label="Val Loss")
plt.title("Loss")
plt.legend(loc="best")
plt.subplot(2, 1, 2)
plt.plot([k for k in range(1, EARLY_STOP+1)],
         train_acc_lst, 'r', label="Train Acc")
plt.plot([k for k in range(1, EARLY_STOP+1)],
         val_acc_lst, 'b', label="Val Acc")
plt.title("Accuracy")
plt.legend(loc="best")
plt.savefig(f"./plots/NNLM_{args.model_number}.png", dpi=300)

# Save params .txt file.
with open(f"./models/NNLM_{args.model_number}.txt", "w") as f:
    f.write(f"Language = {args.lang}\n")
    f.write(f"Batch Size = {BATCH_SIZE}\n")
    f.write(f"n (no. of grams) = {n_gram_n}\n")
    f.write(f"Learning Rate = {LEARNING_RATE}\n")
    f.write(f"Patience to Early Stop the Model = {PATIENCE}\n")
    f.write(f"Vocab Size = {VOCAB_SIZE}\n")
    f.write(f"Embedding Dimension = {EMB_DIM}\n")
    f.write(f"Encoding Hidden Dimension = {HID_DIM}\n")
    f.write(f"No. of Stacked Layers in GRU = {NUM_STACKS}\n")
    f.write(f"Dropout = {DROPOUT}\n")
    f.write(f"Early Stopping happened at Step = {EARLY_STOP}\n")
    f.write(f"Number of Epochs = {N_EPOCHS}\n")
    f.write(f"\n")
