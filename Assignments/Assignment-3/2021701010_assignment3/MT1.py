import torch
#from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import math
import time

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from seq2seq import Encoder, Decoder, Attention, Seq2Seq

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
# from collections import Counter
from vocabBuilder import build_vocab
import io
import argparse

# ===========================================================================================================================

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--model_number', type=str, required=True)
# Parse the argument
args = parser.parse_args()


#==========================================================================================================
# Mention Device...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#==========================================================================================================
path = "./data/ted-talks-corpus/"


# Construct tokenizer
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')


# Define FilePaths...
train_filepaths = [path+"train.en", path+"train.fr"]
val_filepaths = [path+"dev.en", path+"dev.fr"]
test_filepaths = [path+"test.en", path+"test.fr"]

# Build Vocabulary...
en_vocab = build_vocab(train_filepaths[0], en_tokenizer)
fr_vocab = build_vocab(train_filepaths[1], fr_tokenizer)

# Save the vocab:
with open("./data/ted-talks-corpus/en_vocab_ted.pkl", "wb") as f:
    pickle.dump(en_vocab, f)
with open("./data/ted-talks-corpus/fr_vocab_ted.pkl", "wb") as fp:
    pickle.dump(fr_vocab, fp)
#==========================================================================================================

def data_process(filepaths, tokenizer1, tokenizer2, vocab1, vocab2):
    raw_en_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_fr_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_en, raw_fr) in zip(raw_en_iter, raw_fr_iter):
        tensor_1 = torch.tensor([vocab1[token] for token in tokenizer1(raw_en)], dtype=torch.long)
        tensor_2 = torch.tensor([vocab2[token] for token in tokenizer2(raw_fr)], dtype=torch.long)
        data.append((tensor_1, tensor_2))
    return data

train_data = data_process(train_filepaths, en_tokenizer, fr_tokenizer, en_vocab, fr_vocab)
val_data = data_process(val_filepaths, en_tokenizer, fr_tokenizer, en_vocab, fr_vocab)
test_data = data_process(test_filepaths, en_tokenizer, fr_tokenizer, en_vocab, fr_vocab)

#==========================================================================================================

# Batch Size...
BATCH_SIZE = 8
PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']


def generate_batch(data_batch):
    fr_batch, en_batch = [], []
    for (en_item, fr_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        fr_batch.append(torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    fr_batch = pad_sequence(fr_batch, padding_value=PAD_IDX)
    return en_batch, fr_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

#==========================================================================================================

# Finally, we can train and evaluate this model:

def train(model: nn.Module, 
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(tqdm(iterator)):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(tqdm(iterator)):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

#==========================================================================================================

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#==========================================================================================================

# Define Hyper-Parameters of MOdel
ENC_VOCAB_SIZE = len(en_vocab)
DEC_VOCAB_SIZE = len(fr_vocab)
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LEARNING_RATE = 0.001# Adam default

enc = Encoder(ENC_VOCAB_SIZE, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(DEC_VOCAB_SIZE, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1, verbose=False)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=6, factor=0.1)
print(f'The model has {count_parameters(model):,} trainable parameters')

# Note: when scoring the performance of a language translation model in
# particular, we have to tell the ``nn.CrossEntropyLoss`` function to
# ignore the indices where the target is simply padding.

PAD_IDX = en_vocab.get_stoi()['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

#==========================================================================================================

N_EPOCHS = 40
EARLY_STOP = 0# Applying Early Stop
CLIP = 1
ctr = 0
PATIENCE = 6# patience for early stopping
best_valid_loss = float('inf')
train_loss_lst, val_loss_lst, perp_train, perp_val = [], [], [], []
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)

    # Scheduler Step
    scheduler.step()  # valid_loss when other schedulers are there

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02}/{N_EPOCHS} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.6f} | Train PPL: {math.exp(train_loss):.6f}')
    print(f'\t Val. Loss: {valid_loss:.6f} |  Val. PPL: {math.exp(valid_loss):.6f}')
    train_loss_lst.append(train_loss); perp_train.append(math.exp(train_loss))
    val_loss_lst.append(valid_loss); perp_val.append(math.exp(valid_loss))
    EARLY_STOP += 1
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model, f'./models/MT1_{args.model_number}.pth')
        print("Improved! Model Saved...\n\n")
        ctr = 0
    else:
        ctr += 1
        print("\n\n")
    if ctr > PATIENCE:
        break

test_loss = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.6f} | Test PPL: {math.exp(test_loss):7.6f} |')

#=========================================================================================================
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
         perp_train, 'r', label="Train Perplexity")
plt.plot([k for k in range(1, EARLY_STOP+1)],
         perp_val, 'b', label="Val Perplexity")
plt.title("Accuracy")
plt.legend(loc="best")
plt.savefig(f"./plots/MT1_{args.model_number}.png", dpi=300)
# Save params .txt file.
with open(f"./models/MT1_{args.model_number}.txt", "w") as f:
    f.write(f"Batch Size = {BATCH_SIZE}\n")
    f.write(f"Learning Rate = {LEARNING_RATE}\n")
    f.write(f"Patience to Early Stop the Model = {PATIENCE}\n")
    f.write(f"ENCODER Vocab Size = {ENC_VOCAB_SIZE}\n")
    f.write(f"DECODER Vocab Size = {DEC_VOCAB_SIZE}\n")
    f.write(f"ENCODER Embedding Dimension = {ENC_EMB_DIM}\n")
    f.write(f"DECODER Embedding Dimension = {DEC_EMB_DIM}\n")
    f.write(f"ENCODER Encoding Hidden Dimension = {ENC_HID_DIM}\n")
    f.write(f"DECODER Encoding Hidden Dimension = {DEC_HID_DIM}\n")
    f.write(f"ENCODER Dropout = {ENC_DROPOUT}\n")
    f.write(f"DECODER Dropout = {DEC_DROPOUT}\n")
    f.write(f"Early Stopping happened at Step = {EARLY_STOP}\n")
    f.write(f"Number of Epochs = {N_EPOCHS}\n")
    f.write(f"\n")
