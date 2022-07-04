# from __future__ import unicode_literals, print_function, division
import torch
import matplotlib.pyplot as plt
import math
import time
import pickle
from tqdm import tqdm

from NNLM_Model import NNLM
import torch.optim as optim
import torch.nn.functional as F

from seq2seq_MT2 import encoder, decoder, Seq2Seq_MT2
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from vocabBuilder import build_vocab
import io
import argparse

# ===========================================================================================================================

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--english_model_num', type=str, required=True)
parser.add_argument('--french_model_num', type=str, required=True)
parser.add_argument('--model_number', type=str, required=True)
# Parse the argument
args = parser.parse_args()

# ===========================================================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=========================================================================================================
# Construct tokenizer
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

#=========================================================================================================
# TED DATASET
# Define FilePaths...
path = "./data/ted-talks-corpus/"
train_filepaths = [path+"train.en", path+"train.fr"]
val_filepaths = [path+"dev.en", path+"dev.fr"]
test_filepaths = [path+"test.en", path+"test.fr"]

# Build Vocabulary...
en_vocab = build_vocab(train_filepaths[0], en_tokenizer)
fr_vocab = build_vocab(train_filepaths[1], fr_tokenizer)
print(f"\n\nTED Vocab Size:\nEnglish: {len(en_vocab)}\nFrench: {len(fr_vocab)}")
#=========================================================================================================
# NEWS DATASET

# import it from while saved with NNLLM.py
with open("./data/europarl-corpus/en_vocab.pkl", "rb") as fp:
    en_vocab1 = pickle.load(fp)
with open("./data/news-crawl-corpus/fr_vocab.pkl", "rb") as f:
    fr_vocab1 = pickle.load(f)
print(f"\n\nEUROPARL and NEWS Vocab Size:\nEnglish: {len(en_vocab1)}\nFrench: {len(fr_vocab1)}")
#=========================================================================================================
# UPDATE THE SMALL VOCAB
for key in en_vocab.get_stoi().keys():
    if en_vocab1[key] == en_vocab1["<unk>"] and key != "<unk>":
        en_vocab1.append_token(key)

for key in fr_vocab.get_stoi().keys():
    if fr_vocab1[key] == fr_vocab1["<unk>"] and key != "<unk>":
        fr_vocab1.append_token(key)
print(f"After updating the TED Vocab, the Size becomes:\nEnglish: {len(en_vocab1)}\nFrench: {len(fr_vocab1)}")
print("\n\n")
#=========================================================================================================

# Save the vocab:
with open("./data/ted-talks-corpus/en_vocab_ted_MT2.pkl", "wb") as f:
    pickle.dump(en_vocab1, f)
with open("./data/ted-talks-corpus/fr_vocab_ted_MT2.pkl", "wb") as fp:
    pickle.dump(fr_vocab1, fp)

#=========================================================================================================
def data_process(filepaths, tokenizer1, tokenizer2, vocab1, vocab2):
    raw_en_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_fr_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_en, raw_fr) in zip(raw_en_iter, raw_fr_iter):
        en_tensor_ = torch.tensor(
            [vocab1[token] for token in tokenizer1(raw_en)], dtype=torch.long)
        fr_tensor_ = torch.tensor(
            [vocab2[token] for token in tokenizer2(raw_fr)], dtype=torch.long)
        data.append((en_tensor_, fr_tensor_))
    return data


train_data = data_process(train_filepaths, en_tokenizer, fr_tokenizer, en_vocab1, fr_vocab1)
val_data = data_process(val_filepaths, en_tokenizer, fr_tokenizer, en_vocab1, fr_vocab1)
test_data = data_process(test_filepaths, en_tokenizer, fr_tokenizer, en_vocab1, fr_vocab1)
#=========================================================================================================
# Batch Size...
BATCH_SIZE = 6
PAD_IDX = en_vocab1['<pad>']
BOS_IDX = en_vocab1['<bos>']
EOS_IDX = en_vocab1['<eos>']


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (en_item, de_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=True)#, batch_first=True
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX,
                            batch_first=True)  # , batch_first=True
    return en_batch, de_batch  # .transpose(0, 1)


train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
#=========================================================================================================
# for k, (src, trg) in enumerate(train_iter):
#     if k == 2:
#         break
#     print(src.shape, trg.shape)
#=========================================================================================================

# Defining the pre-trained model Class to assist in Pytorch Loading...

_ = NNLM(16000, 512, 256, 3, 0.2)

#=========================================================================================================
enc_model = torch.load(f"./models/{args.english_model_num}.pth", map_location=device)
dec_model = torch.load(f"./models/{args.french_model_num}.pth", map_location=device)
#=========================================================================================================
# Updating the Embedding Dictionary with new words that might exist in TED (but not in Europarl)
weight_arr_enc = enc_model.embedding.weight; weight_arr_enc.requires_grad = False
weight_arr_dec = dec_model.embedding.weight; weight_arr_dec.requires_grad = False

print(f"Embedding Weights Shape before Appending(ENCODER): {weight_arr_enc.shape}")
w = torch.randn((len(en_vocab1) - weight_arr_enc.shape[0]), weight_arr_enc.shape[1]).to(device)
weight_arr_enc = torch.concat((weight_arr_enc, w), dim=0)
weight_arr_enc.requires_grad = True
print(f"Embedding Weights Shape after Appending(ENCODER): {weight_arr_enc.shape}")

print(f"Embedding Weights Shape before Appending(DECODER): {weight_arr_dec.shape}")
wf = torch.randn((len(fr_vocab1) - weight_arr_dec.shape[0]), weight_arr_dec.shape[1]).to(device)
weight_arr_dec = torch.concat((weight_arr_dec, wf), dim=0)
weight_arr_dec.requires_grad = True
print(f"Embedding Weights Shape after Appending(DECODER): {weight_arr_dec.shape}")
print("\n\n")
#=========================================================================================================
print("MOdel Summary:\n", enc_model)
print("List of Layers:\n", list(enc_model.children())[:-2])
#=========================================================================================================
#
#                                            MODEL TRAINING
#
#=========================================================================================================

def train(model: nn.Module, iterator: torch.utils.data.DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(tqdm(iterator)):
        src, trg = src.to(device), trg.to(device)
        # print(f"Target Shape = {trg.shape}")
        optimizer.zero_grad()

        output = model(src, trg)
        # print(f"Output Shape before change = {output.shape}")
        # SINCE <sos> token is gone
        output = output[:, 1:].reshape(-1, output.shape[-1])
        # print(f"Output Shape = {output.shape}")
        trg = trg[:, 1:].reshape(-1)
        # print(f"Target Shape after change = {trg.shape}\n\n\n")
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
#=========================================================================================================


def evaluate(model: nn.Module, iterator: torch.utils.data.DataLoader, criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(tqdm(iterator)):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)
            # After <sos> token
            output = output[:, 1:].reshape(-1, output.shape[-1])# not view
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#=========================================================================================================

# Define Hyper-Parameters of MOdel
ENC_VOCAB_SIZE = len(en_vocab1)
DEC_VOCAB_SIZE = len(fr_vocab1)
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LEARNING_RATE = 0.001

# Create instance of Seq2seq model...
enc = encoder(weight_arr_enc=weight_arr_enc, pre_trained_encoder=enc_model, dropout=ENC_DROPOUT)
dec = decoder(VOCAB_SIZE=DEC_VOCAB_SIZE, weight_arr_dec=weight_arr_dec, dec_hid_dim=DEC_HID_DIM,
              pre_trained_decoder=dec_model, dropout=DEC_DROPOUT)
model = Seq2Seq_MT2(enc, dec, device).to(device)

#=========================================================================================================

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

PAD_IDX = en_vocab1.get_stoi()['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
#=========================================================================================================

N_EPOCHS = 50
EARLY_STOP = 0
CLIP = 1
ctr = 0
PATIENCE = 6

best_valid_loss = float('inf')
train_loss_lst, val_loss_lst = [], []
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)

    # Scheduler Step
    scheduler.step()

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'\nEpoch: {epoch+1:02}/{N_EPOCHS} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    train_loss_lst.append(train_loss)
    val_loss_lst.append(valid_loss)
    EARLY_STOP += 1
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model, f'./models/MT2_{args.model_number}.pth')
        print("Improved! Model Saved...\n\n")
        ctr = 0
    else:
        ctr += 1
        print("\n\n")
    if ctr > PATIENCE:
        break

test_loss = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
#=========================================================================================================
# Plot the model...
plt.figure(figsize=(20, 20))
# plt.subplot(2, 1, 1)
plt.plot([k for k in range(1, EARLY_STOP+1)],
         train_loss_lst, 'r', label="Train Loss")
plt.plot([k for k in range(1, EARLY_STOP+1)],
         val_loss_lst, 'b', label="Val Loss")
plt.title("Loss")
plt.legend(loc="best")
# plt.subplot(2, 1, 2)
# plt.plot([k for k in range(1, EARLY_STOP+1)],
#          train_acc_lst, 'r', label="Train Acc")
# plt.plot([k for k in range(1, EARLY_STOP+1)],
#          val_acc_lst, 'b', label="Val Acc")
# plt.title("Accuracy")
# plt.legend(loc="best")
plt.savefig(f"./plots/MT2_{args.model_number}.png", dpi=300)
# Save params .txt file.
with open(f"./models/MT2_{args.model_number}.txt", "w") as f:
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
