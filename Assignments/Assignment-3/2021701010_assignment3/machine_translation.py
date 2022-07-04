"""machine_translation.py: Demo Script that given a statement tries to
output its BLEU with pre-trained model."""

__author__ = "Aditya Kumar Singh"

import torch
from NNLM_Model import NNLM
from seq2seq_MT2 import encoder, decoder, Seq2Seq_MT2
from seq2seq import Seq2Seq, Encoder, Decoder, Attention
from torchtext.data.utils import get_tokenizer
import pickle
import argparse

# ===========================================================================================================================

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--model_type', type=str, required=True)  # MT1 or MT2
parser.add_argument('--model', type=str, required=True)
# Parse the argument
args = parser.parse_args()
# To run enter
# python machine_translation.py --model_type MT1 --model MT1_3
# ===========================================================================================================================

# Mention Device...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct tokenizer
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

if args.model_type == "MT1":
    # Load Vocab
    with open("./data/ted-talks-corpus/en_vocab_ted.pkl", "rb") as fp:
        en_vocab = pickle.load(fp)
    with open("./data/ted-talks-corpus/fr_vocab_ted.pkl", "rb") as fp:
        fr_vocab = pickle.load(fp)
    ################################# Defining model class which we're importing ####
    # Define Hyper-Parameters of MOdel
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(fr_vocab)
    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    ENC_HID_DIM = 64
    DEC_HID_DIM = 64
    ATTN_DIM = 8
    ENC_DROPOUT = 0
    DEC_DROPOUT = 0

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, DEC_DROPOUT, attn)
    _ = Seq2Seq(enc, dec, device).to(device)

    # Load Model
    model = torch.load(f"./models/{args.model}.pth", map_location=device)

elif args.model_type == "MT2":
    # Load Vocab
    with open("./data/ted-talks-corpus/en_vocab_ted_MT2.pkl", "rb") as fp:
        en_vocab = pickle.load(fp)
    with open("./data/ted-talks-corpus/fr_vocab_ted_MT2.pkl", "rb") as fp:
        fr_vocab = pickle.load(fp)
    ################################# Defining model class which we're importing ####
    # Define Hyper-Parameters of MOdel
    ENC_VOCAB_SIZE = len(en_vocab)
    DEC_VOCAB_SIZE = len(fr_vocab)
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ENC_DROPOUT = 0
    DEC_DROPOUT = 0
    # Initialize Weight matrix...
    nnlm = NNLM(ENC_VOCAB_SIZE, ENC_EMB_DIM, ENC_HID_DIM, 2, 0)
    enc = encoder(torch.zeros((32085, 512)), nnlm, 0)
    dec = decoder(DEC_VOCAB_SIZE, torch.zeros(
        (55104, 512)), DEC_HID_DIM, nnlm, 0)
    _ = Seq2Seq_MT2(enc, dec, device=device).to(device)

    # Load Model
    model = torch.load(f"./models/{args.model}.pth", map_location=device)
else:
    raise ValueError("Please provide correct Model Type!")

# ===========================================================================================================================

# Define some stuffs
PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']
# Input English Sentence
input_sent = input(f"Enter a English sentence:\n")
# Set the model to eval mode...
model.eval()
with torch.no_grad():
    en_item = torch.tensor([en_vocab[token] for token in en_tokenizer(input_sent)], dtype=torch.long)
    # Giving a blank sentence as input for FRENCH...
    fr_item = torch.tensor([PAD_IDX for _ in range(len(en_item))], dtype=torch.long)
    if args.model_type == "MT2":
        src = torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor(
            [EOS_IDX])], dim=0).to(device)  # .transpose(0, 1)
        trg = torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor(
            [EOS_IDX])], dim=0).to(device)  # .transpose(0, 1)
        # print(f"src shape = {src.shape}, trg shape = {trg.shape}")
        output = model(src.unsqueeze(dim=0), trg.unsqueeze(dim=0), 0)  # For Batch First
        output = output.squeeze(dim=0)
    elif args.model_type == "MT1":
        src = torch.cat([torch.tensor([BOS_IDX]), en_item,
                        torch.tensor([EOS_IDX])], dim=0).to(device)
        trg = torch.cat([torch.tensor([BOS_IDX]), fr_item,
                        torch.tensor([EOS_IDX])], dim=0).to(device)
        output = model(src.unsqueeze(dim=1), trg.unsqueeze(dim=1), 0)  # For Batch Second
        output = output.squeeze(dim=1)
    # Prediction in terms of token index
    pred = torch.argmax(output, dim=1)
    # Omitting start Token...
    pred_indexes = [int(ndx.cpu()) for ndx in pred[1:]]
    pred_st = " ".join(fr_vocab.lookup_tokens(pred_indexes))
    pred_st = pred_st.replace("\n", "").strip()

# Print the sentence...
print(f"Translated Sentence: {pred_st}")
