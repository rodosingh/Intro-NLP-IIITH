import torch
from NNLM_Model import NNLM

# import math
from tqdm import tqdm
# import torch.nn as nn
# import torch.optim as optim

from sacrebleu import BLEU
from seq2seq_MT2 import encoder, decoder, Seq2Seq_MT2
from seq2seq import Seq2Seq, Encoder, Decoder, Attention
from torchtext.data.utils import get_tokenizer
import pickle
import io
import argparse

# ===========================================================================================================================

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--model_type', type=str, required=True)# MT1 or MT2
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--BLEU_score_filepath', type=str, required=True)
parser.add_argument('--corpus_type', type=str, required=True)#Train or test
# Parse the argument
args = parser.parse_args()
# To run enter
# python MT_eval.py --model_type MT1 --model MT1_3 --BLEU_score_filepath ./2021701010_MT1_train.txt --corpus_type train 
# ===========================================================================================================================

# Mention Device...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct tokenizer
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# Define Sentence level BLEU score and COrpus level BLEU score...
bleu_st = BLEU(effective_order=True)
bleu_corp = BLEU()

# ===========================================================================================================================
# MODEL SPECIFIC

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

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
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
    dec = decoder(DEC_VOCAB_SIZE, torch.zeros((55104, 512)), DEC_HID_DIM, nnlm, 0)
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
real_sentences, translated_sentences = [], []
# Set the model to eval mode...
model.eval()
with torch.no_grad():
    with open(args.BLEU_score_filepath, "w") as ff:
        raw_en_iter = iter(io.open(f"./data/ted-talks-corpus/{args.corpus_type}.en", encoding="utf8"))
        raw_fr_iter = iter(io.open(f"./data/ted-talks-corpus/{args.corpus_type}.fr", encoding="utf8"))
        for (raw_en, raw_fr) in tqdm(zip(raw_en_iter, raw_fr_iter)):
            en_item = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long)
            fr_item = torch.tensor([fr_vocab[token] for token in fr_tokenizer(raw_fr)], dtype=torch.long)
            if args.model_type == "MT2":
                src = torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor(
                    [EOS_IDX])], dim=0).to(device)#.transpose(0, 1)
                trg = torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor(
                    [EOS_IDX])], dim=0).to(device)  # .transpose(0, 1)
                # print(f"src shape = {src.shape}, trg shape = {trg.shape}")
                output = model(src.unsqueeze(dim=0), trg.unsqueeze(dim=0), 0)# For Batch First
                output = output.squeeze(dim=0)
            elif args.model_type == "MT1":
                src = torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0).to(device)
                trg = torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0).to(device)
                output = model(src.unsqueeze(dim=1), trg.unsqueeze(dim=1), 0)# For Batch Second
                output = output.squeeze(dim=1)
            # Prediction in terms of token index
            pred = torch.argmax(output, dim=1)
            pred_indexes = [int(ndx.cpu()) for ndx in pred[1:]]# Omitting start Token...
            real_indexes = [int(ndx.cpu()) for ndx in trg[1:]]
            pred_st = " ".join(fr_vocab.lookup_tokens(pred_indexes))
            real_st = " ".join(fr_vocab.lookup_tokens(real_indexes[1:-2]))
            score = bleu_st.sentence_score(real_st, [pred_st]).score
            pred_st = pred_st.replace("\n", "").strip()
            ff.write(pred_st + f"\t{score}\n")
            translated_sentences.append([pred_st])
            real_sentences.append(real_st)

# Compute Corpus Level BLEU scores...
corpus_bleu_score = bleu_corp.corpus_score(real_sentences, translated_sentences).score
with open(args.BLEU_score_filepath, 'r+') as fll:
    content = fll.read()
    fll.seek(0, 0)
    fll.write("corpus_BLEU:\t{0:5f}\n".format(corpus_bleu_score) + content)
