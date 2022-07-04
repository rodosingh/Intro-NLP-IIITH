"""language_model.py: Demo Script that given a statement tries to
output its probability with pre-trained model."""

__author__ = "Aditya Kumar Singh"

import torch
from NNLM_Model import NNLM
from torchtext.data.utils import get_tokenizer
from nltk.util import ngrams
import pickle
import io
import argparse

# ===========================================================================================================================

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
# Parse the argument
args = parser.parse_args()
# how to run
# python language_model.py --lang ENGLISH --model NNLM_17
# ===========================================================================================================================

# Mention Device...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
_ = NNLM(16000, 512, 256, 3, 0.2)
NNLM_model = torch.load(f"./models/{args.model}.pth", map_location=device)

# Load Vocab and construct tokenizer
if args.lang == "ENGLISH":
    with open("./data/europarl-corpus/en_vocab.pkl", "rb") as fp:
        vocab = pickle.load(fp)
    # Construct tokenizer
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
elif args.lang == "FRENCH":
    with open("./data/news-crawl-corpus/fr_vocab.pkl", "rb") as fp:
        vocab = pickle.load(fp)
    # Construct tokenizer
    tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# ===========================================================================================================================

# Read file and convert sentence to tensor
BOS_IDX = vocab['<bos>']
input_sent = input(f"Enter the {args.lang} sentence (should be of length > 5):\n")
#Set the model to eval mode
NNLM_model.eval()
with torch.no_grad():
    probability_of_ngram = 1
    # ADDing HERE STARTING AND ENDING TAGS...
    token_lst = tokenizer(input_sent)
    if len(token_lst) >= 5:
        tokens_lst = [BOS_IDX]+[vocab[token] for token in token_lst]  # +[EOS_IDX] [BOS_IDX]*abs(n-2)+
    else:
        raise ValueError("Please enter a sentence of length greater than 5 :(")
    grams_lst = list(ngrams(tokens_lst, 5))
    for gram in grams_lst:
        input_gram = torch.tensor(gram[:-1], dtype=torch.long).to(device)
        target_token = gram[-1]  # as only index is required
        #torch.tensor(, dtype=torch.long)
        # Predict
        output, _ = NNLM_model(input_gram.unsqueeze(dim=0))
        output = torch.exp(output.view(-1))  # as log_softmax output
        probability_of_ngram *= output[target_token].cpu().numpy()
    # Perplexity
    pp_score = (1/probability_of_ngram)**(1/len(grams_lst))

# Print the probability and perplexity score...
print(f"\nProbability: {probability_of_ngram}")
print(f"Perplexity: {pp_score}")
