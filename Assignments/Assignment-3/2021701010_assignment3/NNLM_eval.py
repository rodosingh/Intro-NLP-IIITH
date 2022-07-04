import torch
from nltk.util import ngrams
from tqdm import tqdm
# import torch.nn as nn
# import torch.optim as optim
from tqdm import tqdm
from NNLM_Model import NNLM
from torchtext.data.utils import get_tokenizer
import pickle
import io
import argparse

# ===========================================================================================================================

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--filepath', type=str, required=True)
parser.add_argument('--PP_filename', type=str, required=True)
parser.add_argument('--n', type=int, required=True)
# python NNLM_eval.py --lang ENGLISH --model NNLM_17 --filepath ./data/europarl-corpus/train.europarl
# --PP_filename ./2021701010_LM_train.txt --n 5
# Parse the argument
args = parser.parse_args()

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
raw_lang_iter = iter(io.open(args.filepath, encoding="utf8"))
perplexity_sent = []
#Set the model to eval mode
NNLM_model.eval()
with torch.no_grad():
    with open(args.PP_filename, "w") as f:
        for raw_sent in tqdm(raw_lang_iter):
            probability_of_ngram = 1
            # ADDing HERE STARTING AND ENDING TAGS...
            token_lst = tokenizer(raw_sent)
            if len(token_lst) >= args.n:
                tokens_lst = [BOS_IDX]+[vocab[token] for token in token_lst]  # +[EOS_IDX] [BOS_IDX]*abs(n-2)+
            else:
                continue
            grams_lst = list(ngrams(tokens_lst, args.n))
            for gram in grams_lst:
                input_gram = torch.tensor(gram[:-1], dtype=torch.long).to(device)
                target_token = gram[-1] #as only index is required
                #torch.tensor(, dtype=torch.long)
                # Predict
                output, _ = NNLM_model(input_gram.unsqueeze(dim=0))
                output = torch.exp(output.view(-1))# as log_softmax output
                probability_of_ngram *= output[target_token].cpu().numpy()
            # Perplexity
            pp_score = (1/probability_of_ngram)**(1/len(grams_lst))
            f.write(raw_sent.strip()+f"    {pp_score:.5f}\n")
            # To avoid inf PP score.
            if pp_score == float('inf'):
                perplexity_sent.append(perplexity_sent[-1])
            else:
                perplexity_sent.append(pp_score)
with open(args.PP_filename, 'r+') as fll:
    content = fll.read()
    fll.seek(0, 0)
    fll.write("Average Perplexity Score: {0:5f}\n".format(sum(perplexity_sent)/len(perplexity_sent)) + content)
