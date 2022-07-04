# importing libraries
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

abs_path = "../"
# loading data. Consider changing the path to the actual path where file is present
data = []
with open(abs_path+'reviews_Electronics_5.json') as file:
    for line in tqdm(file):
        data.append(''.join(ch if ch.isalnum() else ' ' for ch in json.loads(line)['reviewText'].lower()))
        
print("Total lines in Data: ", len(data))
print("First line in DATA:\n", data[0])

# Creating word frequency dict
words_freq = defaultdict(int)
for line in tqdm(data):
    line = line.split()
    for word in line:
        words_freq[word]+=1
        
# Data after removing less frequent words
data_2 = []
for sentence in tqdm(data):
    data_2.append(' '.join([word for word in sentence.split() if words_freq[word]>10]))
    
# Convert a collection of text documents to a matrix of token counts with CountVectorizer() function.
vectorizer = CountVectorizer()
vectorized_mat = vectorizer.fit_transform(data_2)
token_list = vectorizer.get_feature_names_out()

# Saving token list to a txt file
with open(abs_path+"models/tokens_list.txt", "w") as outfile:
    outfile.write("\n".join(token_list))
    
print("Count Matrix Shape: ", vectorized_mat.shape)

# Creating co-occurence matrix
co_occ_mat = vectorized_mat.T*vectorized_mat

# Occurence of same word one after other is almost never. hence set diagonal to zero.
co_occ_mat.setdiag(0)

print("Shape of Co-Occurence Matrix", co_occ_mat.shape)

# Saving co-occurence matrix to prevent crashing the ram
sparse.save_npz(abs_path+"models/co_occurence.npz", co_occ_mat)
