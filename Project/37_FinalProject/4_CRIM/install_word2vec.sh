#!/bin/bash
# Download the word2vec model and compile it.. 
wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
unzip source-archive.zip
rm source-archive.zip
cd word2vec/trunk
make

# To Run Word2Vec
# word2vec/trunk/word2vec -train dir-data/preprocessed_corpus_and_vocab/medical.txt 
# -read-vocab dir-data/preprocessed_corpus_and_vocab/medical.txt.vocab
# -output embeddings/medical.txt -cbow 0 -negative 10 -size 300 -window 10 -sample 1e-5 
# -min-count 1 -iter 10 -threads 20 -binary 0 -hs 0

# word2vec/trunk/word2vec -train dir-data/preprocessed_corpus_and_vocab/medical.txt -read-vocab dir-data/preprocessed_corpus_and_vocab/medical.txt.vocab -output embeddings/medical_embed.txt -cbow 0 -negative 10 -size 300 -window 10 -sample 1e-5 -min-count 1 -iter 10 -threads 20 -binary 0 -hs 0