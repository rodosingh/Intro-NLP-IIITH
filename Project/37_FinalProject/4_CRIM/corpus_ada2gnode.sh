#!/bin/bash
mkdir -p /ssd_scratch/cvit/rodosingh/nlp/data/
# scp rodosingh@ada:/share3/rodosingh/nlp/data/2A_med_pubmed_tokenized.tar /ssd_scratch/cvit/rodosingh/nlp/data/.
# scp rodosingh@ada:/share3/rodosingh/nlp/data/2B_music_bioreviews_tokenized.tar /ssd_scratch/cvit/rodosingh/nlp/data/.
# tar -xvf /ssd_scratch/cvit/rodosingh/nlp/data/2A_med_pubmed_tokenized.tar --directory /ssd_scratch/cvit/rodosingh/nlp/data/
# # or -C to specify directory instead of --directory
# tar -xvf /ssd_scratch/cvit/rodosingh/nlp/data/2B_music_bioreviews_tokenized.tar --directory /ssd_scratch/cvit/rodosingh/nlp/data/
# # or -C
# unzip or gunzip package.zip -d /opt (-d to show directory)
# bunzip2 /ssd_scratch/cvit/rodosingh/nlp/data/italian_itwiki_20180420_300d.txt.bz2
# ln -s /ssd_scratch/cvit/rodosingh/nlp/data/2A_med_pubmed_tokenized.txt dir-data/2A_med_pubmed_tokenized.txt
# ln -s /ssd_scratch/cvit/rodosingh/nlp/data/2B_music_bioreviews_tokenized.txt dir-data/2B_music_bioreviews_tokenized.txt

# Bring the preprocessed corpus and Vocab
scp -r rodosingh@ada:/share3/rodosingh/nlp/data/preprocessed_corpus_and_vocab/english.txt /ssd_scratch/cvit/rodosingh/nlp/data/.
scp -r rodosingh@ada:/share3/rodosingh/nlp/data/preprocessed_corpus_and_vocab/english.txt.vocab /ssd_scratch/cvit/rodosingh/nlp/data/.
ln -s /ssd_scratch/cvit/rodosingh/nlp/data/preprocessed_corpus_and_vocab dir-data/preprocessed_corpus_and_vocab
#ln -s /ssd_scratch/cvit/rodosingh/nlp/data/ dir-data/ssd_scratch_data
