# Hypernym discovery

Hypernymy, namely “is-a” relation, is a vital lexical-semantic relation in natural languages, which relates general terms to their instances or subtypes. In this relation, we name a specific instance or subtype hyponym and its related general term hypernym. For instance, (apple, fruit) is in hypernymy relation, where apple is a hyponym and fruit is one of its hypernyms. Or, for the input word “dog”, some valid hypernyms would be “canine”, “mammal” or “animal”. (See
<http://aclweb.org/anthology/S18-1116>).

Here our __objective__ would be as follows:

Given an input term (or a *hyponym* __q__), our alogrithm need to retrieves a ranked list (possibly of length 15) of its suitable hypernyms (__h__) from a large corpus.

The data of [SemEval-2018 Task 9](https://competitions.codalab.org/competitions/17119) is used for training and testing.

## Directory Structure
This submission has the following directory structure-
```
.
├── 1_1_SkipGram_EmbeddingLearning
│   ├── generated_embeddings (Contains the generated embedding files)
│   │   ├── english_sg_emb.txt
│   │   ├── medical_sg_emb.txt
│   │   └── music_sg_emb.txt
│   ├── plots (Contans 3 files which reflect epoch-vs-loss trends)
│   │   ├── english_sg_emb_loss.png
│   │   ├── medical_sg_emb_loss.png
│   │   └── music_sg_emb_loss.png
│   ├── create_vocab.py (Merges all the words from separate files within the given data)
│   ├── model.py (The skip-gram model for creating the word embeddings)
│   ├── dataset.py (Dataset class to load data for training embeddings using hyponym-hypernym pairs)
│   ├── trainer.py (Script to trigger the embedding training)
│   ├── visualize_loss_logs.py (Script to generate plots for logs)
│   └── NNLM_trainer.py (Script to train the NNLM model)
├── 1_2_Baseline_Implementation
│   ├── results (Contans 3 files which reflect the metric scores for different corpus)
│   │   ├── baseline_english_metricResults.txt
│   │   ├── baseline_medical_metricResults.txt
│   │   └── baseline_music_metricResults.txt
│   └── Hypernym_discovery_baseline_implementation.py (Primary training file for baseline implementation)
├── 2_Exploiting_Embedding_Space
├── 3_Exploiting_Hypernym_Hierarchy
├── 4_CRIM
│   ├── ckpts
│   ├── dir-data
│   ├── embeddings
│   ├── plots
│   ├── predictions
│   ├── corpus_ada2gnode.sh
│   ├── Evaluator.py
│   ├── get_corpus.sh
│   ├── get_data.sh
│   ├── hparams.conf
│   ├── install_word2vec.sh
│   ├── link_preprocessed_corpus_vocab.sh
│   ├── new_line_omit.py
│   ├── plotter.py
│   ├── predict.py
│   ├── prep_corpus.py
│   ├── prep_data.py
│   ├── Projector.py
│   ├── README.md
│   ├── task9_scorer.py
│   ├── train.py
│   └── utils.py
└── README.md
```


## Approaches:

1. [Baseline](./1_2_Baseline_Implementation/) approach that uses pre-trained embeddings along with recurrent module to perform hypernymic discovery. For more info refer to the [README](./1_2_Baseline_Implementation/Readme.md) file.
    1. This approach includes [Skipgram](./1_1_SkipGram_EmbeddingLearning/) implementation to learn the embeddings.
2. [TaxoEmbed](./2_Exploiting_Embedding_Space/)
3. [FCA](./3_Exploiting_Hypernym_Hierarchy/)
4. [CRIM](./4_CRIM/): The SoTA model.

## Data and Model Check Points
Download the `data` and `model checkpoint` from this [link](https://drive.google.com/drive/folders/1-FSxL97FMJx6l7D4JlvAGBWL8pkc_Gmx?usp=sharing) and try to put them in above specified directory structure.
