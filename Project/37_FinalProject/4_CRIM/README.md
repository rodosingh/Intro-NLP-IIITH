# CRIM (Hypernym discovery SoTA)

## Requirements

- Python 3 (tested using version 3.6.9)
- [PyTorch](https://pytorch.org/) (tested using version 1.2.0)
- [Pyhocon](https://github.com/chimpler/pyhocon)
- [Joblib](https://github.com/joblib/joblib)
- Lots of disk space (downloading, unzipping, and preprocessing the corpus requires around 40 GB for sub-task 1A)
- `Bash` (to get the data and corpora, and to install `word2vec`)
- `C` compiler if you install `word2vec`

## How to RUN?

Make directory where we can store a lot of data:

```bash
mkdir <dir-data>
```

Get training and evaluation data from the website of [SemEval-2018 Task 9](https://competitions.codalab.org/competitions/17119) (also copies scoring script in current directory):

```bash
./get_data.sh <dir-data>
```

Get corpus from the website of SemEval-2018 Task 9:

*Note: The above mentioned task has 5 subtasks, namely, 1A, 1B, 1C, 2A, and 2B. The subtasks starting with "1" are from language domain while the other two are from Medical and Music domain. As of now except Music and Medical (i.e., 2B and 2A), the corpus for language domain __aren't downloadable__. Since the main purpose of corpus given to us is to learn word embeddings for hyponyms and hypernyms, we for language subtasks, directly downloaded the pre-trained embeddings and used it for training our model to learn hypernymy relation.*

*Subtasks for which corpus is not available, download the embedding `.txt` file into a folder (e.g., "embedding" folder) and directly jump to data pre-processing stage mentioned below (`prep_data.py`)*

```bash
./get_corpus.sh <subtask> <dir-data>
cat * > umbc_webbase_corpus.txt
```
Now download and store pre-trained embeddings for subtask 1A (English), 1B (Italian), and 1C (Spanish)
*With Pre-trained embeddings we're not able to generate embeddings for all the words in given vocab. Almost 40-50% of words are tagged as <UNK> or unknown. Since __UMBC__ corpus (of 48G) is available with us, we preprocessed it and run `word2vec` on it generate the embeddings for all available in train, dev, and test vocab. While we'll not be performing any test no subtask 1B and 1C, as we don't have that particular corpus with us. So Let's see.*

```bash
./get_embeddings.sh <subtask> <embedding-directory>
```

Make preprocessed corpus and vocab (*only for subtask 2A and 2B*):

```bash
python prep_corpus.py <subtask> <path-to-corpus> <dir-datasets> <path-of-output-preprocessed-corpus>
```

Install `word2vec` in current directory (*provided by Google*):

```bash
./install_word2vec.sh
```

Train word embeddings on corpus using `word2vec`. Make sure to use the corpus and vocab that were produced by `prep_corpus.py` (not the raw corpus):

```bash
word2vec/trunk/word2vec -train <path-preprocessed-corpus> -read-vocab <path-preprocessed-corpus>.vocab -output <path-output> -cbow 0 -negative 10 -size 300 -window 10 -sample 1e-5 -min-count 1 -iter 10 -threads 20 -binary 0 -hs 0
```

Parameter Info:
| Key       | Description                                                                                                                                                                                                                                                              | Default Value                         |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| output    | embeddings/<subtask>.txt                                                                                                                                                                                                                                                 |                                       |
| size      | sets the size of word vectors                                                                                                                                                                                                                                            | 100                                   |
| window    | sets maximal skip length between words. In case of SkipGram, how many neighbouring context words to predict? The maximum distance between the current and predicted word within a sentence. E.g. `window` words on the left and `window` words on the left of our target | 5                                     |
| sample    | sets threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; useful range is (0, 1e-5)                                                                                                            | 1e-3                                  |
| hs        | 1 = use Hierarchical Softmax                                                                                                                                                                                                                                             | 0                                     |
| negative  | number of negative examples; common values are 3 - 10 (0 = not used)                                                                                                                                                                                                     | 5                                     |
| threads   | number of used threads                                                                                                                                                                                                                                                   | 12                                    |
| iter      | number of training iterations                                                                                                                                                                                                                                            | 5                                     |
| min-count | This will discard words that appear less than min-count times                                                                                                                                                                                                            | 5                                     |
| alpha     | sets the starting learning rate                                                                                                                                                                                                                                          | 0.025 for skip-gram and 0.05 for CBOW |
| classes   | output word classes rather than word vectors                                                                                                                                                                                                                             | 0 (vectors are written)               |
| debug     | sets debug mode                                                                                                                                                                                                                                                          | 2                                     |
| binary    | save the resulting vectors in binary mode                                                                                                                                                                                                                                | 0 (off)                               |
| saveVocab | the vocabulary will be saved to saveVocab value                                                                                                                                                                                                                          |                                       |
| readVocab | the vocabulary will be read from readVocab value , not constructed from the training data                                                                                                                                                                                |                                       |
| cbow      | use the continuous bag of words model                                                                                                                                                                                                                                    | 1 (use 0 for skip-gram model)         |
| silent    | sets whether any output should be printed to the console                                                                                                                                                                                                                 | false                                 |

Preprocess data and write in a pickle file:

```bash
python prep_data.py <subtask> <dir-datasets> <path-to-embeddings> <path-of-output-file>
```

Review hyperparameter settings in [hparams.conf](./hparams.conf).

Train model on training and dev data in pickle file, write a model and a log file in `dir-model`:

```bash
python train.py <path-to-data-pickle-file> <path-to-hparams.conf> <dir-model>
```
*Disclaimer: Don't create a `directory` to save model checkpoints. Just mention the path where you want to create the directory, it will be automatically created.*

Load trained model, make predictions on test queries:

```bash
python predict.py <path-to-pretrained-model> <path-to-data-pickle-file> <path-to-store-predictions-in-a-file>
```

Evaluate predictions on the test set using scoring script of SemEval-2018 Task 9:

```bash
python2.7 path/to/SemEval-Task9/task9-scorer.py path/to/SemEval-Task9/test/gold/<subtask>.<language>.test.gold.txt path/to/output/pred.txt
```

## Extras

Ensure that you create required  `soft links` mentioned in each folder for the large dataset, corpus, and embeddings (as an example try see [link_preprocessed_corpus.sh](./link_preprocessed_corpus_vocab.sh). Download the `data` and `model checkpoint` from this [link](https://drive.google.com/drive/folders/1-FSxL97FMJx6l7D4JlvAGBWL8pkc_Gmx?usp=sharing) and try to keep it in some large space.
