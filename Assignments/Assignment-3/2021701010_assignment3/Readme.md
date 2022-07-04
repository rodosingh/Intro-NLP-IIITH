# Brief overview over each file and how to use them

## Directory Structure
```bash
project
    |
    |___data
    |   |___europarl-corpus
    |   |___news-crawl-corpus
    |   |___ted-talks-corpus
    |
    |___models
    |___plots
    |___Readme.md
    |___vocabBuilder.py
    |___NNLM.py
    |___NNLM_Model.py
    |___NNLM_eval.py
    |___MT1.py
    |___seq2seq.py
    |___MT2.py
    |___seq2seq_MT2.py
    |___MT_eval.py
    |___language_model.py
    |___machine_translation.py
    |___2021701010_LM_train.txt
    |___2021701010_LM_test.txt
    |___2021701010_MT1_train.txt
    |___2021701010_MT1_test.txt
    |___2021701010_MT2_train.txt
    |___2021701010_MT2_test.txt
    |___2021701010_Report_Assign3.pdf
```

## Question 2: Neural Language Model

### Question 2.1: Training NNLM
To run the `NNLM.py` on English and French enter the following in terminal:

```python3
python $HOME/<path-to-your-folder>/NNLM.py --lang ENGLISH --model_number 17
```
The above code, with help of `NNLM_Model.py` try to train the model and output 3 following things:
1. Save the best checkpoint model under the name of `NNLM_<model_number>.pth` based on its validation loss in `models` folder.
2. A loss curve under the name `NNLM_<model_number>.png` and save it in `plots` folder.
3. A `NNLM_<model_number>.txt` file denoting what parameters are used to train the model.

__Note:__ Here `--lang` denotes which language model to train.

### Question 2.2: Saving Perplexity Scores...
```python3
python $HOME/<path-to-your-folder>/NNLM_eval.py --lang ENGLISH --model NNLM_17 --filepath ./data/europarl-corpus/train.europarl --PP_filename ./2021701010_LM_train.txt --n 5
```
Run `NNLM_eval.py` to generate the following `.txt` files with the above mentioned strategy (example).

1. 2021701010_LM_train.txt
2. 2021701010_LM_test.txt

## Question 3: Machine Translation

### Question 3.1a: MT1 - Creating `seq2seq` machine translation task (*English to French*) from scratch using `Encoder`-`Decoder` architecture.

Run `MT1.py` in following manner to generate the similar 3 things as dicussed above in [Question 2.1](#question-2.1:-training-nnlm)

```python3
python $HOME/<path-to-your-folder>/MT1.py --model_number 3
```

__Note__: It imports model from `seq2seq.py` file to build the final encoder-decoder architecture.

### Question 3.1b: MT2 - Creating `seq2seq` machine translation task (*English to French*) using pretrained `NNLM` models trained in [Question 2.1](#question-2.1:-training-nnlm) in from of `Encoder`-`Decoder`.

The same strategy is fiollowed here as in [Question 3.1a](#question-3.1a) where models are now imported from `seq2seq_MT2.py`

```python
python $HOME/<path-to-your-folder>/MT2.py --english_model NNLM_17 --french_model NNLM_18 --model_number 2
```

### Question 3.2: Computing BLEU scores.

```python3
python MT_eval.py --model_type MT1 --model MT1_3 --BLEU_score_filepath ./2021701010_MT1_train.txt --corpus_type train
```

Run the script `MT_eval.py` to generate the following `.txt` files with the above mentioned strategy (example).

1. 2021701010_MT1_train.txt
2. 2021701010_MT1_test.txt
3. 2021701010_MT2_train.txt
4. 2021701010_MT2_test.txt

## Other Files:
1. `vocabBuilder.py` tries to generate vocabulary given a corpus with the help of inbuilt functions from `torchtext` library.
2. `language_model.py`: On running this file, the expected output is a prompt, which asks for a sentence and provides the __probability__ of that sentence.
Run this using below line:
```python3
python language_model.py --lang <Language Type: ENGLISH or FRENCH> --model <model_name>
```
An example:
```python3
$ python3 language_model.py ../models/trained_en_lm.pth
input sentence: I am a man.
0.89972021
```
3. `machine_translation.py`: Runs the MT model given the following:
```python3
$ python machine_translation.py --model_type <MT1 or MT2> --model <model_name>
```
Input for this part will be the model path to either MT1 or MT2. On running the file, the
expected output is a prompt, which asks for a sentence and provides the translated sentence in target language. Therefore, an example would be:
```python3
$ python machine_translation.py ../models/trained_mt2.pth
input sentence: I am a man.
je suis un homme.
```

# Packages Used

```pyhton3
torch==1.10.2
torchtext==0.11.2
nltk==3.7
sacrebleu==2.0.0
spacy==3.2.4
matplotlib==3.5.1
pickle==4.0
tqdm==4.64.0
```

_"Hope you like the project, do give a star if it helps you in anyway"_
