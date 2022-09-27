#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 13:06:12 2022

@author: rodo
"""
import pickle
import numpy as np
import tokenizer_new as tk
import argparse

# ---------------------- HELPER FUNCTIONS -------------------------------------

# ---------------------- Create N-Grams ------------------
def sent2ngrams(sent, n):
    """
    Split the sentence into n-grams
    """
    # check if sentence is at all full (or not empty)
    tok = sent.split(' ')
    if len(tok) > 0:
        # Empty list to store n-grams
        ng = []
        # Note that for these larger n-grams, we’ll need
        # to assume extra contexts to the left and right of the sentence end.
        # [<SOS> <SOS> <SOS> i hate this <EOS>] for 4-grams
        tokens = ["<SOS>" for _ in range(n-2)]+tok
        # At least n+1 tokens should be there --> n-1 for <SOS>
        # +1 for <EOS> and >=1 for middle context => >(n-1+1)
        for k in range(n, len(tokens)):
            # n-gram sentence
            ng.append(" ".join(tokens[k-n:k]))
        return ng
    return None
    
def ngrams_constructor(n, texts):
    """
    Parameters
    ----------
    n : INT
        length of words to be considered to predict the next.
    texts : List of Strings
        List containing Sentences.

    Returns
    -------
    Frequency Table (a dictionary) for n-grams or tokens.
    """
    ngrams = dict()
    for sent in texts:
        # Create list of n-grams with above function.
        lst_ngrams = sent2ngrams(sent, n)
        if lst_ngrams is not None:
            for s in lst_ngrams:
                if s not in ngrams:
                    ngrams[s] = 1
                else:
                    ngrams[s] += 1
    return ngrams


def recursiveNgramsConstructor(n, texts, unk_thresh=10):
    """
    Parameters:
        n: Highest Order n-gram required
        texts : List of Strings
            List containing Sentences.
        unk_thresh: Threshold count for a
             word to be in <UNK> group.

    Output:
        Dictionary Containing Tables of all n-grams from 1 to n.
    """
    req_dict = {k:ngrams_constructor(k, texts) for k in range(2, n+1)}
    unigrams = ngrams_constructor(1, texts)
    # here those words/unigrams whose count is less than certain threshold
    # put them in <UNK> group.
    unigrams["<UNK>"] = 0
    for key, val in unigrams.items():
        if val <= unk_thresh:
            unigrams["<UNK>"] += val
            _ = unigrams.pop(key)
    req_dict[1] = unigrams
    return req_dict

# ---------------------- LM = Construct Language Model ------------------------

def Cnt(string, cent_dict):
    """

    Parameters
    ----------
    string : string
        a string of word(s).

    Returns
    -------
    count of that string in corpus
    """
    # count no. of words in the given string.
    if len(string) == 0:
        return 0
    nwords = len(string.split(" "))
    # Retieve the count of that string from a central dict.
    try:
        return cent_dict[nwords][string]
    except KeyError:
        # When string doesn't exist in any table then returns 0 or <UNK>
        return 0

def sum_of_counts(history, cent_dict):
    """
    Parameters
    ----------
    history : string
        the past string of word(s).

    Returns
    -------
    cnt : integer
        sum of counts of history + variable_word.

    """
    # initial val
    cnt = 0
    # which n for ngram
    n = len(history.split(" ")) + 1
    # sum all possible counts of 'current' word with given 'history'
    for key, val in cent_dict[n].items():
        req_str = " ".join(key.split(" ")[:-1])
        if req_str == history:
            cnt += val
    return cnt

def cont_count(string, n, cent_dict):
    """
    Parameters
    ----------
    string : string
        the current word whose past context types we need to estimate.
    n : integer
        n stand for ngrams i.e., no. of words.

    Returns
    -------
    cnt : counts
        Total count of strings whose end word is 'string'.
    """
    cnt = 0
    for key in cent_dict[n].keys():
        if key.split(" ")[-1] == string:
            cnt += 1
    return cnt

def count_of_positives(history, cent_dict):
    """
    Parameters
    ----------
    history : string
        the past string of word(s).

    Returns
    -------
    cnt : integer
        No. of history + variable_word having counts > 0.
    """
    # initial val
    cnt = 0
    # which n for ngram
    n = len(history.split(" ")) + 1
    # sum all possible counts of 'current' word with given 'history'
    for key, val in cent_dict[n].items():
        if val > 0:
            req_str = " ".join(key.split(" ")[:-1])
            if req_str == history:
                cnt += 1
    return cnt

def kneserNey(history, current, recur_step, cent_dict):
    """
    Parameters
    ----------
    history : Tuple
        Tuple consisting of past words.
    current : String
        The word whose likelihood is to be estimated.

    Returns
    -------
    probability of happening of current word given the past.
    """
    # n ==> for n-grams
    n = len(history.split())+1
    # Check if current word is in VOCAB
    if current not in cent_dict[1]:
        # return len(list(filter(lambda x: cent_dict[n][x] == 1, cent_dict[n])))/len(cent_dict[n])
        return 0.75/sum(cent_dict[1]["<UNK>"])
    # base condition --- Empty String
    if n == 1:
        return 0.25/len(cent_dict[1]) + 0.75/sum(cent_dict[1]["<UNK>"])
            # 0.75*len(dict(filter(lambda item: item[1]>0, 
            # cent_dict[1].items())))/(tot_unigram_counts*len(cent_dict[1]))
            # As V cancels out with the NUMerator.
    # c_KN = count or continuation_count
    if recur_step == 1:
        try:
            first_term = max(Cnt(" ".join([history, current]), cent_dict)-0.75, 0)/sum_of_counts(history, cent_dict)
        except ZeroDivisionError:
            # Handle 0/0 form
            first_term = 0
    else:
        try:
            first_term = max(cont_count(current, n, cent_dict) - 0.75, 0)/len(cent_dict[n])
        except ZeroDivisionError:
            first_term = 0
    # Define Lambda
    try:
        lamb = (0.75/sum_of_counts(history, cent_dict))*count_of_positives(history, cent_dict)
        
    except ZeroDivisionError:
        # Handle 0/0 error.
        # no point of doing recursion further. Plus, if lamb=0 then first term
        # is also zero as they have same denom and lamb can be only when denom=0.
        # And since lamb have same def for all recursion steps, then it wont't get
        # 0 at lower order, because higher order n-grams are less frequent than lower
        # orders. If a particular lower n-gram doesn't exist then higher too doesn't 
        # exist. So at first step only lamb=0 ==> first_term=0 ==> lamb(eps)/V 
        return 0.75/sum(cent_dict[1]["<UNK>"])
    # New history for further step...
    new_hist = " ".join(history.split()[1:])
    sec_term = lamb*kneserNey(new_hist, current, recur_step+1, cent_dict)
    # NOw combine all the terms to get final term.
    return first_term + sec_term
    
# Witten-Bell discounting.
def wittenBell(history, current, cent_dict):
    """
    Parameters
    ----------
    history : Tuple
        Tuple consisting of past words.
    current : String
        The word whose likelihood is to be estimated.

    Returns
    -------
    probability of happening of current word given the past.
    """
    # Define BASE condition for this recursive function.
    n = len(history.split()) + 1
    # base condition --- Empty String
    if n == 1:
        if current in cent_dict[1]:
            return Cnt(current, cent_dict)/sum(cent_dict[1]["<UNK>"])
        return 1/len(cent_dict[1])

    # define lambda parameter
    try:
        lamb = count_of_positives(history, cent_dict)/(count_of_positives(history, cent_dict) + sum_of_counts(history, cent_dict))
    except ZeroDivisionError:
        # again lambda consists of term in denom which is similar to denom of pML
        # Due to which lamb=ZeroDivisionError iff pML=ZeroDivisionError
        return 1/len(cent_dict[n])

    # The first term in WITTEN-Bell expression and it's exception is handled above.
    pML = Cnt(" ".join([history, current]), cent_dict)/sum_of_counts(history, cent_dict)
    
    # Now arraning all above expressions.
    new_hist = " ".join(history.split()[1:])
    return (1 - lamb)*pML + lamb*wittenBell(new_hist, current, cent_dict)

def perplexityScore(prob_list):
    """
    Input:
        list containing n-gram probability scores
    """
    return np.power(1/np.prod(prob_list), 1/len(prob_list))

def sent2PPScore(sent, n, smoothing, cent_dict, verbosity=0):
    """
    1. Split sentence into n-grams sentence.
    2. Estimate probability of each n-gram.
    3. Compute Perplexity scores of Whole sentence using ALL sentences.
    Input:
        sent: full sentence string.
        n: no. of grams
        smoothing: Type of smoothing
    Output:
        perplexity score of the given n-gram string.
    """
    # Construct n-grams...
    ngrams_lst = sent2ngrams(sent, n); scores = []
    if smoothing == "k":
        for ng in ngrams_lst:
            lst = ng.split(" ")
            hist, current = " ".join(lst[:-1]), lst[-1]
            scores.append(kneserNey(hist, current, 1, cent_dict))
            if verbosity:
                print("PP Score with KNESER-NEY for sentence: {0} = {1:.3f}".format(ng, scores[-1]))
    elif smoothing == "w":
        for ng in ngrams_lst:
            lst = ng.split(" ")
            hist, current = " ".join(lst[:-1]), lst[-1]
            scores.append(wittenBell(hist, current, cent_dict))
            if verbosity:
                print("PP Score with WITTEN-BELL for sentence: {0} = {1:.3f}".format(ng, scores[-1]))
    else:
        raise ValueError("Please enter a correct smoothing format.")
    # calculate PP scores.
    return perplexityScore(scores)


# ======================================= DRIVER CODE ================================================
if __name__ == '__main__':

    # ----------------------- Parse the Command Line Arguments -------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('n_value', type=int)
    parser.add_argument('smoothing', type=str)
    parser.add_argument('path', type=str)

    # Retrieve Arguments
    n = parser.parse_args().n_value
    corpus_path = parser.parse_args().path
    smoothing = parser.parse_args().smoothing

    # ---------------------- Clean Corpora; Build Vocab ---------------------------
    # Try to load the clean and preprocessed COPRUS created before, else Create one.
    try:
        with open("clean_corpora/"+corpus_path.split("/")[-1][:-4]+".txt", "r") as fp:
            preprocessedTexts = fp.readlines()
    except FileNotFoundError:
        # Load the dataset.
        with open(corpus_path, 'r') as fp:
            texts = fp.readlines()
            # readline() returns single line while readlines() return multiple.
        # Create a Tokenizer Object.
        token = tk.Tokenizer()
        # Create an Empty List
        preprocessedTexts = list()
        for txt in texts:
            # Apply all sorts of pre-processing on imported texts.
            txt1 = token.upperToLower(txt)
            txt1 = token.replaceHashtags(txt1)
            txt1 = token.replaceEMAIL(txt1)
            txt1 = token.replaceURL(txt1)
            txt1 = token.replaceMentions(txt1)
            txt1 = token.removeFootNotes(txt1)
            txt1 = token.replacePunctuation(txt1)
            txt1 = token.replaceDateTime(txt1)
            txt1 = token.replaceMobileNumber(txt1)
            txt1 = token.replaceNumericals(txt1)
            txt1 = token.replaceAlphabets(txt1)
            txt1 = token.replaceApostrophe(txt1)
            txt1 = token.replaceSpecialCharsFromWords(txt1, flag=True)
            txt1 = token.addSOSEOS(txt1)
            preprocessedTexts.append(txt1)
        tk.saveText(preprocessedTexts, "clean_corpora/"+corpus_path.split("/")[-1][:-4])
        _ = tk.vocabBuilder(preprocessedTexts, "vocab/"+corpus_path.split("/")[-1][:-4]+"Vocabulary")
    # ---------------------------- TRAIN and TEST split -----------------------------------------
    np.random.seed(23)
    idx = np.random.choice(len(preprocessedTexts), 1000, replace=False)
    train, test = [], []
    for id in range(len(preprocessedTexts)):
        if id in idx:
            test.append(preprocessedTexts[id])
        else:
            train.append(preprocessedTexts[id])

    # ------------ Create N-grams recursively and store their Frequency table for TRAINING corpus.
    try:
        # Try to read existing Stored Dictionary.
        with open("frequency-table/"+corpus_path.split("/")[-1][:-4]+"CentralDictTrain.pickle", 'rb') as handle:
            cent_dict = pickle.load(handle)
    except FileNotFoundError:
        # when File is not there--> create it
        cent_dict = recursiveNgramsConstructor(n=n, texts=train)
        with open("frequency-table/"+corpus_path.split("/")[-1][:-4]+"CentralDictTrain.pickle", "wb") as handle:
            pickle.dump(cent_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Add Frequency for an unknown word: <UNK> from the count of Words seen once.
    # cent_dict[1]['<UNK>'] = len(list(filter(lambda x: cent_dict[1][x] == 1, cent_dict[1])))

    # ------------------------ Apply Smoothing; k=Kneser-Ney; w=WittenBell -----------------------------
    # Steps:
    #     1) Split the Sentence into 4-gram Sentences.
    #     2) Compute probability of having that particular 4th word given past 3 words in 4-gram sentences.
    #     3) Take PERPLEXITY scores of probabilities computed for each sentences to know how likely is that sentence.
    #     4) Take Average PERPLEXITY scores for both TRAIN and TEST to know how likely they are.
    
    # FOR TRAINING LET'S COMPUTE Probability for each sentence and then Perplexity scores.
    pp_score_lst_train, pp_score_lst_test = [], []
    # Open a Document and Save the scores calculated.
    LM = input("Enter the name for score TXT file (for which LM): ")

    # ----------------------------------------------- TRAIN ----------------------------------------------
    print("="*20+"COMPUTATION STARTED FOR TRAINING SET"+"="*20)
    with open("scores/2021701010_"+LM+"_train-perplexity.txt", "w") as flm:
        for j, sent in enumerate(train):
            # Omit the newline character
            sent = sent.strip()
            # If by any chance sentence is of length 0.
            if len(sent) != 0:
                pp_score_lst_train.append(sent2PPScore(sent, n, smoothing, cent_dict=cent_dict))
                flm.write(sent+"    PP Score = {0:.3f}\n".format(pp_score_lst_train[-1]))
                print(f"Sentence {j+1} is PROCESSED!!!")
                if j == 1000:
                    break
    with open("scores/2021701010_"+LM+"_train-perplexity.txt", 'r+') as fll:
        content = fll.read()
        fll.seek(0, 0)
        fll.write("Average Perplexity Score: {0:.3f}\n".format(np.mean(pp_score_lst_train)) + content)
    print("="*50+"DONE!!!"+"="*50)

    # ----------------------------------------------- TEST ----------------------------------------------
    print("="*20+"COMPUTATION STARTED FOR TESTING SET"+"="*20)
    with open("scores/2021701010_"+LM+"_test-perplexity.txt", "w") as flm:
        for l, sent_ in enumerate(test):
            if l == 250:
                break
            # Omit the newline character
            sent_ = sent_.strip()
            # if by any chance sentence is empty
            if len(sent_) != 0:
                pp_score_lst_test.append(sent2PPScore(sent_, n, smoothing, cent_dict=cent_dict))
                flm.write(sent_+"    PP Score = {0:.3f}\n".format(pp_score_lst_test[-1]))
                print(f"Sentence {l+1} is PROCESSED!!!")
    with open("scores/2021701010_"+LM+"_test-perplexity.txt", 'r+') as fll:
        content = fll.read()
        fll.seek(0, 0)
        fll.write("Average Perplexity Score: {0:.3f}\n".format(np.mean(pp_score_lst_test)) + content)
    print("="*50+"DONE!!!"+"="*50)
