#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 16:43:31 2022

@author: rodo
"""
# RegEx package to perform tokenization
import re
import argparse


# Tokenization Class.
class Tokenizer:
    def __init__(self):
        pass
        # super().__init__(**kwargs)
    # txt: It is a sentence in the given corpus.
    # 1) ---------------------- Replace Hashtags ---------------
    def replaceHashtags(self, txt):
        return re.sub('\#[a-zA-Z]\w+', '<HASHTAG>', txt)
    # 2.0) ---------------------- Replace EMAILs ---------------

    def replaceEMAIL(self, txt):
        # (?<= [\w\.\/\:\~\-])*([\w\~\-] +\@[\w\~\-]+)(?=[\w\.\/\:\~\-]*)
        return re.sub(r'\S*[\w\~\-]\@[\w\~\-]\S*', r'<EMAIL>', txt)
    # 2.5) ---------------------- Replace URLs ---------------

    def replaceURL(self, txt):
        return re.sub(r'(https?:\/\/|www\.)?\S+[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\S+', r'<URL>', txt)
    # 3) ---------------------- Replace Mentions ---------------

    def replaceMentions(self, txt):
        return re.sub('@\w+', '<MENTION>', txt)

    # 4) ---------------------- Change UPPER to Lower ---------------
    def upperToLower(self, txt): return txt.lower()

    # 5) ---------------------- Remove Footnotes ---------------
    def removeFootNotes(self, txt): return re.sub(
        '\[.*\s+-\s+.*\]', '<FOOTNOTE>', txt)

    # 6) ---------------------- Replace Multiple Punctuations with Single ------
    def replacePunctuation(self, txt): return re.sub(
        r'(!|"|\#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|‘|\{|\||\}|~)\1{1,}', r'\1', txt)
    # curly bracket ==> {atleast count, atmost count}
    # \1 ==> Group or the whole sentence.
    # '(!+|"+|\#+|\$+|%+|&+|'+|\(+|\)+|\*+|\++|,+|-+|\.+|\/+|:+|;+|<+|=+|>+|\?+|@+|\[+|\\+|\]+|\^+|_+|‘+|\{+|\|+|\}+|~+)'
    
    # 7) ---------------------- Handle Date-Time ---------------
    def replaceDateTime(self, txt):
        txt = re.sub(
            r'\d{2,4}\-\d\d-\d{2,4}|\d{2,4}\/\d\d\/\d{2,4}|\d{2,4}:\d\d:?\d{2,4}', '<DATE>', txt)
        return re.sub(r'\d+:\d\d:?\d{0,2}?( am|am| pm|pm)', r'<TIME>', txt)
        # can be improved more like 10th May, may or 10thMay, 10thmay

    # 7) ---------------------- Handle MObile Number ---------------
    def replaceMobileNumber(self, txt):
        # https://blog.insycle.com/phone-number-formatting-crm
        return re.sub(r'[\+0-9\-\(\)\.]{3,}[\-\.]?[0-9\-\.]{3,}', r'<MOB>', txt)

    # 7) ---------------------- Handle other random floats and integers (isolated) ---------------
    def replaceNumericals(self, txt):
        return re.sub(r'(?<=\s)[\:\.]?\d*[\:\.]?\d*[\:\.]?(?=\s)', r'<NUM>', txt)

    # 8) ---------------------- Replace Multiple alphabets with Double ---------------
    def replaceAlphabets(self, txt): return re.sub(r'(\w)\1{2,}', r'\1\1', txt)
    # \w: AlphaNumeric digits.

    # 9) ---------------------- Replace Apostrophe Words ------
    def replaceApostrophe(self, txt_line):
        # Replacing n't with not
        t1 = re.sub(r'can\'t', r'can not', txt_line)
        t1 = re.sub(r'won\'t', r'will not', txt_line)
        t1 = re.sub(r'([a-zA-Z]+)n\'t', r'\1 not', txt_line)
        # Replacing That's, he's, she's etc. with <root> is.
        t1 = re.sub(r'([a-zA-Z]+)\'s', r'\1 is', t1)
        # Replacing i'm.
        t1 = re.sub(r'([iI])\'m', r'\1 am', t1)
        # Replacing we've, i've.
        t1 = re.sub(r'([a-zA-Z]+)\'ve', r'\1 have', t1)
        # Replacing i'd, they'd.
        t1 = re.sub(r'([a-zA-Z]+)\'d', r'\1 had', t1)
        # Replacing i'll, they'll.
        t1 = re.sub(r'([a-zA-Z]+)\'ll', r'\1 will', t1)
        # Replacing we're, they're.
        t1 = re.sub(r'([a-zA-Z]+)\'re', r'\1 are', t1)
        # Replacing tryin', doin'.
        t1 = re.sub(r'([a-zA-Z]+)in\'', r'\1ing', t1)
        return t1

    # 10) ---------------------- Removing Special Chars Embedded in Words ------
    def replaceSpecialCharsFromWords(self, txt, flag=False):
        # handle Hyphen words, e.g., short-sightedness to short sightedness
        t1 = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', txt)
        # Again scan for numericals like 29-year-old.
        t1 = self.replaceNumericals(t1)
        # Put spaces around different punctuations.
        t1 = re.sub(
            r'([\*\-\#\%\!\"\$\&\'\(\)\+\,\-\.\/\:\;\=\?\@\[\\\]\^\_\‘\{\|\}\~])', r' \1 ', t1)
        # Remove punctuations as they don't put any value in N-gram.
        if flag:
            t1 = re.sub(
                r'(?<=[a-z0-9\s\*\-\#\%\!\"\$\&\'\(\)\+\,\-\.\/\:\;\=\?\@\[\\\]\^\_\‘\{\|\}\~\<\>])[\*\-\#\%\!\"\$\&\'\(\)\+\,\-\.\/\:\;\=\?\@\[\\\]\^\_\‘\{\|\}\~\<\>](?=[a-z0-9\s\*\-\#\%\!\"\$\&\'\(\)\+\,\-\.\/\:\;\=\?\@\[\\\]\^\_\‘\{\|\}\~\<\>])', r'', t1)
        # Also remove spaces in between words or symbols(punctuations) ocurring twice.
        return re.sub(r'\s{2,}', r' ', t1.strip())

    # 10) ---------------------- Add Sentence Beginning and Ending Tag ------
    def addSOSEOS(self, txt):
        # Add Beginning TAG --> Lookahead after any single char at starting.
        # t1 = re.sub(r'(^.)(?=.*)', r'<SOS> \1', txt)
        # # Add Ending TAG --> Lookbehind
        # return re.sub(r'(.$)', r'\1 <EOS>\n', t1)
        return "<SOS> " + txt.strip() + " <EOS>\n"
    # 11) ---------------------- End of Class Tokenizer ---------------

# ------------------------- Write these Cleaned Text into a doc -----------


def saveText(texts, filename):
    """
    Parameters
    ----------
    texts : STRING
        Preprocessed Text (after applying all pre-processing functions).
    filename: STRING
        Name of Text file
    Returns
    -------
    None (Save it in a file).
    """
    with open(filename+".txt", "w") as fp:
        for ele in texts:
            fp.write(ele)
# ---------------------- Split Each Line according to spaces -------


def vocabBuilder(texts, filename):
    """
    Parameters
    ----------
    filename : STRING
        name for the Vocabulary Text File.

    Returns
    -------
    None.
    """
    vocab = set()
    for txt in texts:
        vocab.update([re.sub(r'\s', '', ele) for ele in re.split(" ", txt)])
    with open(filename+".txt", "w") as f:
        for elem in sorted(vocab):
            f.write(elem + "\n")
    return list(vocab)


if __name__ == "__main__":
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    # Corpus path
    corpus_path = args.path  # "./corpora/general-tweets.txt"

    # Load the dataset.
    with open(corpus_path, 'r') as fp:
        texts = fp.readlines()
        # readline() returns single line while readlines() return multiple.
    # Create a Tokenizer Object.
    token = Tokenizer()
    # Create an Empty List
    preprocessedTexts = list()
    for txt in texts:
        # Apply all sorts of pre-processing on imported texts (basically a sentence).
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
        txt1 = token.replaceSpecialCharsFromWords(txt1, flag=False)
        txt1 = token.addSOSEOS(txt1)
        preprocessedTexts.append(txt1)
    name_of_file = input("Enter name of file: ")
    saveText(preprocessedTexts, "clean_corpora/2021701010_"+name_of_file)
    _ = vocabBuilder(preprocessedTexts, "vocab/2021701010_"+name_of_file+"Vocabulary")


# ROUGH

# stri = "tyu***dsdvbbv wdhvckjbevk--dfvv-fvfvfv-vn *dfvefvev5hgyfg*efvjekjv iop#qwe kfvbhedfv kejbdvksefv sekjfvbkshedfv"
# re.sub(r'(?<=\S)([\*\-#]+)(?=\S?)', r'', stri)
# match = re.findall(r'(?P<g1>\w+)n\'t|(?P<g2>\w+)\'ll|(?P<g3>\w+)\'m|(?P<g4>\w+)\'d',
#         "don't and he'll have has i'm she'd")
# match.group()
# (!"\#\$\%&\'\(\)\*\+,-\.\/\:;\<\=\>\?@\[\\\]\^_‘\{\|\}~)
# (\<g2> will)|(\<g3> am)|(\<g4> had)
# |?(g2)\g<g2> will|?(g3)\g<g3> am|?(g4)\g<g4> had|
# (?(g1)(?P=g1) not)|(?(g2)(?P=g2) will)|(?(g3)(?P=g3) am)|(?(g4)(?P=g4) had)
