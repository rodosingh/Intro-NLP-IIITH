DATA_DIR = "../../data/" # "./DATA_SemEval2018-Task9/"
ENGLISH_HYPONYMS_TRAIN_FILE = "training/data/1A.english.training.data.txt"
MEDICAL_HYPONYMS_TRAIN_FILE = "training/data/2A.medical.training.data.txt"
MUSIC_HYPONYMS_TRAIN_FILE = "training/data/2B.music.training.data.txt"

ENGLISH_HYPONYMS_DEV_FILE = "trial/data/1A.english.trial.data.txt"
MEDICAL_HYPONYMS_DEV_FILE = "trial/data/2A.medical.trial.data.txt"
MUSIC_HYPONYMS_DEV_FILE = "trial/data/2B.music.trial.data.txt"

ENGLISH_HYPONYMS_TEST_FILE = "test/data/1A.english.test.data.txt"
MEDICAL_HYPONYMS_TEST_FILE = "test/data/2A.medical.test.data.txt"
MUSIC_HYPONYMS_TEST_FILE = "test/data/2B.music.test.data.txt"

ENGLISH_HYPERNYMS_TRAIN_FILE = "training/gold/1A.english.training.gold.txt"
MEDICAL_HYPERNYMS_TRAIN_FILE = "training/gold/2A.medical.training.gold.txt"
MUSIC_HYPERNYMS_TRAIN_FILE = "training/gold/2B.music.training.gold.txt"

ENGLISH_HYPERNYMS_DEV_FILE = "trial/gold/1A.english.trial.gold.txt"
MEDICAL_HYPERNYMS_DEV_FILE = "trial/gold/2A.medical.trial.gold.txt"
MUSIC_HYPERNYMS_DEV_FILE = "trial/gold/2B.music.trial.gold.txt"

ENGLISH_HYPERNYMS_TEST_FILE = "test/gold/1A.english.test.gold.txt"
MEDICAL_HYPERNYMS_TEST_FILE = "test/gold/2A.medical.test.gold.txt"
MUSIC_HYPERNYMS_TEST_FILE = "test/gold/2B.music.test.gold.txt"

ENGLISH_VOCAB_FILENAME = "vocabulary/english.txt.vocab"
MEDICAL_VOCAB_FILENAME = "vocabulary/medical.txt.vocab"
MUSIC_VOCAB_FILENAME = "vocabulary/music.txt.vocab"

ENGLISH_OUTFILENAME = "english_merged_vocab.txt"
MEDICAL_OUTFILENAME = "medical_merged_vocab.txt"
MUSIC_OUTFILENAME = "music_merged_vocab.txt"

TARGET_TRAIN_HYPO = MUSIC_HYPONYMS_TRAIN_FILE
TARGET_TRAIN_HYPER = MUSIC_HYPERNYMS_TRAIN_FILE
TARGET_DEV_HYPO = MUSIC_HYPONYMS_DEV_FILE
TARGET_DEV_HYPER = MUSIC_HYPERNYMS_DEV_FILE
TARGET_TEST_HYPO = MUSIC_HYPONYMS_TEST_FILE
TARGET_TEST_HYPER = MUSIC_HYPERNYMS_TEST_FILE
TARGET_VOCAB = MUSIC_VOCAB_FILENAME

OUT_FILENAME = MUSIC_OUTFILENAME

def read_file(DATA_DIR, FILENAME):
    f = open(DATA_DIR + FILENAME)
    lines = f.readlines()
    f.close()
    modified_lines = list()
    for line in lines:
        words = line.strip().split("\t")
        modified_line = ""
        for word in words:
            underscored_word = "_".join(word.split(" ")).lower()
            modified_line += underscored_word + "\t"
        modified_lines.append(modified_line)
    return modified_lines

def merge_hypernyms(hypernyms):
    all_hypernyms = list()
    for hypers in hypernyms:
        all_hypernyms += hypers
    return list(set(all_hypernyms))

hyponyms_train = [line.strip().split("\t")[0] for line in read_file(DATA_DIR, TARGET_TRAIN_HYPO)]
hypernyms_train = merge_hypernyms([line.strip().split("\t") for line in read_file(DATA_DIR, TARGET_TRAIN_HYPER)])
hyponyms_dev = [line.strip().split("\t")[0] for line in read_file(DATA_DIR, TARGET_DEV_HYPO)]
hypernyms_dev = merge_hypernyms([line.strip().split("\t") for line in read_file(DATA_DIR, TARGET_DEV_HYPER)])
hyponyms_test = [line.strip().split("\t")[0] for line in read_file(DATA_DIR, TARGET_TEST_HYPO)]
hypernyms_test = merge_hypernyms([line.strip().split("\t") for line in read_file(DATA_DIR, TARGET_TEST_HYPER)])

vocab = list(set([line.strip().split("\t")[0] for line in read_file(DATA_DIR, TARGET_VOCAB)]))

all_words = list()

all_words += hyponyms_train
all_words += hypernyms_train
all_words += hyponyms_dev
all_words += hypernyms_dev
all_words += hyponyms_test
all_words += hypernyms_test

new_vocab = list(set(all_words))

print("LENGTH OF EXISTING VOCAB: {} AND NEW VOCAB: {}".format(len(vocab), len(new_vocab)))

f = open(OUT_FILENAME, "w")
for word in new_vocab:
    f.write(word + "\n")
f.close()

print("NEW VOCAB FILE {} CREATED SUCCESSFULLY.".format(OUT_FILENAME))
