import codecs

with codecs.open("dir-data/preprocessed_corpus_and_vocab/english.txt", "r", "utf-8") as f, codecs.open("dir-data/preprocessed_corpus_and_vocab/english_new.txt", "w", "utf-8") as fp:
    cnt = 0
    while True:
        line = f.readline()
        if not line:
            break
        cnt += 1
        if cnt % 100000 == 0:
            print(f"{cnt} lines are processed!")
        if len(line.strip()) > 0:
            fp.write(line)  # .strip()
