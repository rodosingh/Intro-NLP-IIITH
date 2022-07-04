# loading data. Consider changing the path to the actual path where file is present
with open("models/data.txt", "w") as fp:
    with open('reviews_Electronics_5.json') as file:
        for line in tqdm(file):
            fp.write(re.sub(' +', ' ', ''.join(ch if ch.isalnum() else ' ' for ch in json.loads(line)['reviewText'].lower()))+'\n')
