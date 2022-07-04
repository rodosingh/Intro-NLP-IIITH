with open("../models/train_data.txt", "w") as ff:
    with open("../models/data.txt", "r") as fk:
        for _ in tqdm(range(700000)):
            lines = fk.readline()
            #subset_lines = lines[:100000]
            #for line in subset_lines:
            ff.write(lines)
