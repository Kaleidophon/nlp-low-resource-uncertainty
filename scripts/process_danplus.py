"""
Process the Dan+ dataset. In this case, it is just about using at most single
NE tag per word and selecting the train file.
"""

import os

DATA_DIR = "../data/raw/danish"
TARGET_DIR = "../data/processed/danplus"
FILES = {
    "da_news_train.tsv": "train.csv",
    "da_news_dev.tsv": "val.csv",
    "da_news_test.tsv": "test.csv",
    "da_twitter_test.tsv": "ood_test.txt",
}


if __name__ == "__main__":

    for in_file_name, out_file_name in FILES.items():
        in_path, out_path = os.path.join(DATA_DIR, in_file_name), os.path.join(
            TARGET_DIR, out_file_name
        )

        with open(in_path, "r") as in_file, open(out_path, "w") as out_file:
            tokens, labels = [], []

            for line in in_file.readlines():
                line = line.strip()
                line = line.replace('"', "'")

                if line == "":
                    if len(tokens) == len(labels):
                        out_file.write(f"{' '.join(tokens)}\t{' '.join(labels)}\n")
                    tokens, labels = [], []

                else:
                    parts = line.split("\t")
                    tokens.append(parts[0])
                    labels.append(parts[1])
