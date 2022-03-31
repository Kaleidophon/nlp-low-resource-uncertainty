"""
Preprocess the CLINC OOS dataset by writing splits into different files, tokenizing and
"""

# STD
import codecs
import json

# EXT
from transformers import BertTokenizer

# CONST
CLINC_PATHS = {
    "clinc": "../data/raw/clinc/data_full.json",
    "clinc_plus": "../data/raw/clinc/data_oos_plus.json",
}


if __name__ == "__main__":

    for dataset, path in CLINC_PATHS.items():

        with open(path, "r") as file:
            data = json.load(file)

            for split, sentences in data.items():
                tokenized_sentences, labels = zip(
                    *[
                        (
                            sentence.split(" "),
                            label,
                        )
                        for sentence, label in sentences
                        if label != "oos" or "oos" in split
                    ]
                )

                with codecs.open(
                    f"../data/processed/{dataset}/{split}.csv", "wb", "utf-8"
                ) as split_file:
                    for tokenized_sentence, label in zip(tokenized_sentences, labels):
                        split_file.write(f"{' '.join(tokenized_sentence)}\t{label}\n")
