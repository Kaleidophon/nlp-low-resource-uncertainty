"""
Process the swahili news classification dataset.
"""

# EXT
from spacy.pipeline import Sentencizer
from spacy.lang.en import English

DATA_PATH = "../data/raw/swahili/train.csv"
TARGET_PATH = "../data/processed/swahili/ood_test.csv"


if __name__ == "__main__":
    nlp = English()
    sent = Sentencizer("sentencizer")
    punct_chars = sent.default_punct_chars
    nlp.add_pipe("sentencizer")
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))

    with open(DATA_PATH, "r") as in_file, open(TARGET_PATH, "w") as out_file:
        lines = []

        for i, line in enumerate(in_file.readlines()):
            if i == 0:
                continue

            line = line.strip()
            print(line)

            if line == "":
                continue

            lines.append(line)

            # Split all lines found so far
            if line[-1] in punct_chars:
                cat_lines = " ".join(lines)

                # Sentencize
                for s in nlp(cat_lines).sents:
                    out_file.write(str(s) + "\n")

                lines = []
