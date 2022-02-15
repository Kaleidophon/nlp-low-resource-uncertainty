"""
Process the Finnish UD dataset.
"""

# STD
import os

DATA_DIR = "../data/raw/finnish"
TARGET_DIR = "../data/processed/finnish"
FILES = {
    "fi_tdt-ud-train.conllu": "train.csv",
    "fi_tdt-ud-dev.conllu": "val.csv",
    "fi_tdt-ud-test.conllu": "test.csv",
    "fi_ood-ud-test.conllu": "ood_test.txt",
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

                elif line.startswith("#"):
                    continue

                else:
                    parts = line.split("\t")
                    tokens.append(parts[1])
                    labels.append(parts[3])
