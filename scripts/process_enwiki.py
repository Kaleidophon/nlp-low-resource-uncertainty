"""
Process the English and the Swahili Wikipedia.
"""

# STD
import codecs
import os
import random

# CONST
FILE_PATHS = {
    "../data/raw/english/enwiki.txt": "../data/processed/english/",
    # "../data/raw/swahili/swwiki.txt": "../data/processed/swahili/"
}  # Define input and outputs paths for wiki files
SPLITS = (0.8, 0.1, 0.1)
MIN_TOKENS = 4


if __name__ == "__main__":

    for in_path, out_dir in FILE_PATHS.items():
        print(in_path)
        train_path = os.path.join(out_dir, "train.txt")
        val_path = os.path.join(out_dir, "val.txt")
        test_path = os.path.join(out_dir, "test.txt")

        with codecs.open(in_path, "r", "utf-8", errors="ignore") as in_file:
            with codecs.open(train_path, "w", "utf-8") as train_file, codecs.open(
                val_path, "w", "utf-8"
            ) as val_file, codecs.open(test_path, "w", "utf-8") as test_file:
                article_lines = []

                for line in in_file.readlines():
                    print(line)
                    line = line.strip()

                    if line.startswith("<doc"):
                        continue

                    # End of article, process lines
                    elif line.startswith("</doc"):
                        r = random.random()

                        if r < SPLITS[0]:
                            train_file.write("\n".join(article_lines))

                        elif SPLITS[0] <= r < SPLITS[0] + SPLITS[1]:
                            val_file.write("\n".join(article_lines))

                        else:
                            test_file.write("\n".join(article_lines))

                        article_lines = []

                    else:
                        if len(line.split()) > MIN_TOKENS and "|" not in line:
                            article_lines.append(line)
