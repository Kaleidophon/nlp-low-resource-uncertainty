"""
Process Shakespeare's plays into a plain text file for language modelling.
"""

# STD
import re
import os

SHAKESPEARE_DIR = "../data/raw/english/shakespeares"
OUTFILE = "../data/processed/english/ood_test.txt"


if __name__ == "__main__":
    shakespeare_files = os.listdir(SHAKESPEARE_DIR)

    if os.path.exists(OUTFILE):
        os.remove(OUTFILE)

    with open(OUTFILE, "a") as out_file:
        for file in shakespeare_files:
            path = os.path.join(SHAKESPEARE_DIR, file)

            with open(path, "r") as infile:
                # Skip all the info in the beginning
                while True:
                    line = infile.readline()

                    if line.strip() == "ACT 1":
                        break

                prefix = ""
                while True:

                    try:
                        line = infile.readline()
                    except ValueError:
                        break

                    if line == "":
                        break

                    # Underline
                    if "=" in line or line.startswith("Scene"):
                        continue

                    # Empty line
                    elif line.strip() == "":
                        continue

                    # Name line
                    elif line.upper() == line:
                        continue

                    else:
                        # Remove actions / names and such
                        line = line.strip()
                        line = prefix + line
                        line = re.sub("\[.+?\]", "", line)

                        if line.strip() == "":
                            prefix = ""
                            continue

                        tokens = line.split()

                        if tokens[0].upper() == tokens[0]:
                            tokens = tokens[1:]

                        if len(tokens) == 0:
                            prefix = ""
                            continue

                        line = " ".join(tokens)

                        # Check whether line is a full sentence
                        if line[-1] not in ".;?!":
                            prefix = line + " "
                            continue
                        else:
                            prefix = ""

                        line = re.sub("\[.+?\]", "", line)

                        out_file.write(line + "\n")
