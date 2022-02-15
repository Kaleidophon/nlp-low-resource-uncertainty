"""
Get the perplexity using a trigram language model for some sub-sampled corpus.
"""

# STD
import os

# CONST
# TODO: Turn these into script args
MACHINE_TYPE = "macosx"
RELATIVE_SRILM_PATH = "../../srilm-1.7.3/"
RELATIVE_SRILM_CMD_PATH = f"../../srilm-1.7.3/bin/{MACHINE_TYPE}"
NUM_EVALS = 5


def get_srilm_perplexity(train_path: str, test_path: str) -> float:
    ngrams_path = "./temp.lm"
    weights_path = "./temp2.lm"

    # Build n-gram counts and interpolation weights
    os.system(
        f"{RELATIVE_SRILM_CMD_PATH}/ngram-count -text {train_path} -lm {ngrams_path} --order 3"
    )
    os.system(
        f"{RELATIVE_SRILM_CMD_PATH}/ngram-count -text {train_path} -init-lm {ngrams_path} -lm {weights_path}"
    )

    # Get results
    stream = os.popen(
        f"{RELATIVE_SRILM_CMD_PATH}/ngram -lm {weights_path} -ppl {test_path}"
    )
    output = str(stream.read())
    ppl = float(output.strip().split()[-3])

    # Clean up
    os.remove(ngrams_path)
    os.remove(weights_path)

    return ppl


if __name__ == "__main__":
    # Set some environment variables for SRILM
    os.environ["SRILM"] = os.path.abspath(RELATIVE_SRILM_PATH)
    os.environ["MACHINE_TYPE"] = MACHINE_TYPE
    os.environ[
        "PATH"
    ] = f"{os.environ['PATH']}:{os.path.abspath(RELATIVE_SRILM_PATH)}/bin/{MACHINE_TYPE}"

    # Sub-sample training data and save to temporary file
    # TODO

    # Get perplexities over multiple runs
    # TODO: Debug
    get_srilm_perplexity("./train.txt", "./test.txt")
