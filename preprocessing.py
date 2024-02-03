import requests
import io
import zipfile
from collections import Counter

FILEPATH = 'files/'
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'


def download_data(url: str = 'http://dong.li/lang2logic/seq2seq_jobqueries.zip') -> None:
    """Download and unzip raw data, train.txt and test.txt from the source url.

    Parameters
    ----------
    url : str
        data resource url"""

    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
            zip_ref.extractall(FILEPATH)
        print("Extraction completed.")
    else:
        print("Failed to download the zip file.")


def extract_file(filename: str) -> tuple[list[list[str]], list[list[str]]]:
    """Extracts queries and corresponding logical forms from either
    train.txt or test.txt.

    Parameters
    ----------
    filename : str
        name of the file to extract from

    Returns
    ----------
    tuple[list[list[str]], list[list[str]]]
        a tuple of a list of queries and their corresponding logical forms
        each in the form of a list of string tokens"""

    queries, logical_forms = [], []
    with open(FILEPATH + filename) as f:
        for line in f:
            line = line.strip()  # remove new line character
            query, logical_form = line.split('\t')

            query = query.split(' ')[::-1]  # reversed inputs are used the paper (section 4.2)
            logical_form = ["<s>"] + logical_form.split(' ') + ["</s>"]

            queries.append(query)
            logical_forms.append(logical_form)
    return queries, logical_forms


def build_vocab_index(data: list[list], min_word_freq: int = 0):
    vocab = Counter()
    for val in data:
        vocab.update(val)

    word2idx = {}

    for word, count in vocab.items():
        if count >= min_word_freq:
            word2idx[word] = len(word2idx)
    word2idx['<UNK>'] = len(word2idx)
    word2idx['<PAD>'] = len(word2idx)

    idx2word = {i: word for word, i in word2idx.items()}
    vocab = list(word2idx.keys())

    return vocab, word2idx, idx2word


def tokenize(data: list[list], word2idx: dict):
    """Tokenize the data, mapping each word to a unique token index in the vocab dictionary.

    Parameters
    ----------
    data : """
    tokens = [[word2idx.get(word, word2idx['<UNK>']) for word in sentence] for sentence in data]
    return tokens


def pad(seq, max_len, pad_token_idx):
    """Pads a given sequence to the max length using the given padding token index

      Parameters
      ----------
      seq : list[int]
          sequence in the form of a list of vocab indices
      max_len : int
          length sequence should be padded to
      pad_token_idx
          vocabulary index of the padding token

      Returns
      ----------
      list[int]
          padded sequence"""

    seq = seq[:max_len]
    padded_seq = seq + (max_len - len(seq)) * [pad_token_idx]
    return padded_seq


def tokenize_and_pad(data: list[list], word2idx: dict, length_scale=1):
    tokens = tokenize(data, word2idx)
    max_target_len = int(max([len(i) for i in tokens])*length_scale)
    tokens = [pad(i, max_target_len, word2idx[PAD_TOKEN]) for i in tokens]
    return tokens


if __name__ == '__main__':
    download_data()
    query_train, lf_train = extract_file('train.txt')
    query_test, lf_test = extract_file('test.txt')

    query_vocab, query_word2idx, query_idx2word = build_vocab_index(query_train, min_word_freq=2)
    lf_vocab, lf_word2idx, lf_idx2word = build_vocab_index(lf_train, min_word_freq=0)

    query_train_tokens = tokenize_and_pad(query_train, query_word2idx)
    query_test_tokens = tokenize_and_pad(query_test, query_word2idx)
    lf_train_tokens = tokenize_and_pad(lf_train, lf_word2idx, length_scale=1.5)
    lf_test_tokens = tokenize_and_pad(lf_test, lf_word2idx, length_scale=1.5)

