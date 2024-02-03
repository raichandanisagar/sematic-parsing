import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
import preprocessing as prep
from model import Seq2Seq


class JobsDataset(Dataset):
    """Defines a Dataset object for the Jobs dataset to be used with Dataloader"""

    def __init__(self, queries, logical_forms):
        """Initializes a JobsDataset

        Parameters
        ----------
        queries : list[list[int]]
            a list of queries, which have been tokenized and padded, in the form
            of a list of vocab indices
        logical_forms : list[list[int]]
            a list of corresponding logical forms, which have been tokenized and
            padded, in the form of a list of vocab indices"""

        self.queries = queries
        self.logical_forms = logical_forms

    def __len__(self) -> int:
        """Returns the amount of paired queries and logical forms in the dataset

        Returns
        ----------
        int
            length of the dataset"""

        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        """Returns a paired query and logical form at the specified index

        Parameters
        ----------
        idx : int
            specified index of the dataset

        Returns
        ----------
        tuple[list[int], list[int]]
            paired query and logical form at the specified index, in the form of
            a list of vocab indices"""

        return self.queries[idx], self.logical_forms[idx]


def build_dataset(query_tokens, lf_tokens):
    return JobsDataset(query_tokens, lf_tokens)


def collate(batch: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Used as collate_fn when creating the Dataloaders from the dataset
      Parameters
      ----------
      batch : list[tuple[list[int], list[int]]]
          a list of outputs of __getitem__

      Returns
      ----------
      tuple[torch.Tensor, torch.Tensor]
          a batched set of input sequences and a batched set of target sequences"""

    src, tgt = default_collate(batch)
    return torch.stack(src), torch.stack(tgt)


def build_dataloader(dataset: JobsDataset, batch_size: int, **kwargs) -> tuple[DataLoader]:
    """Used as collate_fn when creating the Dataloaders from the dataset, batching
      the training data according to the inputted batch size and batching the
      testing data with a batch size of 1

      Parameters
      ----------
      dataset : JobsDataset
          train/test dataset
      batch_size : int
          batch size to be used during train/test

      Returns
      ----------
      tuple[DataLoader, DataLoader]
          a training and testing DataLoader"""

    dataloader = DataLoader(dataset, batch_size=batch_size, **kwargs)
    return dataloader


def train(model: nn.Module, train_dataloader: DataLoader, num_epochs: int = 5,
          device: str = "cpu") -> nn.Module:
    """
  Trains your model!

  Parameters
  ----------
  model : nn.Module
      your model!
  train_dataloader : DataLoader
      a dataloader of the training data from build_dataloaders
  num_epochs : int
      number of epochs to train for
  device : str
      device that the model is running on

  Returns
  ----------
  A Seq2Seq trained model
  """
    model = model.to(device)
    loss_fn = nn.NLLLoss(ignore_index=LF_PAD_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    total_loss = []

    for epoch in range(num_epochs):
        epoch_loss = 0

        for i, (query_batch, label_batch) in enumerate(train_dataloader):
            query_batch = query_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            preds = model(query_batch, label_batch, device, training=True)

            loss = loss_fn(preds[1:].view(-1, preds.size(-1)), label_batch[1:].view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        total_loss.append((epoch_loss / len(train_dataloader)))
        print(f'Epoch: {epoch}, Loss: {epoch_loss / len(train_dataloader)}')

    return model


def evaluate(model: nn.Module, dataloader: DataLoader, device: str = "cuda") -> tuple[int, int]:
    """
  Evaluates your model!

  Parameters
  ----------
  model : nn.Module
      your model!
  dataloader : DataLoader
      a dataloader of the testing data from build_dataloaders
  device : str
      device that the model is running on

  Returns
  ----------
  tuple[int, int]
      per-token accuracy and exact_match accuracy
  """
    total_pta, total_tokens, total_fma = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for i, (query, label) in enumerate(dataloader):
            query = query.to(device)
            label = label.to(device)

            preds = model(query, label, device, training=False)
            pred_tokens = torch.argmax(preds[1:], dim=2)
            check = pred_tokens == label[1:len(preds)]

            total_pta += torch.sum(check)
            total_tokens += len(check)
            total_fma += torch.all(check)

    return (total_pta / total_tokens).item(), (total_fma / (i + 1)).item()


if __name__ == '__main__':
    prep.download_data()
    query_train, lf_train = prep.extract_file('train.txt')
    query_test, lf_test = prep.extract_file('test.txt')

    query_vocab, query_word2idx, query_idx2word = prep.build_vocab_index(query_train, min_word_freq=2)
    lf_vocab, lf_word2idx, lf_idx2word = prep.build_vocab_index(lf_train, min_word_freq=0)

    query_train_tokens = prep.tokenize_and_pad(query_train, query_word2idx)
    query_test_tokens = prep.tokenize_and_pad(query_test, query_word2idx)
    lf_train_tokens = prep.tokenize_and_pad(lf_train, lf_word2idx, length_scale=1.5)
    lf_test_tokens = prep.tokenize_and_pad(lf_test, lf_word2idx, length_scale=1.5)

    jobs_train = build_dataset(query_train_tokens, lf_train_tokens)
    jobs_test = build_dataset(query_test_tokens, lf_test_tokens)

    dataloader_train = build_dataloader(jobs_train, batch_size=25, shuffle=True, collate_fn=collate)
    dataloader_test = build_dataloader(jobs_test, batch_size=1, shuffle=False, collate_fn=collate)

    QUERY_VOCAB_LEN = len(query_vocab)
    QUERY_PAD_INDEX = query_word2idx['<PAD>']
    QUERY_UNK_INDEX = query_word2idx['<UNK>']

    LF_VOCAB_LEN = len(lf_vocab)
    LF_SOS_INDEX = lf_word2idx['<s>']
    LF_EOS_INDEX = lf_word2idx['</s>']
    LF_PAD_INDEX = lf_word2idx['<PAD>']

    HIDDEN_SIZE = 256
    EMBEDDING_SIZE = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Seq2Seq(HIDDEN_SIZE, QUERY_VOCAB_LEN, EMBEDDING_SIZE, QUERY_PAD_INDEX, LF_VOCAB_LEN,
                    EMBEDDING_SIZE, LF_PAD_INDEX, LF_SOS_INDEX, LF_EOS_INDEX)

    model = train(model, dataloader_train, num_epochs=20, device=device)
    test_per_token_accuracy, test_exact_match_accuracy = evaluate(model, dataloader_test, device=device)
    print(f'Test Per-token Accuracy: {test_per_token_accuracy}')
    print(f'Test Exact-match Accuracy: {test_exact_match_accuracy}')
