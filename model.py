import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Defines the LSTM layer for the encoder and decoder of the model."""

    def __init__(self, vocab_sz: int, embd_sz: int, hidden_sz: int,
                 pad_index: int, num_layers: int) -> None:
        """Initialize the LSTM layer with the following parameters:

    Parameters
    ----------
    vocab_sz : int - Size of the vocabulary as per source or target sequence
    embd_sz : int - Size of the embedding layer
    hidden_sz : int - Hidden size of LSTM as defined by torch.nn.LSTM
    pad_index : int - Padding index of the vocabulary as per source or target sequence
    num_layers : int (default 1) - Number of LSTM layers
    """

        super().__init__()

        self.vocab_sz = vocab_sz
        self.embd_sz = embd_sz
        self.hidden_sz = hidden_sz
        self.pad_index = pad_index
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_sz, embd_sz, padding_idx=pad_index)
        self.lstm = nn.LSTM(embd_sz, hidden_sz, num_layers=num_layers)

    def init_hidden(self, batch_sz: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Create the first hidden state and cell state.

    Parameters
    ----------
    batch_sz: int - Number of batches in the sequences
    device: str - Set device to GPU/CPU.

    Returns
    -------
    hidden_state: torch.Tensor (num_layers x batch_sz x hidden_sz) Zero tensor
    cell_state: torch.Tensor (num_layers x batch_sz x hidden_sz) Zero tensor"""

        hidden_state = torch.zeros(self.num_layers, batch_sz, self.hidden_sz).to(device)
        cell_state = torch.zeros(self.num_layers, batch_sz, self.hidden_sz).to(device)
        return hidden_state, cell_state


class Encoder(LSTM):
    """Defines the encoder stack of the model"""

    def __init__(self, vocab_sz: int, embd_sz: int, hidden_sz: int,
                 pad_index: int, num_layers: int) -> None:
        """Initialize the encoder with the following parameters:

    Parameters
    ----------
    vocab_sz : int - Size of the vocabulary as per source or target sequence
    embd_sz : int - Size of the embedding layer
    hidden_sz : int - Hidden size of LSTM as defined by torch.nn.LSTM
    pad_index : int - Padding index of the vocabulary as per source or target sequence
    num_layers : int (default 1) - Number of LSTM layers
    """
        super().__init__(vocab_sz, embd_sz, hidden_sz, pad_index, num_layers)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor | None = None,
                cell_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder layer.

        Parameters
        ----------
        x: torch.Tensor - source sequence as a tensor of integers that represent tokens in the vocabulary.
        hidden_state: torch.Tensor - Hidden state of the last timestep (1 x batch_sz x hidden_sz)
        cell_state: torch.Tensor - Cell state of the last timestep (1 x batch_sz x hidden_sz)

        Returns
        -------
        LSTM output (seq_len x batch_sz x hidden_sz)
        Hidden state of the last timestep (1 x batch_sz x hidden_sz)
        Cell state of the last timestep (1 x batch_sz x hidden_sz)"""

        embd = self.embedding(x)
        out, (hidden_state, cell_state) = self.lstm(embd, (hidden_state, cell_state))
        return out, hidden_state, cell_state


class AttentionLayer(nn.Module):
    """Defines the attention layer of the model"""

    def __init__(self, hidden_sz: int) -> None:
        """Initialize the attention layer

      Parameters
      ----------
      hidden_sz:int - Hidden size of the model"""

        super().__init__()
        self.hidden_sz = hidden_sz

        self.softmax = nn.Softmax(dim=1)
        self.ll1 = nn.Linear(hidden_sz, hidden_sz)
        self.ll2 = nn.Linear(hidden_sz, hidden_sz)
        self.tanh = nn.Tanh()

    def forward(self, encoder_states: torch.Tensor, decoder_state: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention layer.

    Parameters
    ----------
    encoder_states: torch.Tensor - Tensor of encoder hidden states (batch_sz x seq_len x hidden_sz)
    decoder_state: torch.Tensor - Decoder hidden state at the current timestep (batch_sz x hidden_sz x 1)

    Returns
    --------
    The attention tensor of shape (batch_sz x hidden_sz)"""

        alignment = torch.matmul(encoder_states, decoder_state)  # tensor of batch_sz x seq_len x 1
        weights = self.softmax(alignment)  # softmax to redistribute alignment in range [0,1]
        context = torch.matmul(encoder_states.transpose(1, 2), weights)  # tensor of batch_sz x hidden_sz x 1
        attention = self.tanh(self.ll1(decoder_state.squeeze(2)) + self.ll2(context.squeeze(2)))
        return attention


class Decoder(LSTM):
    def __init__(self, vocab_sz: int, embd_sz: int, hidden_sz: int,
                 pad_index: int, num_layers: int) -> None:
        """Initialize the decoder layer

    Parameters
    ----------
    vocab_sz: int - Size of the vocabulary as per source or target sequence
    embd_sz: int - Size of the embedding layer
    hidden_sz: int - Hidden size of LSTM as defined by torch.nn.LSTM
    pad_index: int - Padding index of the vocabulary as per source or target sequence
    num_layers: int - Number of LSTM layers (default 1)"""

        super().__init__(vocab_sz, embd_sz, hidden_sz, pad_index, num_layers)

        self.attention_layer = AttentionLayer(hidden_sz)  # batch_sz x hidden_sz
        self.ll3 = nn.Linear(hidden_sz, vocab_sz)  # batch_sz x vocab_sz
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, encoder_states: torch.Tensor, hidden_state: torch.Tensor,
                cell_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of decoder.

    Parameters
    ----------
    x: torch.Tensor (1 x batch_sz) - Target sequence/token i.e. output of the previous timestep
    encoder_states: torch.Tensor (seq_len x batch_sz x hidden_sz) - Tensor of the encoder hidden states
    hidden_state: torch.Tensor (1 x batch_sz x hidden_sz) - Hidden state of the previous timestep
    cell_state: torch.Tensor (1 x batch_sz x hidden_sz) - Cell state of the previous timestep

    Returns
    --------
    pred: torch.Tensor - (batch_sz x lf_vocab_sz)
    hidden_state: torch.Tensor - (1 x batch_sz x embedding_sz)
    cell_state: torch.Tensor - (1 x batch_sz x embedding_sz)
    """

        embd = self.embedding(x)
        out, (hidden_state, cell_state) = self.lstm(embd, (hidden_state, cell_state))

        # the decoder hidden state transpose is necessary to ensure the tensor multiplication is appropriate ((1 x batch_sz x hidden_sz) -> (batch_sz x hidden_sz x 1))
        decoder_state = hidden_state.transpose(0, 1).transpose(1, 2)

        attention = self.attention_layer(encoder_states.transpose(0, 1), decoder_state)
        output = self.ll3(attention)  # batch_sz x vocab_sz
        pred = self.softmax(output)
        return pred, hidden_state, cell_state


class Seq2Seq(nn.Module):
    def __init__(self, hidden_sz: int,
                 encoder_vocab_sz: int, encoder_embd_sz: int, encoder_pad_index: int,
                 decoder_vocab_sz: int, decoder_embd_sz: int, decoder_pad_index: int,
                 decoder_sos_index: int, decoder_eos_index: int,
                 encoder_num_layers=1, decoder_num_layers=1) -> None:
        """Initialize a sequence to sequence model

    Parameters
    ----------
    hidden_sz: int - Hidden size of the model that remains constant through the encoder and decoder LSTM layers
    encoder_vocab_sz: int - Size of vocabulary of the input sequence
    encoder_embd_sz: int - Size of the encoder embedding layer
    encoder_pad_index: int - Padding index in the input tokens
    decoder_vocab_sz: int - Size of the vocabulary of the tokens in the output sequence
    decoder_embd_sz: int - Size of the decoder embedding layer
    decoder_pad_index: int - Padding index in the target tokens
    decoder_sos_index: int - Start of sentence index in the target tokens
    decoder_eos_index: int - End of sentence index in the target tokens
    encoder_num_layers: int - Number of LSTM layers in the encoder (default 1)
    decoder_num_layers: int - Number of LSTM layers in the decoder (default 1)"""

        super().__init__()

        self.hidden_sz = hidden_sz

        self.encoder_vocab_sz = encoder_vocab_sz
        self.encoder_embd_sz = encoder_embd_sz
        self.encoder_pad_index = encoder_pad_index

        self.decoder_vocab_sz = decoder_vocab_sz
        self.decoder_embd_sz = decoder_embd_sz
        self.decoder_pad_index = decoder_pad_index

        self.decoder_sos_index = decoder_sos_index
        self.decoder_eos_index = decoder_eos_index

        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers

        self.encoder = Encoder(encoder_vocab_sz, encoder_embd_sz, hidden_sz,
                               encoder_pad_index, encoder_num_layers)

        self.decoder = Decoder(decoder_vocab_sz, decoder_embd_sz, hidden_sz,
                               decoder_pad_index, decoder_num_layers)

    def forward(self, input_sequence: torch.Tensor, target_sequence: torch.Tensor,
                device: str, training: bool = True):
        """Forward pass of the sequence-to-sequence model.

    Parameters
    ----------
    input_sequence: torch.Tensor (input_seq_len x batch_size) - Tokenized input sequence
    target_sequence: torch.Tensor (target_seq_len x batch_size) - Tokenized target sequence
    device: str - set GPU/CPU
    training: bool - True when model is being called in training loop; else False

    Returns
    --------
    Decoder predictions as a tensor of the shape (seq_len x batch_sz x target_vocab_sz)."""

        hidden_state, cell_state = self.encoder.init_hidden(input_sequence.shape[1], device)
        encoder_states, hidden_state, cell_state = self.encoder(input_sequence, hidden_state, cell_state)

        target_batch_sz = target_sequence.shape[1]
        target_max_len = target_sequence.shape[0]

        decoder_preds = torch.zeros(target_max_len, target_batch_sz, self.decoder_vocab_sz).to(device)
        decoder_input = torch.empty(1, target_batch_sz, dtype=torch.long).fill_(self.decoder_sos_index).to(device)

        for i in range(1, target_max_len):
            pred, hidden_state, cell_state = self.decoder(decoder_input, encoder_states, hidden_state, cell_state)
            decoder_preds[i] = pred

            top_pred = torch.argmax(pred, dim=1)
            label = target_sequence[i]

            if training:
                decoder_input = label.unsqueeze(0)
            else:
                if label[0].item() == self.decoder_eos_index:
                    decoder_preds = decoder_preds[:i + 1]
                    break
                decoder_input = top_pred.unsqueeze(0)
        return decoder_preds


