# Semantic Parsing: Language to Logical Form with Neural Attention

### Project Overview
In this project, I have replicated the paper [Language to Logical Form with Neural Attention](https://aclanthology.org/P16-1004.pdf) which presents a novel method to perform Semantic parsing. In semantic parsing we hope to extract meaning from natural language and map it to machine interpretable representations– the broader goal being that once meaning is represented in a formal/logical way it can be processed by computers for a wider range of tasks. Through the proposed encoder-decoder architecture with the attention, the authors highlight that the model shows much higher generalizability, lesser need for feature engineering and higher cross-domain application. The more traditional setup, prior to this model, involved a lot more feature engineering and domain-specific understading. The dataset used for this project can be found [here](http://dong.li/lang2logic/seq2seq_jobqueries.zip).

### Model Overview
Five classes:
1. `LSTM` - Inherits from nn.Module; generalized implementation of an LSTM with an embedding layer that is the parent of the Encoder and Decoder classes.
2. `Encoder` - Inherits from LSTM class. Forward pass on the encoder layer takes the source sequence (query) as input, along with a hidden state and cell state (each initialized to zero), passes the source through the encoder and returns the output of the LSTM, along with the last hidden state and cell state.
3. `Attention` - Inherits from nn.Module. In the forward pass takes the encoder states and decoder state (of current timestep) as input, and returns a tensor (batch_sz x hidden_sz) which is represents the "importance"/"focus" of the current decoding timestep on each token of encoding.
4. `Decoder` - Inherits from the LSTM class. Forward pass on the decoder layer looks similar to encoder, but with the added pass through the attention layer. Output of the attention layer is passed through a linear layer to provide the final prediciton tensor (batch_sz x target_vocab_sz), along with the last hidden state and cell state.
5. `Seq2Seq` - Inherits from nn.Module. Merges each of the above layers. In the forward pass receives the source and target sequence as inputs. Passes the source sequence through the encoder, followed by a loop through each position in the target sequence to gather decoder predictions for each timestep. During training the true label of the current timestep serves as an input to the next decoder timestep; at inference the decoder predictions are fed back into the decoder.

Finally, functions `create_model`, `train` and `evaluate` control the creation, training and evaluation of the model respectively.

Train batch size = 20  
Test batch size = 1  
Model hidden size = 256  
Embedding size = 64  
Number of training epochs = 20  

### Results
| Test per-token accuracy | Test exact match accuracy |
|-----------------|----------------|
| 89.98%| 84.28% |
