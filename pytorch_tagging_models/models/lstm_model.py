import torch
from torch.nn import Module, Embedding, LSTM, Linear, Sequential, Dropout
from torch.nn.utils.rnn import PackedSequence

from lstm_dataset import Vocabularizer

class LSTMClassifier(Module):
    def __init__(self,
                 num_classes,
                 dropout_rate, 
                 hidden_size_lstm, 
                 num_layers_lstm,
                 dropout_rate_lstm,
                 bidirectional_lstm,
                 out_size,
                 vocab: Vocabularizer, embeds):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate_lstm = dropout_rate_lstm
        self.num_layers_lstm = num_layers_lstm
        self.hidden_size_lstm = hidden_size_lstm
        self.bidirectional_lstm = bidirectional_lstm
        self.out_size = out_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.emb_matrix = embeds
        self.embeddings = Embedding.from_pretrained(embeds, 
                                                    freeze=False, 
                                                    padding_idx=vocab.PAD_IDX)

        self.lstm = LSTM(input_size=embeds.size(1),
                         batch_first=True,
                         hidden_size=hidden_size_lstm,
                         num_layers=num_layers_lstm,
                         dropout=dropout_rate_lstm,
                         bidirectional=bidirectional_lstm)

        self.out_dropout = Dropout(dropout_rate)

        cur_out_size = hidden_size_lstm * num_layers_lstm
        if bidirectional_lstm:
            cur_out_size *= 2
        out_layers = []
        for cur_hidden_size in out_size:
            out_layers.append(Linear(cur_out_size, cur_hidden_size))
            cur_out_size = cur_hidden_size
        out_layers.append(Linear(cur_out_size, num_classes))

        self.net = Sequential(*out_layers)

    def forward(self, input):
        x = input.data
        x = self.embeddings(x)
        x_pack = PackedSequence(x, input.batch_sizes,
                                sorted_indices=input.sorted_indices,
                                unsorted_indices=input.unsorted_indices)
        _, (ht, _) = self.lstm(x_pack)
        ht = ht.transpose(0, 1)
        ht = ht.reshape(ht.size(0), -1)
        output = self.net(ht)
        return output
    
