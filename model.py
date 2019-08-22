import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import Settings as s

class Network(nn.Module):

    def __init__(self, vocab_size, embed_d, hidden_d, output_d, context_d,
                 dropout, pad_idx):
        super().__init__()

        self.device = s.device

        self.hidden_d = hidden_d

        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(vocab_size, embed_d,
                                      padding_idx=pad_idx)

        self.lstm = nn.LSTM(bidirectional=True, num_layers=2,
                            input_size=embed_d, hidden_size=hidden_d,
                            batch_first=True, dropout=dropout)

        ## Word-level hierarchical attention:
        self.ui = nn.Linear(2*hidden_d, context_d)
        self.uw = nn.Parameter(torch.randn(context_d))

        ## Output:
        self.fc = nn.Linear(2*hidden_d, output_d)

    def forward(self, x, seqlens):

        x = x.to(self.device) # B X T

        embeds = self.embedding(x) # B X T X EmbD
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, seqlens,
                                                          batch_first=True)
            # 960 (B*T) X 300 (N) B*T X EmbD

        enc_packed, (h_n, c_n) = self.lstm(packed_embeds)
            # (B*T) X HdD*2
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc_packed,
                                                  batch_first=True)
            # B X T X HdD*2

        ## Word-level hierarchical attention:
        u_it = torch.tanh(self.ui(enc)) # B X T X CtD
        weights = torch.softmax(u_it.matmul(self.uw), dim=1).unsqueeze(1)
            # B X 1 X T
        sent = torch.sum(weights.matmul(enc), dim=1) # B X HdD*2

        logits = self.fc(sent) # B X OutD

        if s.args.dataset == 'imdb':
            preds = torch.round(torch.sigmoid(logits))
        else:
            preds = logits.argmax(-1)

        return logits, weights, preds
