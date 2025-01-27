# model_def.py

import torch.nn as nn
import torch

class SimpleBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_eng_feats):
        super(SimpleBiLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.bilstm = nn.LSTM(emb_dim, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.feat_dense = nn.Sequential(
            nn.Linear(n_eng_feats, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.final = nn.Sequential(
            nn.Linear(128 * 2 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x, eng):
        e = self.emb(x)
        o, _ = self.bilstm(e)
        o = self.dropout(o)
        fwd = o[:, -1, :128]
        bwd = o[:, 0, 128:]
        lstm_out = torch.cat([fwd, bwd], dim=-1)
        feat_out = self.feat_dense(eng)
        combined = torch.cat([lstm_out, feat_out], dim=-1)
        return self.final(combined)
