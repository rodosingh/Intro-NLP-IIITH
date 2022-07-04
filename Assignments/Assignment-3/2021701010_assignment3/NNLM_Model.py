import torch.nn as nn

class NNLM(nn.Module):
    def __init__(self, VOCAB_SIZE: int, emb_dim: int, enc_hid_dim: int, num_stacks: int, dropout: float):
        super(NNLM, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.embedding_size = emb_dim
        self.output_size = VOCAB_SIZE
        self.dropout = dropout
        self.num_layers = num_stacks

        self.embedding = nn.Embedding(VOCAB_SIZE, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.enc_hid_dim,
                          num_layers=self.num_layers, batch_first=True, dropout=dropout)
        # input_size, output_size
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.out1 = nn.Linear(self.enc_hid_dim, self.output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        out = self.drop_layer(self.embedding(inputs))
        outs, hidden = self.gru(out)
        out = self.log_softmax(self.out1(outs[:, -1]))
        return out, hidden

if __name__ == "__main__":
    VOCAB_SIZE=1000
    emb_dim=512
    enc_hid_dim=256
    num_stacks=3
    dropout=0
    nnlm = NNLM(VOCAB_SIZE, emb_dim, enc_hid_dim, num_stacks, dropout)
