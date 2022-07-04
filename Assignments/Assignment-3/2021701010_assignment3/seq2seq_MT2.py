import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from NNLM_Model import NNLM
from torch import Tensor

#=========================================================================================================

class encoder(nn.Module):
    def __init__(self, weight_arr_enc: torch.tensor, pre_trained_encoder: nn.Module, dropout: float):
        super(encoder, self).__init__()
        # Other Attributes
        self.dropout = dropout
        self.enc = pre_trained_encoder

        # Model Specific
        self.enc.embedding.weight = torch.nn.Parameter(weight_arr_enc)
        self.drop_layer = nn.Dropout(p=self.dropout)

    def forward(self, inputs):
        # inputs of shape (batch_size, max_len_of_source_batch), where each row is a sentence
        emb_out = self.drop_layer(self.enc.embedding(inputs))
        # emb_out shape = (batch_size, max_len_of_source_batch, embedding_vector_size)
        out, hidden = self.enc.gru(emb_out)
        # out shape = (batch_size, max_len, hidden_size)
        # hidden = (stack_of_layers, batch_size, hidden_size)
        return out, hidden

#=========================================================================================================


class decoder(nn.Module):
    def __init__(self, VOCAB_SIZE: int, weight_arr_dec: torch.tensor, dec_hid_dim: int, pre_trained_decoder: nn.Module, dropout: float):
        super(decoder, self).__init__()
        # Other Attributes
        self.dec_hid_dim = dec_hid_dim
        self.output_size = VOCAB_SIZE
        self.dropout = dropout
        self.dec = pre_trained_decoder

        # Model Specific
        self.dec.embedding.weight = torch.nn.Parameter(weight_arr_dec)
        self.out2 = nn.Linear(self.dec_hid_dim, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        self.drop_layer = nn.Dropout(p=self.dropout)

    def forward(self, inputs, hidden):
        # inputs of shape (batch_size, 1), where each row is a word, initially it
        # is a starting token...<sos>
        emb_out = self.drop_layer(self.dec.embedding(inputs))
        # emb_out shape = (batch_size, 1, embedding_vector_size)
        out, hidden = self.dec.gru(emb_out, hidden)
        # out shape = (batch_size, 1, hidden_size)
        # hidden = (stack_of_layers, batch_size, hidden_size)
        out = self.out2(out.squeeze(dim=1))
        # out shape = (batch_size, VOCAB_SIZE),as sueezed through 1st dimension not 0th.
        out = self.softmax(out)
        return out, hidden

#=========================================================================================================

class Seq2Seq_MT2(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.4) -> Tensor:
        batch_size = src.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, max_len,
                              trg_vocab_size).to(self.device)
        _, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[:, 0]
        # Apply Teacher Forcing
        teacher_force = random.random() < teacher_forcing_ratio
        if teacher_force:
            for t in range(1, max_len):
                output, hidden = self.decoder(output.unsqueeze(dim=1), hidden)
                outputs[:, t] = output
                output = trg[:, t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output.unsqueeze(dim=1), hidden)
                outputs[:, t] = output
                output = output.argmax(dim=1)

        return outputs

#=========================================================================================================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Hyper-Parameters of MOdel
    ENC_VOCAB_SIZE = 32085
    DEC_VOCAB_SIZE = 55104
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ENC_DROPOUT = 0
    DEC_DROPOUT = 0
    # Initialize Weight matrix...
    nnlm = NNLM(ENC_VOCAB_SIZE, ENC_EMB_DIM, ENC_HID_DIM, 2, 0)
    enc = encoder(torch.zeros((32085, 512)), nnlm, 0)
    dec = decoder(DEC_VOCAB_SIZE, torch.zeros(
        (55104, 512)), DEC_HID_DIM, nnlm, 0)
    model = Seq2Seq_MT2(enc, dec, device=device).to(device)
    print(model)
    print(enc)
    # print(attn)
    print(dec)
