import torch
import torch.nn as nn
from torch import Tensor
import random
import torch.nn.functional as F

#==========================================================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float):
        super(Encoder, self).__init__()
        # create attributes
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        # create objects
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor):
        # forward function that tries to fit the model
        embedded = self.dropout(self.embedding(src))
        outputs, hid = self.rnn(embedded)
        hid = torch.tanh(self.fc(torch.cat((hid[-2, :, :], hid[-1, :, :]), dim=1)))
        return outputs, hid

#==========================================================================================================

class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        # forward function that tries to fit the model
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        importance_weight = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
        attention = torch.sum(importance_weight, dim=2)
        return F.softmax(attention, dim=1)

#==========================================================================================================

class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: int, attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self, input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor):
        # forward function that tries to fit the model
        # print(f"INput Shape in Decoder = {input.shape}")
        embedded = self.dropout(self.embedding(input.unsqueeze(0)))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        output = self.out(torch.cat((output.squeeze(0), weighted_encoder_rep.squeeze(0), embedded.squeeze(0)), dim=1))

        return output, decoder_hidden.squeeze(0)

#==========================================================================================================

class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5):

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0, :]
        # for teacher forcing with a whole sentence
        teacher_force = random.random() < teacher_forcing_ratio
        if teacher_force:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_outputs)
                outputs[t] = output
                output = trg[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_outputs)
                outputs[t] = output
                output = output.max(1)[1]

        return outputs

#==========================================================================================================

if __name__ == '__main__':
    
    # Mention Device...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define Hyper-Parameters of MOdel
    INPUT_DIM = 55000
    OUTPUT_DIM = 32000
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ATTN_DIM = 32
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM,
                DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)
    # print(model)
    # print(enc)
    # print(attn)
    # print(dec)
