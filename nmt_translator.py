
import os
import random
from functools import lru_cache

import torch
import torch.nn as nn
import sentencepiece as spm

DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, MAX_LEN = 96, 192, 1, 0.2, 24
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class DecoderBasic(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, token, hidden, cell):
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return self.fc_out(output.squeeze(1)), hidden, cell

class Seq2SeqBasic(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

class EncoderAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0,
                             batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        n_layers = hidden.shape[0] // 2
        hidden = torch.cat([hidden[:n_layers], hidden[n_layers:]], dim=2)
        cell = torch.cat([cell[:n_layers], cell[n_layers:]], dim=2)
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
    def forward(self, decoder_hidden, encoder_outputs, mask):
        src_len = encoder_outputs.shape[1]
        hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=1)

class DecoderAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.attention = Attention(hid_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(hid_dim * 2 + emb_dim, hid_dim, n_layers,
                             dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, token, hidden, cell, encoder_outputs, mask):
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))
        attention = self.attention(hidden[-1], encoder_outputs, mask)
        weighted = torch.bmm(attention.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell, attention

class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

@lru_cache(maxsize=1)
def get_tokenizers():
    sp_en = spm.SentencePieceProcessor(model_file=f"{CHECKPOINT_DIR}/en_tokenizer.model")
    sp_am = spm.SentencePieceProcessor(model_file=f"{CHECKPOINT_DIR}/am_tokenizer.model")
    return sp_en, sp_am

@lru_cache(maxsize=1)
def get_model():
    sp_en, sp_am = get_tokenizers()
    src_vocab, tgt_vocab = sp_en.get_piece_size(), sp_am.get_piece_size()
    model = Seq2SeqAttn(
        EncoderAttn(src_vocab, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT),
        DecoderAttn(tgt_vocab, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT),
        DEVICE,
    ).to(DEVICE)
    ckpt = torch.load(f"{CHECKPOINT_DIR}/attention_seq2seq_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def normalize_english(text):
    import re
    return re.sub(r"\s+", " ", str(text).strip().lower())

@torch.inference_mode()
def translate(text: str) -> str:
    sp_en, sp_am = get_tokenizers()
    model = get_model()
    clean = normalize_english(text)
    ids = [BOS_ID] + sp_en.encode(clean, out_type=int)[:MAX_LEN] + [EOS_ID]
    src = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    enc_out, hidden, cell = model.encoder(src)
    mask = src != PAD_ID
    token = torch.tensor([BOS_ID], dtype=torch.long, device=DEVICE)
    generated = []
    for _ in range(MAX_LEN):
        output, hidden, cell, _ = model.decoder(token, hidden, cell, enc_out, mask)
        nxt = output.argmax(1).item()
        if nxt == EOS_ID:
            break
        if nxt not in (PAD_ID, BOS_ID):
            generated.append(nxt)
        token = torch.tensor([nxt], dtype=torch.long, device=DEVICE)
    return sp_am.decode(generated)
