import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


def weight_norm_linear(in_features, out_features):
    return weight_norm(nn.Linear(in_features, out_features), dim=None)


class VQA(nn.Module):
    def __init__(self, q_embedding, i_embedding, attention, decoder, hidden_dim):
        super(VQA, self).__init__()
        self.q_embedding = q_embedding
        self.i_embedding = i_embedding
        self.attention = attention
        self.q_hid = weight_norm_linear(q_embedding.hidden_dim, hidden_dim)
        self.i_hid = weight_norm_linear(i_embedding.hidden_dim, hidden_dim)
        self.decoder = decoder
        self.num_classes = decoder.num_classes

    def forward(self, image, q_embed):
        i_embed = self.i_embedding(image)
        q_embed = self.q_embedding(q_embed)
        attention = self.attention(i_embed, q_embed)
        i_embed = torch.mul(i_embed, attention).sum(dim=1)
        return self.decoder(torch.mul(self.i_hid(i_embed), self.q_hid(q_embed)))


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_dim, embed_dim, num_layers):
        super(QuestionEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True)
        self.rnn_hid = embed_dim
        self.hidden_dim = embed_dim * (1 + self.rnn.bidirectional)

    def forward(self, question):
        embed = self.embedding(question)
        rnn_out, _ = self.rnn(embed)
        forward = rnn_out[:, -1, :self.rnn_hid]
        backward = rnn_out[:, 0, self.rnn_hid:]
        state = torch.cat((forward, backward), dim=1)
        state = state.view(question.size(0), -1)
        return state


class ImageEncoder(nn.Module):
    def __init__(self, channels=3):
        super(ImageEncoder, self).__init__()
        self.hidden_dim = 512
        feature_layers = []
        for layer in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']:
            if layer == 'M':
                feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                feature_layers.append(nn.Conv2d(channels, layer, kernel_size=3, padding=1))
                feature_layers.append(nn.BatchNorm2d(layer))
                feature_layers.append(nn.ReLU(inplace=True))
                channels = layer
        self.features = nn.Sequential(*feature_layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        return x.flatten(start_dim=2).transpose(1, 2)


class VQAttention(nn.Module):
    def __init__(self, i_dim, q_dim, hidden_dim, dropout):
        super(VQAttention, self).__init__()
        self.i_hid = weight_norm_linear(i_dim, hidden_dim)
        self.q_hid = weight_norm_linear(q_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Dropout(dropout),
            weight_norm_linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, i_embed, q_embed):
        i_hid = self.i_hid(i_embed)
        q_hid = self.q_hid(q_embed).unsqueeze(dim=1).repeat(1, i_embed.size(1), 1)
        return self.attention(torch.mul(i_hid, q_hid))


class VQADecoder(nn.Module):
    def __init__(self, state_dim, vocab_dim, dropout):
        super(VQADecoder, self).__init__()
        self.num_classes = vocab_dim
        self.decoder = nn.Sequential(
            weight_norm_linear(state_dim, vocab_dim),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            weight_norm_linear(vocab_dim, vocab_dim)
        )

    def forward(self, state):
        return self.decoder(state)


def make_model(q_vocab, a_vocab, hidden_dim):
    q_embedding = QuestionEncoder(len(q_vocab), 300, 2)
    i_embedding = ImageEncoder()
    attention = VQAttention(i_embedding.hidden_dim, q_embedding.hidden_dim, hidden_dim, dropout=0.2)
    decoder = VQADecoder(hidden_dim, len(a_vocab), dropout=0.5)
    return VQA(q_embedding, i_embedding, attention, decoder, hidden_dim)
