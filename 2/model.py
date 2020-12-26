import torch
import torch.nn as nn


class VQA(nn.Module):
    def __init__(self, q_embedding, i_embedding, attention, decoder):
        super(VQA, self).__init__()
        self.q_embedding = q_embedding
        self.i_embedding = i_embedding
        self.attention = attention
        self.decoder = decoder

    def forward(self, image, q_embed, question_length):
        i_embed = self.i_embedding(image)
        q_embed, q_state = self.q_embedding(q_embed, question_length)
        # i_embed *= self.attention(i_embed, q_state)
        return self.decoder(torch.mul(i_embed, q_state))


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_dim, embed_dim, num_layers, hidden_dim, dropout):
        super(QuestionEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(in_features=embed_dim * num_layers * 2, out_features=hidden_dim)

    def forward(self, question, length):
        embed = self.embedding(question)
        packed = nn.utils.rnn.pack_padded_sequence(embed, length, batch_first=True)
        lstm_out, (h, c) = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        state = torch.cat((h, c))
        state = state.view(question.size(0), -1)
        return unpacked, self.linear(state)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim, channels=3):
        super(ImageEncoder, self).__init__()
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
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, hidden_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VQAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(VQAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, i_embed, q_embed):
        iq_embed = torch.cat((i_embed, q_embed), dim=1)
        return self.attention(iq_embed)


class VQADecoder(nn.Module):
    def __init__(self, state_dim, vocab_dim, dropout=0.5):
        super(VQADecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            nn.Linear(state_dim, vocab_dim),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            nn.Linear(vocab_dim, vocab_dim),
        )

    def forward(self, state):
        return self.decoder(state)


def make_model(q_vocab, a_vocab, hidden_dim):
    q_embedding = QuestionEncoder(len(q_vocab), 300, 2, hidden_dim, dropout=0.1)
    i_embedding = ImageEncoder(hidden_dim)
    attention = VQAttention(hidden_dim)
    decoder = VQADecoder(hidden_dim, len(a_vocab), dropout=0.5)
    return VQA(q_embedding, i_embedding, attention, decoder)
