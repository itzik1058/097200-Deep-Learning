import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


def weight_norm_linear(in_features, out_features):
    return weight_norm(nn.Linear(in_features, out_features), dim=None)


class VQA(nn.Module):
    def __init__(self, question_encoder, image_encoder, attention, decoder, hidden_dim):
        super(VQA, self).__init__()
        self.question_encoder = question_encoder
        self.image_encoder = image_encoder
        self.attention = attention
        self.question_hidden = weight_norm_linear(question_encoder.hidden_dim, hidden_dim)
        self.image_hidden = weight_norm_linear(image_encoder.hidden_dim, hidden_dim)
        self.decoder = decoder
        self.num_classes = decoder.num_classes

    def forward(self, image, question):
        image = self.image_encoder(image)
        question = self.question_encoder(question)
        attention = self.attention(image, question)
        image_attention = attention.sum(dim=1).unsqueeze(dim=2)
        image = torch.mul(image, image_attention).sum(dim=1)
        image = self.image_hidden(image)
        question = self.question_hidden(question)
        return self.decoder(torch.mul(image, question))


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, question_length, embed_dim, num_layers, dropout):
        super(QuestionEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.question_length = question_length
        self.hidden_dim = embed_dim * (1 + self.rnn.bidirectional)

    def forward(self, question):
        assert question.size(1) == self.question_length
        embed = self.embedding(question)
        states, _ = self.rnn(embed)
        forward = states[:, -1, :self.rnn.hidden_size]
        backward = states[:, 0, self.rnn.hidden_size:]
        final_state = torch.cat((forward, backward), dim=1)
        final_state = final_state.view(question.size(0), -1)
        return final_state


class ImageEncoder(nn.Module):
    def __init__(self, channels=3):
        super(ImageEncoder, self).__init__()
        self.hidden_dim = channels
        feature_layers = []
        for layer in [18, 'M', 36, 'M', 72, 72, 'M', 144, 144, 'M', 288, 288, 'M']:
            if layer == 'M':
                feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                feature_layers.append(nn.Conv2d(self.hidden_dim, layer, kernel_size=3, padding=1))
                feature_layers.append(nn.BatchNorm2d(layer))
                feature_layers.append(nn.ReLU(inplace=True))
                self.hidden_dim = layer
        self.features = nn.Sequential(*feature_layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        return x.flatten(start_dim=2).transpose(1, 2)


class VQAttention(nn.Module):
    def __init__(self, image_dim, question_dim, hidden_dim, dropout):
        super(VQAttention, self).__init__()
        self.image_hidden = weight_norm_linear(image_dim, hidden_dim)
        self.question_hidden = weight_norm_linear(question_dim, hidden_dim)
        self.attention = nn.Sequential(nn.Dropout(dropout), weight_norm_linear(hidden_dim, 1), nn.Softmax(dim=1))

    def forward(self, image, question):
        image_hidden = self.image_hidden(image)
        question_hidden = self.question_hidden(question).unsqueeze(dim=1)
        return self.attention(image_hidden * question_hidden)


class VQADecoder(nn.Module):
    def __init__(self, state_dim, vocab_dim, dropout):
        super(VQADecoder, self).__init__()
        self.num_classes = vocab_dim
        self.decoder = nn.Sequential(
            weight_norm_linear(state_dim, vocab_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            weight_norm_linear(vocab_dim, vocab_dim)
        )

    def forward(self, state):
        return self.decoder(state)


def make_model(vocab_size, num_classes, question_length, hidden_dim):
    q_encoder = QuestionEncoder(vocab_size, question_length, embed_dim=300, num_layers=2, dropout=0.3)
    i_encoder = ImageEncoder()
    attention = VQAttention(i_encoder.hidden_dim, q_encoder.hidden_dim, hidden_dim, dropout=0.4)
    decoder = VQADecoder(hidden_dim, num_classes, dropout=0.5)
    return VQA(q_encoder, i_encoder, attention, decoder, hidden_dim)
