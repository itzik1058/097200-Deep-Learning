import torch
import torch.utils.data as data
import json
from PIL import Image


class TokenDictionary:
    def __init__(self):
        self.words = []
        self.tokens = {}

    def __len__(self):
        return len(self.tokens)

    @property
    def pad_token(self):
        return len(self.tokens)

    def tokenize(self, sentence, as_sentence=False):
        tokens = []
        if as_sentence:
            if sentence not in self.tokens:
                self.words.append(sentence)
                self.tokens[sentence] = len(self)
            tokens.append(self.tokens[sentence])
            return tokens
        words = sentence.lower().replace(',', '').replace('?', '').replace('\'s', '').split()
        for word in words:
            if word not in self.tokens:
                self.words.append(word)
                self.tokens[word] = len(self)
            tokens.append(self.tokens[word])
        return tokens

    def save(self, path):
        torch.save((self.tokens, self.words), path)

    def load(self, path):
        self.tokens, self.words = torch.load(path)


class VQADataset(data.Dataset):
    def __init__(self, path, q_dict, a_dict, validation=False, image_dim=64, max_question_length=14):
        super(VQADataset, self).__init__()
        self.entries = []
        self.q_dict = q_dict
        self.a_dict = a_dict
        self.img_dict = {}
        self.image_dim = image_dim
        self.process_data(path, validation, max_question_length)

    def process_data(self, path, validation, max_question_length):
        name = 'val' if validation else 'train'
        images = path / f'{name}2014'
        questions = json.load((path / f'v2_OpenEnded_mscoco_{name}2014_questions.json').open('r'))['questions']
        annotations = json.load((path / f'v2_mscoco_{name}2014_annotations.json').open('r'))['annotations']
        questions = sorted(questions, key=lambda q: q['question_id'])
        annotations = sorted(annotations, key=lambda a: a['question_id'])
        img_size = (self.image_dim, self.image_dim)
        for i, (question, annotation) in enumerate(zip(questions, annotations)):
            if i % 100 == 0:
                print(f'{i} items loaded')
            assert question['question_id'] == annotation['question_id']
            assert question['image_id'] == annotation['image_id']
            image_id = question['image_id']
            if image_id not in self.img_dict:
                img = Image.open(images / f'COCO_{name}2014_{str(image_id).zfill(12)}.jpg')
                self.img_dict[image_id] = img.resize(img_size).convert('RGB')
            q_tokens = self.q_dict.tokenize(question['question'])[:max_question_length]
            a_token = self.a_dict.tokenize(annotation['multiple_choice_answer'], as_sentence=True)
            if len(q_tokens) < max_question_length:
                q_tokens = [self.q_dict.pad_token] * (max_question_length - len(q_tokens)) + q_tokens
            self.entries.append((image_id, q_tokens, a_token))

    def __getitem__(self, item):
        img_id, q_tokens, a_token = self.entries[item]
        return self.img_dict[img_id], q_tokens, a_token

    def __len__(self):
        return len(self.entries)

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        return torch.load(path)
