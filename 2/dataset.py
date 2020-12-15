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

    def tokenize(self, sentence):
        words = sentence.lower().replace(',', '').replace('?', '').replace('\'s', '').split()
        tokens = []
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
    def __init__(self, path, dictionary, validation=False, max_question_length=14):
        super(VQADataset, self).__init__()
        self.entries = []
        self.dictionary = dictionary
        self.load(path, validation, max_question_length)

    def load(self, path, validation, max_question_length):
        name = 'val' if validation else 'train'
        images = path / f'{name}2014'
        questions = json.load((path / f'v2_OpenEnded_mscoco_{name}2014_questions.json').open('r'))['questions']
        annotations = json.load((path / f'v2_mscoco_{name}2014_annotations.json').open('r'))['annotations']
        questions = sorted(questions, key=lambda q: q['question_id'])
        annotations = sorted(annotations, key=lambda a: a['question_id'])
        for question, annotation in zip(questions, annotations):
            assert question['question_id'] == annotation['question_id']
            assert question['image_id'] == annotation['image_id']
            image_id = question['image_id']
            image = Image.open(images / f'COCO_{name}2014_{str(image_id).zfill(12)}.jpg')
            q_tokens = self.dictionary.tokenize(question['question'])[:max_question_length]
            a_tokens = self.dictionary.tokenize(annotation['multiple_choice_answer'])
            if len(q_tokens) < max_question_length:
                q_tokens = [self.dictionary.pad_token] * (max_question_length - len(q_tokens)) + q_tokens
            self.entries.append((image, q_tokens, a_tokens))

    def __getitem__(self, item):
        return self.entries[item]

    def __len__(self):
        return len(self.entries)
