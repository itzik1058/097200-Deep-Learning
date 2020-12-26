import torch
import torch.utils.data as data
import json


class Vocabulary:
    def __init__(self):
        self.words = [None]
        self.tokens = {None: 0}

    def __len__(self):
        return len(self.tokens)

    @property
    def pad_token(self):
        return 0

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
    def __init__(self, path, q_dict, a_dict, img_dict, img_data, validation=False):
        super(VQADataset, self).__init__()
        self.entries = []
        self.q_dict = q_dict
        self.a_dict = a_dict
        self.img_dict = img_dict
        self.img_data = img_data
        self.process_data(path, validation)

    def process_data(self, path, validation):
        name = 'val' if validation else 'train'
        questions = json.load((path / f'v2_OpenEnded_mscoco_{name}2014_questions.json').open('r'))['questions']
        annotations = json.load((path / f'v2_mscoco_{name}2014_annotations.json').open('r'))['annotations']
        questions = sorted(questions, key=lambda q: q['question_id'])
        annotations = sorted(annotations, key=lambda a: a['question_id'])
        for question, annotation in zip(questions, annotations):
            assert question['question_id'] == annotation['question_id']
            assert question['image_id'] == annotation['image_id']
            image_id = question['image_id']
            q_tokens = self.q_dict.tokenize(question['question'])
            a_token = self.a_dict.tokenize(annotation['multiple_choice_answer'], as_sentence=True)
            self.entries.append((self.img_dict[image_id], q_tokens, a_token))

    def __getitem__(self, item):
        img_idx, q_tokens, a_token = self.entries[item]
        return self.img_data[img_idx], q_tokens, a_token

    def __len__(self):
        return len(self.entries)
