import torch
import torch.utils.data as data
import json
from collections import Counter


class Vocabulary:
    def __init__(self):
        self.words = [None]
        self.tokens = {None: 0}

    def __len__(self):
        return len(self.tokens)

    @property
    def pad_token(self):
        return 0

    def tokenize(self, sentence, split=True, insert=False):
        tokens = []
        if not split:
            if sentence not in self.tokens:
                if not insert:
                    return None
                self.words.append(sentence)
                self.tokens[sentence] = len(self)
            tokens.append(self.tokens[sentence])
            return tokens
        words = sentence.lower().replace(',', '').replace('?', '').replace('\'s', '').split()
        for word in words:
            if word not in self.tokens:
                if not insert:
                    return None
                self.words.append(word)
                self.tokens[word] = len(self)
            tokens.append(self.tokens[word])
        return tokens

    def save(self, path):
        torch.save((self.tokens, self.words), path)

    def load(self, path):
        self.tokens, self.words = torch.load(path)


class VQADataset(data.Dataset):
    def __init__(self, path, q_vocab, a_vocab, img_dict, img_data, validation=False):
        super(VQADataset, self).__init__()
        self.entries = []
        self.q_vocab = q_vocab
        self.a_vocab = a_vocab
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
            q_tokens = self.q_vocab.tokenize(question['question'])
            answer_count = Counter()
            for answer in annotation['answers']:
                a_tokens = self.a_vocab.tokenize(answer['answer'], split=False)
                if a_tokens is None:
                    continue
                a_token = a_tokens[0]
                answer_count[a_token] += 1
            answers, scores = [], []
            for answer, count in answer_count.items():
                answers.append(answer)
                scores.append(min(1, count * 0.3))
            self.entries.append((self.img_dict[image_id], q_tokens, answers, scores))

    @staticmethod
    def collate(batch):
        images, questions, question_lengths, annotations = [], [], [], []
        for image, question, annotation in batch:
            images.append(torch.tensor(image, dtype=torch.float))
            questions.append(torch.tensor(question))
            annotations.append(annotation)
            question_lengths.append(len(question))
        images = torch.stack(images)
        questions = torch.nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=0)
        question_lengths = torch.tensor(question_lengths, requires_grad=False)
        annotations = torch.stack(annotations)
        question_lengths, indices = torch.sort(question_lengths, descending=True)
        images = images[indices, :]
        questions = questions[indices, :]
        annotations = annotations[indices, :]
        return images, questions, question_lengths, annotations

    def __getitem__(self, item):
        img_idx, q_tokens, answers, scores = self.entries[item]
        annotation = torch.zeros(len(self.a_vocab))
        for answer, score in zip(answers, scores):
            annotation[answer] = score
        return self.img_data[img_idx], q_tokens, annotation

    def __len__(self):
        return len(self.entries)
