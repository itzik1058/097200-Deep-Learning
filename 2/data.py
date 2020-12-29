import torch
import torch.utils.data as data
import json
import pickle
import h5py


class Vocabulary:
    def __init__(self):
        self.words = [None]
        self.tokens = {None: 0}

    def __len__(self):
        return len(self.tokens)

    @property
    def pad_token(self):
        return 0

    def tokenize(self, sentence, insert=False):
        tokens = []
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
    def __init__(self, path, cache_path, validation=False, max_question_length=None):
        super(VQADataset, self).__init__()
        self.entries = []
        self.vocab = None
        self.img_dict = None
        self.img_data = None
        self.ans2label = None
        self.process_data(path, cache_path, validation, max_question_length)

    def process_data(self, path, cache_path, validation, max_question_length):
        name = 'val' if validation else 'train'
        self.vocab = Vocabulary()
        self.vocab.load(cache_path / 'vocab.pkl')
        self.img_dict = pickle.load((cache_path / f'{name}_imgmap.pkl').open('rb'))
        self.img_data = h5py.File(cache_path / f'{name}_img.hdf5', 'r')['images']
        target = pickle.load((cache_path / f'{name}_target.pkl').open('rb'))
        self.ans2label = pickle.load((cache_path / f'trainval_ans2label.pkl').open('rb'))
        questions = json.load((path / f'v2_OpenEnded_mscoco_{name}2014_questions.json').open('r'))['questions']
        # annotations = json.load((path / f'v2_mscoco_{name}2014_annotations.json').open('r'))['annotations']
        questions = {q['question_id']: q for q in questions}
        # annotations = {a['annotatin_id']: a for a in annotations}
        for entry in target:
            image_id = entry['image_id']
            # if len(self.entries) > 50:  # TODO remove this
            #     continue
            q_tokens = self.vocab.tokenize(questions[entry['question_id']]['question'])
            if max_question_length:
                q_tokens = q_tokens[:max_question_length]
                pad = max_question_length - len(q_tokens)
                q_tokens = [self.vocab.pad_token] * pad + q_tokens
            # answer_count = Counter()
            # for answer in annotation['answers']:
            #     a_tokens = self.a_vocab.tokenize(answer['answer'], split=False)
            #     if a_tokens is None:
            #         continue
            #     a_token = a_tokens[0]
            #     answer_count[a_token] += 1
            # answers, scores = [], []
            # for answer, count in answer_count.items():
            #     answers.append(answer)
            #     scores.append(min(1, count * 0.3))
            self.entries.append((self.img_dict[image_id], q_tokens, entry['labels'], entry['scores']))

    @staticmethod
    def collate(batch):
        images, questions, annotations = [], [], []
        for image, question, annotation in batch:
            images.append(torch.tensor(image, dtype=torch.float))
            questions.append(torch.tensor(question))
            annotations.append(annotation)
        images = torch.stack(images)
        questions = torch.nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=0)
        annotations = torch.stack(annotations)
        return images, questions, annotations

    def __getitem__(self, item):
        img_idx, q_tokens, answers, scores = self.entries[item]
        annotation = torch.zeros(len(self.ans2label))
        for answer, score in zip(answers, scores):
            annotation[answer] = score
        return self.img_data[img_idx], q_tokens, annotation

    def __len__(self):
        return len(self.entries)
