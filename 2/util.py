import re
import pickle
import json
import numpy
import h5py
from collections import Counter
from pathlib import Path
from PIL import Image
from data import Vocabulary
from time import time

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've"
}

numbers = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
           'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile('(?!<=\d)(\.)(?!\d)')
comma_strip = re.compile('(\d)(\,)(\d)')
punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']


def preprocess(text):
    for p in punct:
        text = text.replace(p, '')
    text = ' '.join(text.split())
    text = period_strip.sub('', text, re.UNICODE)
    text = ' '.join(numbers.get(contractions.get(w, w), w) for w in text.split() if w not in articles)
    return text.replace(',', '')


def make_cache(data_path, cache_path, min_annotation_occurrences, image_dim):
    q_dict = Vocabulary()
    a_dict = Vocabulary()
    cache_time = time()
    for name in ('train', 'val'):
        question_time = time()
        print(f'cache {name}')
        images = data_path / f'{name}2014'
        questions = json.load((data_path / f'v2_OpenEnded_mscoco_{name}2014_questions.json').open('r'))['questions']
        annotations = json.load((data_path / f'v2_mscoco_{name}2014_annotations.json').open('r'))['annotations']
        annotation_counts = Counter()
        for question in questions:
            q_dict.tokenize(question['question'], insert=True)
        for annotation in annotations:
            annotation_counts[preprocess(annotation['multiple_choice_answer'])] += 1
        for annotation, count in annotation_counts.items():
            if count < min_annotation_occurrences:
                continue
            a_dict.tokenize(annotation, split=False, insert=True)
        print(f'{len(questions)} questions and annotations cached in {time() - question_time:.2f}s')
        if Path(cache_path / f'{name}_img.hdf5').is_file() and Path(cache_path / f'{name}_imgmap.pkl').is_file():
            continue
        img_cache_time = time()
        img_size = (image_dim, image_dim)
        n_images = len(list(images.glob('*')))
        img_dict = {}
        with h5py.File(cache_path / f'{name}_img.hdf5', 'w') as h5:
            img_data = h5.create_dataset('images', shape=(n_images, 3, image_dim, image_dim), dtype='i')
            for i, image in enumerate(images.glob('*')):
                if i % 10000 == 0:
                    print(f'{i} images cached')
                img_id = int(image.name.replace('.jpg', '')[-12:])
                img_dict[img_id] = i
                img = numpy.array(Image.open(image).resize(img_size).convert('RGB'))
                img_data[i, :] = img.reshape((3, image_dim, image_dim))
        pickle.dump(img_dict, Path(cache_path / f'{name}_imgmap.pkl').open('wb'))
        print(f'{n_images} images cached in {time() - img_cache_time:.2f}s')
    q_dict.save(cache_path / 'q_vocab.pkl')
    a_dict.save(cache_path / 'a_vocab.pkl')
    print(f'data cached in {time() - cache_time:.2f}s')
