import pickle
import json
import numpy
import h5py
from pathlib import Path
from PIL import Image
from data import Vocabulary
from time import time


def make_cache(data_path, cache_path, image_dim, validation_only=False):
    q_dict = Vocabulary()
    cache_time = time()
    for name in ('train', 'val'):
        if name != 'val' and validation_only:
            continue
        question_time = time()
        print(f'cache {name}')
        images = data_path / f'{name}2014'
        questions = json.load((data_path / f'v2_OpenEnded_mscoco_{name}2014_questions.json').open('r'))['questions']
        for question in questions:
            q_dict.tokenize(question['question'], insert=True)
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
    q_dict.save(cache_path / 'vocab.pkl')
    print(f'data cached in {time() - cache_time:.2f}s')
