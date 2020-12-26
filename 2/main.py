from data import Vocabulary, VQADataset
from preprocess import preprocess
from train import train
from pathlib import Path
import pickle
import json
import numpy
import h5py
from PIL import Image


data_path = Path('data')
cache_path = Path('cache')


def make_cache(image_dim=64):
    q_dict = Vocabulary()
    a_dict = Vocabulary()
    for name in ('train', 'val'):
        print(f'cache {name}')
        images = data_path / f'{name}2014'
        questions = json.load((data_path / f'v2_OpenEnded_mscoco_{name}2014_questions.json').open('r'))['questions']
        annotations = json.load((data_path / f'v2_mscoco_{name}2014_annotations.json').open('r'))['annotations']
        for question in questions:
            q_dict.tokenize(question['question'])
        for annotation in annotations:
            a_dict.tokenize(preprocess(annotation['multiple_choice_answer']), as_sentence=True)
        img_size = (image_dim, image_dim)
        n_images = len(list(images.glob('*')))
        print(f'{len(questions)} questions, {n_images} images')
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
    q_dict.save(cache_path / 'q_vocab.pkl')
    a_dict.save(cache_path / 'a_vocab.pkl')


def evaluate_hw2():
    q_dict, a_dict = Vocabulary(), Vocabulary()
    q_dict.load(cache_path / 'q_vocab.pkl')
    a_dict.load(cache_path / 'a_vocab.pkl')
    val_img_map = pickle.load((cache_path / 'val_imgmap.pkl').open('rb'))
    val_img_data = h5py.File(cache_path / 'val_img.hdf5', 'r')['images']
    val_dataset = VQADataset(data_path, q_dict, a_dict, val_img_map, val_img_data, validation=True)


def main():
    cache_path.mkdir(exist_ok=True)
    cache = ['q_vocab.pkl', 'a_vocab.pkl', 'train_img.hdf5', 'train_imgmap.pkl', 'val_img.hdf5', 'val_imgmap.pkl']
    if not all((cache_path / path).is_file() for path in cache):
        make_cache()
    q_dict, a_dict = Vocabulary(), Vocabulary()
    q_dict.load(cache_path / 'q_vocab.pkl')
    a_dict.load(cache_path / 'a_vocab.pkl')
    tr_img_dict = pickle.load((cache_path / 'train_imgmap.pkl').open('rb'))
    tr_img_data = numpy.array(h5py.File(cache_path / 'train_img.hdf5', 'r')['images'])
    train_dataset = VQADataset(data_path, q_dict, a_dict, tr_img_dict, tr_img_data)
    train(train_dataset)


if __name__ == '__main__':
    main()
