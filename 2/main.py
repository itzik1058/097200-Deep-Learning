from util import make_cache
from train import train
from pathlib import Path


data_path = Path('/datashare')
cache_path = Path('')


def main():
    cache_path.mkdir(exist_ok=True)
    cache = ['vocab.pkl', 'train_img.hdf5', 'train_imgmap.pkl', 'val_img.hdf5', 'val_imgmap.pkl']
    if not all((cache_path / path).is_file() for path in cache):
        make_cache(data_path, cache_path, image_dim=64)
    train(data_path, cache_path)


if __name__ == '__main__':
    main()
