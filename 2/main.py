from data import VQADataset
from util import make_cache
from train import train
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
from time import time


data_path = Path('data/')
cache_path = Path('data/cache')


@torch.no_grad()
def evaluate_hw2():
    val_dataset = VQADataset(data_path, cache_path, validation=True)
    val_loader = data.DataLoader(val_dataset, batch_size=100, shuffle=True, collate_fn=val_dataset.collate)
    vqa_model = torch.load('model.pkl')
    criterion = nn.BCEWithLogitsLoss()
    val_loss = 0
    val_score = 0
    for batch, (image, question, annotation) in enumerate(val_loader):
        annotation = annotation.cuda()
        result = vqa_model(image.cuda(), question.cuda())
        loss = criterion(result, annotation)
        result_score = nn.functional.one_hot(result.argmax(dim=1), num_classes=vqa_model.num_classes)
        batch_score = torch.sum(result_score * annotation).item() / question.size(0)
        val_loss += loss.item() * question.size(0)
        val_score += batch_score * question.size(0)
    val_loss /= len(val_dataset)
    val_score /= len(val_dataset)
    return val_score


def main():
    cache_path.mkdir(exist_ok=True)
    cache = ['vocab.pkl', 'train_img.hdf5', 'train_imgmap.pkl', 'val_img.hdf5', 'val_imgmap.pkl']
    if not all((cache_path / path).is_file() for path in cache):
        make_cache(data_path, cache_path, min_annotation_occurrences=1, image_dim=64)
    train(data_path, cache_path)
    evaluate_hw2()


if __name__ == '__main__':
    main()
