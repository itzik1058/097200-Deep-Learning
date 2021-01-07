import torch
import torch.nn as nn
import torch.utils.data as data
from util import make_cache
from data import VQADataset
from pathlib import Path


@torch.no_grad()
def evaluate_hw2():
    data_path = Path('/datashare')
    cache_path = Path('')
    cache = ['vocab.pkl', 'val_img.hdf5', 'val_imgmap.pkl']
    if not all((cache_path / path).is_file() for path in cache):
        make_cache(data_path, cache_path, image_dim=64, validation_only=True, verbose=False)
    val_dataset = VQADataset(data_path, cache_path, validation=True)
    val_loader = data.DataLoader(val_dataset, batch_size=100, shuffle=True, collate_fn=val_dataset.collate)
    vqa_model = torch.load('model.pkl')
    vqa_model.eval()
    score = 0
    for batch, (image, question, annotation) in enumerate(val_loader):
        result = vqa_model(image.cuda(), question.cuda())
        result_score = nn.functional.one_hot(result.argmax(dim=1), num_classes=vqa_model.num_classes)
        score += torch.sum(result_score * annotation.cuda()).item()
    score /= len(val_dataset)
    return score


if __name__ == '__main__':
    print(evaluate_hw2())
