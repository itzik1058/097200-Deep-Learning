import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from data import VQADataset
from model import make_model
from time import time


def train(train_dataset: VQADataset):
    train_loader = data.DataLoader(train_dataset, batch_size=100, shuffle=True, collate_fn=train_dataset.collate)
    vqa_model = make_model(train_dataset.vocab, train_dataset.ans2label, 2048).cuda()
    optimizer = optim.Adam(vqa_model.parameters(), lr=4e-3)
    criterion = nn.BCEWithLogitsLoss()
    print('train')
    vqa_model.train()
    for epoch in range(30):
        epoch_loss = 0
        epoch_score = 0
        epoch_time = time()
        for batch, (image, question, annotation) in enumerate(train_loader):
            batch_time = time()
            annotation = annotation.cuda()
            result = vqa_model(image.cuda(), question.cuda())
            loss = criterion(result, annotation)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vqa_model.parameters(), max_norm=0.25)
            optimizer.step()
            optimizer.zero_grad()
            result_score = nn.functional.one_hot(result.argmax(dim=1), num_classes=vqa_model.num_classes)
            batch_score = torch.sum(result_score * annotation).item() / question.size(0)
            # print(f'Epoch {epoch} Batch {batch} Loss {loss.item():.3f} Score {batch_score:.3f} '
            #       f'done in {time() - batch_time:.2f}s')
            epoch_loss += loss.item() * question.size(0)
            epoch_score += batch_score * question.size(0)
        epoch_loss /= len(train_dataset)
        epoch_score /= len(train_dataset)
        print(f'Epoch {epoch} Loss {epoch_loss:.3f} Score {epoch_score:.3f} done in {time() - epoch_time:.2f}s')
    torch.save(vqa_model, 'model.pkl')
