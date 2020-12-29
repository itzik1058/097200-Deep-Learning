import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from data import VQADataset
from model import make_model
from time import time
import matplotlib.pyplot as plt


def train(data_path, cache_path):
    t = time()
    train_dataset = VQADataset(data_path, cache_path)
    print(f'Train dataset loaded in {time() - t:.2f}s with {len(train_dataset)} entries')
    t = time()
    val_dataset = VQADataset(data_path, cache_path, validation=True)
    print(f'Validation dataset loaded in {time() - t:.2f}s with {len(val_dataset)} entries')
    train_loader = data.DataLoader(train_dataset, batch_size=100, shuffle=True, collate_fn=train_dataset.collate)
    val_loader = data.DataLoader(val_dataset, batch_size=500, shuffle=True, collate_fn=val_dataset.collate)
    vqa_model = make_model(train_dataset.vocab, train_dataset.ans2label, 2048).cuda()
    optimizer = optim.Adam(vqa_model.parameters(), lr=4e-3)
    criterion = nn.BCEWithLogitsLoss()
    print('train')
    vqa_model.train()
    tr_losses, tr_scores, val_losses, val_scores = [], [], [], []
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
        tr_losses.append(epoch_loss)
        tr_scores.append(epoch_score)
        val_losses, val_scores = 0, 0
        with torch.no_grad():
            for batch, (image, question, annotation) in enumerate(val_loader):
                annotation = annotation.cuda()
                result = vqa_model(image.cuda(), question.cuda())
                loss = criterion(result, annotation)
                result_score = nn.functional.one_hot(result.argmax(dim=1), num_classes=vqa_model.num_classes)
                batch_score = torch.sum(result_score * annotation).item() / question.size(0)
                val_losses += loss.item() * question.size(0)
                val_scores += batch_score * question.size(0)
        val_losses /= len(val_dataset)
        val_scores /= len(val_dataset)
        print(f'Epoch {epoch} Train Loss {epoch_loss:.4f} Train Score {epoch_score:.3f} '
              f'Validation Loss {val_losses:.4f} Validation Score {val_scores:.3f} done in {time() - epoch_time:.2f}s')
    plt.subplot(2, 1, 1)
    plt.plot(tr_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.title('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(tr_scores, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.title('Score')
    plt.show()
    torch.save(vqa_model, 'model.pkl')
