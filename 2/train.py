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
    vqa_model = make_model(len(train_dataset.vocab), len(train_dataset.ans2label), train_dataset.question_length, 4096)
    vqa_model = vqa_model.cuda()
    optimizer = optim.Adam(vqa_model.parameters(), lr=4e-3)
    criterion = nn.BCEWithLogitsLoss()
    print('train')
    tr_losses, tr_scores, val_losses, val_scores = [], [], [], []
    for epoch in range(25):
        epoch_time = time()
        epoch_loss, epoch_score = 0, 0
        vqa_model.train()
        for batch, (image, question, annotation) in enumerate(train_loader):
            annotation = annotation.cuda()
            result = vqa_model(image.cuda(), question.cuda())
            loss = criterion(result, annotation)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vqa_model.parameters(), max_norm=0.25)
            optimizer.step()
            optimizer.zero_grad()
            result_score = nn.functional.one_hot(result.argmax(dim=1), num_classes=vqa_model.num_classes)
            batch_score = torch.sum(result_score * annotation).item() / question.size(0)
            epoch_loss += loss.item() * question.size(0)
            epoch_score += batch_score * question.size(0)
        epoch_loss /= len(train_dataset)
        epoch_score /= len(train_dataset)
        tr_losses.append(epoch_loss)
        tr_scores.append(epoch_score)
        val_loss, val_score = 0, 0
        vqa_model.eval()
        with torch.no_grad():
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
        val_losses.append(val_loss)
        val_scores.append(val_score)
        print(f'Epoch {epoch} Train Loss {epoch_loss:.4f} Train Score {epoch_score:.3f} '
              f'Validation Loss {val_loss:.4f} Validation Score {val_score:.3f} done in {time() - epoch_time:.2f}s')
    torch.save(vqa_model, 'model.pkl')
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(tr_losses, label='Train')
    ax[0].plot(val_losses, label='Validation')
    ax[0].set_xlabel('Epoch')
    ax[0].set_title('Loss')
    ax[1].plot(tr_scores, label='Train')
    ax[1].plot(val_scores, label='Validation')
    ax[1].set_xlabel('Epoch')
    ax[1].set_title('Score')
    plt.legend()
    plt.show()
