import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from data import VQADataset
from model import make_model


def collate(batch):
    images, questions, question_lengths, annotations = [], [], [], []
    for image, question, annotation in batch:
        images.append(torch.tensor(image, dtype=torch.float))
        questions.append(torch.tensor(question))
        annotations.append(torch.tensor(annotation))
        question_lengths.append(len(question))
    images = torch.stack(images)
    questions = torch.nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=0)
    question_lengths = torch.tensor(question_lengths, requires_grad=False)
    annotations = torch.stack(annotations)
    question_lengths, indices = torch.sort(question_lengths, descending=True)
    images = images[indices, :]
    questions = questions[indices, :]
    annotations = annotations[indices, :].view(-1)
    return images, questions, question_lengths, annotations


def train(train_dataset: VQADataset):
    train_loader = data.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)
    vqa_model = make_model(train_dataset.q_dict, train_dataset.a_dict, 1024).cuda()
    optimizer = optim.Adagrad(vqa_model.parameters(), lr=4e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    vqa_model.train()
    print('train')
    for epoch in range(1000):
        epoch_loss = 0
        epoch_accuracy = 0
        for image, question, question_length, annotation in train_loader:
            result = vqa_model(image.cuda(), question.cuda(), question_length.cuda())
            loss = criterion(result, annotation.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vqa_model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_accuracy += torch.sum(torch.argmax(result, dim=1).cpu() == annotation).item()
            # print(loss.item(), torch.mean((torch.argmax(result, dim=1).cpu() == annotation).float()).item())
        scheduler.step()
        epoch_loss /= len(train_dataset)
        epoch_accuracy /= len(train_dataset)
        print(f'{epoch_loss:.3f}\t{epoch_accuracy:.3f}')
    torch.save(vqa_model, 'model.pkl')
