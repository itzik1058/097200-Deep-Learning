from train import *


def evaluate_hw1():
    model = torch.load('model.pkl')
    mnist_test = MNIST('', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=True)
    test_accuracy = 0
    for image, label in test_loader:
        pred: torch.Tensor = model.forward(image.to(device)).argmax(dim=1)
        test_accuracy += (pred == label.to(device)).sum().item()
    test_accuracy /= len(test_loader.dataset)
    return test_accuracy


if __name__ == '__main__':
    print('Test accuracy', evaluate_hw1())
