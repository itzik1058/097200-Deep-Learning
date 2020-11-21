import torch
import torch.utils.data
import torchvision
import time


device = 'cuda'


def leaky_relu(x: torch.Tensor):
    return x.clamp(min=0) + 0.01 * x.clamp(max=0)


def leaky_relu_gradient(grad: torch.Tensor, x: torch.Tensor):
    grad = grad.clone()
    grad[x < 0] *= 0.01
    return grad


class MNISTClassifier:
    def __init__(self, layers):
        self.num_layers = len(layers) + 1
        self.weights = []
        self.bias = []
        for layer_in, layer_out in layers:
            self.weights.append(torch.randn(size=(layer_in, layer_out), device=device) / (layer_out ** 0.5))
            self.bias.append(torch.zeros(size=(layer_out,), device=device))
        self.weights_grad = [torch.zeros_like(weights) for weights in self.weights]
        self.bias_grad = [torch.zeros_like(biases) for biases in self.bias]
        self.num_steps = 0
        self.layer_out = []
        self.layer_activations = []
        self.loss_grad = None

    def forward(self, x):
        self.layer_out = []
        self.layer_activations = []
        self.loss_grad = None
        out = x.view(-1, self.weights[0].shape[0])
        self.layer_activations.append(out)
        for i, (weights, bias) in enumerate(zip(self.weights, self.bias)):
            out = out.mm(weights) + bias
            self.layer_out.append(out)
            out = leaky_relu(out)
            self.layer_activations.append(out)
        return out

    def loss(self, x, y):
        out = self.forward(x)
        y = y.view(-1)
        y_prob = torch.zeros(size=(y.shape[0], self.weights[-1].shape[1]), device=y.device)
        y_prob[torch.arange(y.shape[0]), y] = 1
        error = out - y_prob
        self.loss_grad = 2 * error
        return error.pow(2).sum()

    def backward(self):
        bias_grad = [torch.empty_like(biases) for biases in self.bias]
        weights_grad = [torch.empty_like(weights) for weights in self.weights]
        delta = leaky_relu_gradient(self.loss_grad, self.layer_out[-1])
        bias_grad[-1] = delta.mean(dim=0)
        weights_grad[-1] = torch.matmul(self.layer_activations[-2].unsqueeze(dim=2), delta.unsqueeze(dim=1)).mean(dim=0)
        for layer in range(2, self.num_layers):
            delta = leaky_relu_gradient(delta.mm(self.weights[-layer + 1].t()), self.layer_out[-layer])
            bias_grad[-layer] = delta.mean(dim=0)
            weights_grad[-layer] = torch.matmul(self.layer_activations[-layer - 1].unsqueeze(dim=2),
                                                delta.unsqueeze(dim=1)).mean(dim=0)
        self.weights_grad = [grad + weights_grad[i] for i, grad in enumerate(self.weights_grad)]
        self.bias_grad = [grad + bias_grad[i] for i, grad in enumerate(self.bias_grad)]
        self.num_steps += 1

    def step(self, lr=0.01):
        self.weights = [w - (lr / self.num_steps) * grad for w, grad in zip(self.weights, self.weights_grad)]
        self.bias = [b - (lr / self.num_steps) * grad for b, grad in zip(self.bias, self.bias_grad)]
        self.weights_grad = [torch.zeros_like(weights) for weights in self.weights]
        self.bias_grad = [torch.zeros_like(biases) for biases in self.bias]
        self.num_steps = 0


def main():
    mnist_train = torchvision.datasets.MNIST('data/', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST('data/', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1000, shuffle=True)
    classifier = MNISTClassifier(layers=[[784, 1296], [1296, 1296], [1296, 10]])
    lr = 1e-2
    train_loss = []
    test_loss = []
    test_accuracy = 0
    for epoch in range(10):
        start = time.time()
        epoch_loss = []
        for image, label in train_loader:
            loss = classifier.loss(image.to(device), label.to(device)).item()
            epoch_loss.append(loss)
            train_loss.append(loss)
            classifier.backward()
            classifier.step(lr=lr)
        print(f'Epoch {epoch + 1} ({time.time() - start:.2f}s) loss {sum(epoch_loss) / len(epoch_loss):.2f}')
    for image, label in test_loader:
        pred = classifier.forward(image.to(device)).argmax(dim=1)
        test_accuracy += (pred == label.to(device)).sum().item()
        test_loss.append(classifier.loss(image.to(device), label.to(device)).item())
    test_accuracy /= len(test_loader.dataset)
    print('Test loss', sum(test_loss) / len(test_loss))
    print('Test accuracy', test_accuracy)


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
