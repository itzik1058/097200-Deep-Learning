import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import gan
import time
import matplotlib.pyplot as plt


def train(generator, discriminator, train_loader, n_epoch, **kwargs):
    generator.cuda()
    discriminator.cuda()
    generator.train()
    discriminator.train()
    gen_optimizer = torch.optim.Adam(generator.parameters(), **kwargs)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), **kwargs)
    criterion = torch.nn.BCELoss()
    generator_train_loss, discriminator_train_loss = [], []
    example = torch.randn(64, generator.latent_size).cuda()
    try:
        for epoch in range(n_epoch):
            epoch_time = time.time()
            generator_train_loss.append(0)
            discriminator_train_loss.append(0)
            for image, _ in train_loader:
                batch_size = image.size(0)
                image = image.cuda()
                real_label = torch.ones(batch_size).cuda()
                fake_label = torch.zeros(batch_size).cuda()

                disc_optimizer.zero_grad()
                real_loss = criterion(discriminator(image), real_label)
                real_loss.backward()
                latent = torch.randn(batch_size, generator.latent_size).cuda()
                generated = generator(latent)
                fake_loss = criterion(discriminator(generated.detach()), fake_label)
                fake_loss.backward()
                disc_optimizer.step()

                gen_optimizer.zero_grad()
                gen_loss = criterion(discriminator(generated), real_label)
                gen_loss.backward()
                gen_optimizer.step()

                generator_train_loss[-1] += gen_loss.item()
                discriminator_train_loss[-1] += real_loss.item() + fake_loss.item()
            generator_train_loss[-1] /= train_loader.batch_size
            discriminator_train_loss[-1] /= train_loader.batch_size
            if epoch % 1 == 0:
                with torch.no_grad():
                    generated = generator(example).detach().cpu()
                grid = torchvision.utils.make_grid(generated, padding=2, normalize=True)
                plt.imshow(grid.numpy().transpose(1, 2, 0))
                plt.show()
            print(f'Epoch {epoch}/{n_epoch} done in {time.time() - epoch_time:.2f}s with loss '
                  f'Generator({generator_train_loss[-1]:.3f}) Discriminator({discriminator_train_loss[-1]:.3f})')
    except KeyboardInterrupt:
        pass
    torch.save(generator.state_dict(), 'generator.pkl')
    plt.plot(generator_train_loss, label='Generator')
    plt.plot(discriminator_train_loss, label='Discriminator')
    plt.xticks(range(n_epoch))
    plt.xlabel('Epoch')
    plt.title('BCE Loss')
    plt.legend()
    plt.show()


def main():
    transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CelebA('', split='train', transform=transform, download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    generator, discriminator = gan.make_model(channels=3, latent_size=100, hidden_dim=64)
    train(generator, discriminator, train_loader, n_epoch=10, lr=0.0002, betas=(0.5, 0.999))


if __name__ == '__main__':
    main()
