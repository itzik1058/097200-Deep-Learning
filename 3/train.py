import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import gan
import time
import matplotlib.pyplot as plt
import imageio
import skimage
import glob


def train():
    n_epoch, lr, betas = 50, 0.0002, (0.5, 0.999)
    transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64),
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CelebA('', split='train', transform=transform, download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    generator = gan.Generator(channels=3, latent_size=128, num_classes=2, hidden_dim=64).cuda()
    discriminator = gan.Discriminator(channels=3, hidden_dim=64).cuda()
    generator.apply(gan.weights_init_normal)
    discriminator.apply(gan.weights_init_normal)
    generator.train()
    discriminator.train()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    criterion = torch.nn.BCELoss()
    generator_train_loss, discriminator_train_loss = [], []
    example = torch.randn(64, generator.latent_size).cuda()
    example_attr = torch.randint(2, size=(64, 2)).float().cuda()
    try:
        for epoch in range(n_epoch):
            epoch_time = time.time()
            generator_train_loss.append(0)
            discriminator_train_loss.append(0)
            for image, attr in train_loader:
                batch_size = image.size(0)
                image = image.cuda()
                attr = attr[:, [15, 20]].float().cuda()
                real_label = torch.ones(batch_size).cuda()
                fake_label = torch.zeros(batch_size).cuda()

                latent = torch.randn(batch_size, generator.latent_size).cuda()
                generated = generator(latent, attr)

                disc_optimizer.zero_grad()
                real_loss = criterion(discriminator(image), real_label)
                fake_loss = criterion(discriminator(generated.detach()), fake_label)
                disc_loss = real_loss + fake_loss
                disc_loss.backward()
                disc_optimizer.step()

                gen_optimizer.zero_grad()
                gen_loss = criterion(discriminator(generated), real_label)
                gen_loss.backward()
                gen_optimizer.step()

                generator_train_loss[-1] += gen_loss.item()
                discriminator_train_loss[-1] += disc_loss.item()
            generator_train_loss[-1] /= train_loader.batch_size
            discriminator_train_loss[-1] /= train_loader.batch_size
            if epoch % 1 == 0:
                with torch.no_grad():
                    generated = generator(example, example_attr).detach().cpu()
                grid = torchvision.utils.make_grid(generated, padding=2, normalize=True)
                plt.imshow(grid.numpy().transpose(1, 2, 0))
                plt.axis('off')
                plt.grid()
                plt.savefig(f'train_img/{epoch}.png')
                plt.show()
            print(f'Epoch {epoch}/{n_epoch} done in {time.time() - epoch_time:.2f}s with loss '
                  f'Generator({generator_train_loss[-1]:.3f}) Discriminator({discriminator_train_loss[-1]:.3f})')
            torch.save(generator.state_dict(), f'model/generator_{epoch}.pkl')
    except KeyboardInterrupt:
        pass
    torch.save(generator.state_dict(), f'generator.pkl')
    plt.plot(generator_train_loss, label='Generator')
    plt.plot(discriminator_train_loss, label='Discriminator')
    plt.xlabel('Epoch')
    plt.title('BCE Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


def interpolate(path, generator, latent, attr, steps=60):
    images = []
    for k in range(len(latent) - 1):
        for i in range(steps + 1):
            t = i / steps
            lat = (1 - t) * latent[k] + t * latent[k + 1]
            a = (1 - t) * attr[k] + t * attr[k + 1]
            with torch.no_grad():
                generated = generator(lat, a).detach().cpu()
            images.append(torchvision.utils.make_grid(generated, padding=2, normalize=True).numpy().transpose(1, 2, 0))
    images = [skimage.img_as_ubyte(img) for img in images]
    imageio.mimsave(path, images + images[::-1], fps=30)


def make_results():
    generator = gan.Generator(channels=3, latent_size=128, num_classes=2, hidden_dim=64).cuda()
    generator.load_state_dict(torch.load('generator.pkl'))
    generator.eval()
    files = sorted(glob.glob('train_img/*.png'), key=lambda n: int(n.split('\\')[-1].replace('.png', '')))
    images = [skimage.img_as_ubyte(plt.imread(name)) for name in files]
    imageio.mimsave('result/train.gif', images, fps=5)
    size = 64
    latent = []
    attr = []
    for _ in range(3):
        latent.append(torch.randn(size, generator.latent_size).cuda())
        attr.append(torch.randint(2, size=(size, 2)).float().cuda())
    interpolate('result/interpolation.gif', generator, latent, attr)


def reproduce_hw3():
    train()
    make_results()


if __name__ == '__main__':
    reproduce_hw3()
