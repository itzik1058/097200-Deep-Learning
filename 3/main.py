import torch
import gan
import matplotlib.pyplot as plt


def inverse_transform(image, mean=0.5, std=0.5):
    image = image * std + mean
    return image.clamp(0, 1).numpy()


@torch.no_grad()
def main():
    generator, _ = gan.make_model(channels=3, latent_size=64, hidden_dim=128)
    generator.cuda()
    generator.load_state_dict(torch.load('generator.pkl'))
    generator.eval()
    latent = torch.randn(1, generator.latent_size).cuda()
    generated = generator(latent).cpu()
    plt.imshow(inverse_transform(generated[-1]).transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    main()
