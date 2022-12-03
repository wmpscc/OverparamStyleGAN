import torch
from models.stylegan2.model import Generator
from torchvision.utils import save_image
from models.generator_overparams import decoder, mapping
import numpy as np

device = torch.device('cuda:0')


def init_model(image_size=1024, latent_dim=512):
    generator = Generator(image_size, latent_dim, 8).eval()
    checkpoint = torch.load('./checkpoint/stylegan2-ffhq-config-f.pt', map_location='cpu')
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator.eval()

    truncation_latent = torch.load('./truncation_latent')
    return generator, truncation_latent


if __name__ == '__main__':
    generator, truncation_latent = init_model()
    generator, truncation_latent = generator.to(device), truncation_latent.to(device)

    noise = torch.randn((1, 1, 1, 512))
    tmp = []
    for i in range(512):
        z = (torch.randn((1, 1, 1, 512)) + noise) / np.sqrt(2)
        tmp.append(z)
    Z = torch.cat(tmp, dim=-2).to(device)

    W = mapping(generator, Z, truncation_latent, 0.9)  # get a latent code
    img = decoder(generator, W)
    save_image(img * 0.5 + 0.5, './result/sample.png')
