import os
import torch
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
from PIL import Image
from models.stylegan2.model import Generator
from torchvision.utils import save_image
from models.generator_overparams import decoder
import lpips_metric
from tqdm import tqdm

device = torch.device('cuda:0')


def load_img(path_img, img_size=(256, 256)):
    transform = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    img = Image.open(path_img)
    img = transform(img)
    img.unsqueeze_(0)
    return img


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
    truncation_latent = torch.repeat_interleave(truncation_latent, 512, dim=2)
    img = load_img('./data/President_Barack_Obama.jpg', (1024, 1024)).to(device)

    styles = torch.randn((1, 1, 512, 512), device=device)
    styles = truncation_latent + 0.7 * (styles - truncation_latent)
    styles.requires_grad_(True)

    optimE = optim.AdamW([styles], lr=0.01, weight_decay=0.05, betas=(0.9, 0.999))

    percept = lpips_metric.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device)
    for i in tqdm(range(10000)):
        img_gen = decoder(generator, styles)
        img_gen_down = F.interpolate(img_gen, size=(img.shape[-2], img.shape[-1]))

        p_loss = percept(img_gen_down, img).sum()
        mse_loss = F.mse_loss(img_gen_down, img)
        rec_errors = F.l1_loss(img_gen_down, img)

        loss = p_loss + 0.5 * mse_loss
        optimE.zero_grad()
        loss.backward()
        optimE.step()

        print(i, 'p_loss:', p_loss.detach().cpu().item(), 'mse_loss:', mse_loss.detach().cpu().item(),
              'rec_errors:', rec_errors.detach().cpu().item())
        if i % 100 == 0:
            os.makedirs('./result', exist_ok=True)
            save_image(img_gen * 0.5 + 0.5, f'./result/inversion{i}.png')
