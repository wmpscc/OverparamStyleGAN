import torch
import torch.nn.functional as F


def mapping(G, Z, truncation_latent, truncation):
    B, L, I, J = Z.shape
    W = []
    for i in range(I):
        style = G.style(Z[:, :, i, :]).unsqueeze(2)  # torch.Size([1, 1, 1, 512])
        if truncation < 1:
            style = truncation_latent + truncation * (style - truncation_latent)
        W.append(style)
    W = torch.cat(W, dim=-2)  # torch.Size([1, 1, 512, 512])

    return W


index = [0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16]


def conv_warper(layer, input, style, noise):
    conv = layer.conv
    modulation = conv.modulation
    batch, in_channel, height, width = input.shape
    _, O, I, _, _ = conv.weight.shape

    # style: B, 512, 512
    style = modulation(style[:, :O, :]).unsqueeze(-1).unsqueeze(-1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    out = layer.noise(out, noise=noise)
    out = layer.activate(out)

    return out


def toRGB_warper(layer, input, style, skip=None):
    conv = layer.conv
    modulation = conv.modulation
    batch, in_channel, height, width = input.shape
    _, O, I, _, _ = conv.weight.shape

    # style: B, 512, 512
    style = modulation(style[:, :O, :]).unsqueeze(-1).unsqueeze(-1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    out = out + layer.bias
    if skip is not None:
        skip = layer.upsample(skip)
        out = out + skip

    return out


def decoder(G, W):
    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    out = G.input(W[0])
    out = conv_warper(G.conv1, out, W[:, 0], noise[0])
    skip = toRGB_warper(G.to_rgb1, out, W[:, 0], None)

    i = 1
    j = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
            G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs):
        out = conv_warper(conv1, out, W[:, 0], noise=noise1)
        out = conv_warper(conv2, out, W[:, 0], noise=noise2)
        skip = toRGB_warper(to_rgb, out, W[:, 0], skip)

        i += 2
        j += 1

    image = skip

    return image
