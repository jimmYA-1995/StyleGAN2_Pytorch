import logging
import argparse
from pathlib import Path

import torch
import numpy as np
import skimage.io as io
from torch.nn import functional as F
from torchvision import transforms, utils
from tqdm import tqdm

import lpips
from models import Generator


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


def load_condition_sample(sample_dir, batch_size):
    IMG_EXTS = ['jpg', 'png', 'jpeg']
    sample_paths = []
    for ext in IMG_EXTS:
        sample_paths.extend(list(Path(sample_dir).glob(f'*.{ext}')))
    assert len(sample_paths) >= batch_size, f"Need more samples. {len(sample_paths)} < {batch_size}"
    cond_samples = []
                
    trf = [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [5],
                                     inplace=True),
            ]
    transform = transforms.Compose(trf)
    
    for i, p in enumerate(sample_paths):
        if batch_size != -1 and i == batch_size:
            break
        cond_samples.append(transform(io.imread(p)[..., None])[None, ...])
    return torch.cat(cond_samples, dim=0)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--space', choices=['z', 'w'])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('ckpt', metavar='CHECKPOINT')

    args = parser.parse_args()

    latent_dim = 512
    
    logging.info("load checkpoitns & model")
    ckpt = torch.load(args.ckpt)

    g = Generator(latent_dim, 0, args.size, extra_channels=0, use_sk=True, use_mk=False, is_training=False).to(device)
    g.load_state_dict(ckpt['g_ema'])
    g.eval()
    
    sample_dir = Path('~/data/deepfashion_h_p_r/sample_sk').expanduser()
    logging.info(f"loading conditional samples from {sample_dir}")
    cond_samples = load_condition_sample(sample_dir, args.batch).to(device)
    repeat_cond_samples = cond_samples.clone()
    repeat_cond_samples = repeat_cond_samples.unsqueeze(1).repeat(1, 2, 1, 1, 1).view(-1, 1, args.size, args.size)
    

    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
    )

    distances = []

    n_batch = args.n_sample // args.batch
    resid = args.n_sample - (n_batch * args.batch)
    batch_sizes = [args.batch] * n_batch + [resid]
    
    idx = 0
    with torch.no_grad():
        for batch in tqdm(batch_sizes):

            inputs = torch.randn([batch * 2, latent_dim], device=device)
            lerp_t = torch.rand(batch, device=device)

            # if args.space == 'w':
            latent = g.get_latent([inputs], sk=cond_samples)
            n_broadcast = latent.shape[1]
            latent =  latent[:,0,:]
            latent_t0, latent_t1 = latent[::2], latent[1::2]
            latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
            latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + args.eps)
            latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)
            latent_e = latent_e.unsqueeze(1).repeat(1, n_broadcast, 1)
            
            image = g.synthesis_network(latent_e, sk=repeat_cond_samples[:latent_e.shape[0]])
            
            j=0
            for img1, img2 in zip(image[::2], image[1::2]):
                utils.save_image(
                            img1,
                            f'/root/notebooks/tmp/ppl_samples1/img-{str(idx).zfill(4)}-{j}.png',
                            nrow=1, #int(args.batch ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                utils.save_image(
                            img2,
                            f'/root/notebooks/tmp/ppl_samples2/img-{str(idx).zfill(4)}-{j}.png',
                            nrow=1, #int(args.batch ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                j+=1
           
            if args.crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode='bilinear', align_corners=False
                )

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                args.eps ** 2
            )
            distances.append(dist.to('cpu').numpy())
            idx += 1

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )
    
    print(distances)
    print('ppl:', filtered_dist.mean())
