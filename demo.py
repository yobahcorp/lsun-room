import pathlib
import click
import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from datasets import sequence
from trainer import core

torch.backends.cudnn.benchmark = True


class Predictor:

    def __init__(self, weight_path):
        self.model = core.LayoutSeg.load_from_checkpoint(weight_path, backbone='resnet101')
        self.model.freeze()
        self.model.cpu()

    @torch.no_grad()
    def feed(self, image: torch.Tensor, alpha=.4) -> np.ndarray:
        _, outputs = self.model(image.unsqueeze(0).cpu())
        label = core.label_as_rgb_visual(outputs.cpu()).squeeze(0)
        # blend_output = (image / 2 + .5) * (1 - alpha) + (label * alpha)
        blend_output = label
        return blend_output.permute(1, 2, 0).numpy()


@click.group()
def cli():
    pass


@cli.command()
@click.option('--path', type=click.Path(exists=True))
@click.option('--weight', type=click.Path(exists=True))
@click.option('--image_size', default=320, type=int)
def image(path, weight, image_size):
    logger.info('Press `q` to exit the sequence inference.')
    predictor = Predictor(weight_path=weight)
    images = sequence.ImageFolder(image_size, path)

    for image, shape, _ in images:
        output = cv2.resize(predictor.feed(image), shape)
        Image.fromarray((output * 255).astype(np.uint8)).save('output.png')
        # cv2.imshow('layout', output[..., ::-1])
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

if __name__ == '__main__':
    cli()
