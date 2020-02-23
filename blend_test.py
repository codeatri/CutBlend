import glob
import random

import numpy as np
from PIL import Image


def rand_bbox(H, W, lam):
    """
    Randomly choosing bounding box within an image.
    :param H: Height of the image
    :param W: Width of the image
    :param lam: lambda
    :return: Indexes for the box to crop to blend
    """
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def main():
    """
    This runs blending logic required for CutBlend Data Augmentation Algorithm.
    :return:
    """
    image_list = []
    for filename in glob.glob('data/input_training_highres/*.png'):
        im = Image.open(filename).resize((500, 500))
        image_list.append(im)

    alpha_list = []
    for filename in glob.glob('data/gt_training_highres/*.png'):
        im = Image.open(filename).resize((500, 500))
        rgbimg = Image.new("RGBA", im.size)
        rgbimg.paste(im)
        alpha_list.append(rgbimg)

    for i, image in enumerate(image_list):
        # Load Images
        rand_index = random.randint(0, len(image_list)-1)
        foreground = np.array(image_list[rand_index])
        background = np.array(image)
        result = np.copy(background)
        lam = np.random.beta(1., 1.)

        mask = np.array(alpha_list[rand_index].convert("RGB"))/255.

        bbx1, bby1, bbx2, bby2 = rand_bbox(background.shape[1], background.shape[0], lam)
        alpha = mask[bbx1:bbx2, bby1:bby2, :]

        # Alpha Masking
        foreground = np.multiply(foreground[bbx1:bbx2, bby1:bby2, :], alpha)
        background = np.multiply(background[bbx1:bbx2, bby1:bby2, :], 1 - alpha)
        result[bbx1:bbx2, bby1:bby2, :] = np.add(foreground, background)

        # Display image
        Image.fromarray(result).show()


if __name__ == '__main__':
    main()
