import os 
import argparse
from skimage.io import imread, imsave
import time
from inpainter import Inpainter


def main():
    args = parse_args()

    image = imread(args.input_image)
    img_base = os.path.splitext(os.path.basename(args.input_image))[0]
    mask = imread(args.mask)

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    output_image = Inpainter(image, mask, patch_size=args.patch_size).inpaint()
    imsave('./data/output/exemplar-based/{}_res.png'.format(img_base), output_image)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image',
                        help='the image containing objects to be removed')
    parser.add_argument('mask',
                        help='the mask image of the region to be removed')
    parser.add_argument('-ps',
                        '--patch-size',
                        help='the size of the patches',
                        type=int,
                        default=9)
    parser.add_argument('-o',
                        '--output',
                        help='the file path to save the output image',
                        default='./data/output/{}')
    return parser.parse_args()


if __name__ == '__main__':
    # WIP: computation time, metrics (PSNR, SSIM, LPIPS)
    start = time.process_time()
    main()
    print(time.process_time() - start)
