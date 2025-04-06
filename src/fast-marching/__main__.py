import time
import os 
import imageio
import matplotlib.pyplot as plt 
import argparse
import inpainter 


def main():
    args = parse_args()

    img = imageio.imread(args.in_path)
    out = img.copy()
    img_base = os.path.splitext(os.path.basename(args.in_path))[0]
    mask_img = imageio.imread(args.mask_path)
    mask = mask_img[:, :, 0].astype(bool, copy=False) if len(mask_img.shape) == 3 else mask_img[:, :].astype(bool, copy=False)

    inpainter.inpaint(out, mask, args.radius[0])
    imageio.imwrite('./data/output/fast-marching/{}_res.png'.format(img_base), out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', metavar='input_img', type=str,
    	 				help='path to input image')
    parser.add_argument('mask_path', metavar='mask_img', type=str,
    					help='path to mask image')
    parser.add_argument('-r', '--radius', metavar='R', nargs=1, type=int, default=[5],
					help='neighborhood radius')
    return parser.parse_args()


if __name__ == "__main__":
    # WIP: computation time, metrics (PSNR, SSIM, LPIPS)
    start = time.process_time()
    main()
    print(time.process_time() - start)