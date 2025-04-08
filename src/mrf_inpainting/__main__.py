
import argparse
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filters import sobel
from scipy.ndimage import binary_dilation
from inpainter import MRFInpainting


# Example usage
def main():
    """
    Example usage of the MRF-based image inpainting algorithm.
    """
    # Create an instance of the MRF inpainting class
    args = parse_args()
    inpainter = MRFInpainting(patch_size=9, search_window=30, alpha=0.8, max_iterations=400)
    

    try:
        # Load image and mask
        image, mask = inpainter.load_image_and_mask(image_path=args.in_path, mask_path=args.mask_path)
        
        # Perform inpainting
        inpainted_image = inpainter.priority_inpaint()

        plt.imshow(inpainted_image)
        plt.show()
        # cv2.imwrite("inpainted_result.jpg", inpainted_image)
        print("Inpainting completed and result saved.")
        
    except Exception as e:
        print(f"Error: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', metavar='input_img', type=str,
    	 				help='path to input image')
    parser.add_argument('mask_path', metavar='mask_img', type=str,
    					help='path to mask image')
    return parser.parse_args()


if __name__ == "__main__":
    main()