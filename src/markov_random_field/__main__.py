
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import time 

from utils.data_structures import Image2BInpainted
import utils.eeo as eeo


def loading_data(folder_path, image_filename, mask_filename, patch_size, stride, thresh=50):
    image_inpainted_name, _ = os.path.splitext(image_filename)
    image_inpainted_name = image_inpainted_name + '_'

    # loading the image and the mask
    image_rgb = imageio.imread(folder_path + '/' + image_filename)

    mask = imageio.imread(folder_path + '/' + mask_filename)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask = np.greater_equal(mask, thresh).astype(np.uint8)
    
    # on the image: set everything that's under the mask to cyan (for debugging purposes
    cyan = [0, 255, 255]
    image_rgb[mask.astype(bool), :] = cyan
    
    image = Image2BInpainted(image_rgb, mask, patch_size=patch_size, stride=stride)
    image.inpainting_approach = Image2BInpainted.USING_RBG_VALUES
    return image, image_inpainted_name


def inpaint_image(folder_path, image_filename, mask_filename, patch_size, stride, thresh_uncertainty, max_nr_labels, max_nr_iterations, thresh=128):
    """
    :param thresh: value between 0 and 255 where a high values means the mask has to be extremely certain before it is inpainted.
    :return:
    """

    image, image_inpainted_name = loading_data(folder_path, image_filename, mask_filename, patch_size, stride, thresh=thresh)
    
    print("\nNumber of pixels to be inpainted: " + str(np.count_nonzero(image.mask)))

    print("\n... Initialization ...")
    eeo.initialization(image, thresh_uncertainty)

    print("\n... Label pruning ...")
    eeo.label_pruning(image, thresh_uncertainty, max_nr_labels)

    print("\n... Computing pairwise potential matrix ...")
    eeo.compute_pairwise_potential_matrix(image, max_nr_labels)

    print("\n... Computing label cost ...")
    eeo.compute_label_cost(image, max_nr_labels)

    print("\n... Neighborhood consensus message passing ...")
    eeo.neighborhood_consensus_message_passing(image, max_nr_labels, max_nr_iterations)

    print("\n... Generating inpainted image ...")
    eeo.generate_inpainted_image(image)

    filename_inpainted =  './data/output/markov-random-field/' + image_inpainted_name + '_res.jpg'
    imageio.imwrite(filename_inpainted, image.inpainted)
    # plt.imshow(image.inpainted, interpolation='nearest')
    # plt.show()


def main():
    patch_size = 8  # needs to be an even number
    stride = patch_size // 2 
    thresh_uncertainty = 10360 
    max_nr_labels = 10
    max_nr_iterations = 10
    
    folder_path = 'data/imgs'
    image_filename = 'fibers.png'
    mask_filename = 'fibers_mask.jpg'
    
    inpaint_image(folder_path, image_filename, mask_filename, patch_size, stride, thresh_uncertainty, max_nr_labels, max_nr_iterations)


if __name__ == "__main__":
    start = time.process_time()
    main()
    print(time.process_time() - start)
