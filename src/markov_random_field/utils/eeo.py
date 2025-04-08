
import numpy as np
import sys
import os
import imageio

import math
from scipy import signal


from .data_structures import Image2BInpainted, Node, coordinates_to_position, position_to_coordinates, UP, DOWN, LEFT, RIGHT, get_half_patch_from_patch, opposite_side

POOL_SIZE = 8


# using the rgb values of the patches for comparison, as opposed to their descriptors
def initialization(image, thresh_uncertainty):

    global nodes
    global nodes_count
    global nodes_order

    nodes = {}  # the indices in this list patches match the node_id
    nodes_count = 0
    nodes_order = []
    
    # for all the patches in an image with stride $stride
    for y in range(0, image.width - image.patch_size + 1, image.stride):
        for x in range(0, image.height - image.patch_size + 1, image.stride):

            patch_mask_overlap = image.mask[x: x + image.patch_size, y: y + image.patch_size]
            patch_mask_overlap_nonzero_elements = np.count_nonzero(patch_mask_overlap)

            # determine with which regions is the patch overlapping
            if patch_mask_overlap_nonzero_elements == 0:
                patch_overlap_source_region = True
                patch_overlap_target_region = False
            elif patch_mask_overlap_nonzero_elements == image.patch_size**2:
                patch_overlap_source_region = False
                patch_overlap_target_region = True
            else:
                patch_overlap_source_region = True
                patch_overlap_target_region = True

            if patch_overlap_target_region:
                patch_position = coordinates_to_position(x, y, image.height, image.patch_size)
                node = Node(patch_position, patch_overlap_source_region, x, y)
                nodes[patch_position] = node
                nodes_count += 1

    labels_diametar = 100

    # using the rgb values of the patches for comparison, as opposed to their descriptors
    if image.inpainting_approach == Image2BInpainted.USING_RBG_VALUES:
        for i, node in enumerate(nodes.values()):

            sys.stdout.write("\rInitialising node " + str(i + 1) + "/" + str(nodes_count))

            if node.overlap_source_region:

                node_rgb = image.rgb[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
                mask_3ch = np.repeat(mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))
                node_rgb = node_rgb * (1 - mask_3ch)

                # compare the node patch to all patches that are completely in the source region
                for y_compare in range(max(node.y_coord - labels_diametar, 0),
                                       min(node.y_coord + labels_diametar, image.width - image.patch_size + 1)):
                    for x_compare in range(max(node.x_coord - labels_diametar, 0),
                                           min(node.x_coord + labels_diametar, image.height - image.patch_size + 1)):

                        patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size]
                        patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                        if patch_compare_mask_overlap_nonzero_elements == 0:
                            patch_compare_rgb = image.rgb[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size, :]
                            patch_compare_rgb = patch_compare_rgb * (1 - mask_3ch)

                            patch_difference = np.sum(np.subtract(node_rgb, patch_compare_rgb, dtype=np.int32) ** 2)

                            patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height, image.patch_size)
                            node.differences[patch_compare_position] = patch_difference
                            node.labels.append(patch_compare_position)

                temp_min_diff = min(node.differences.values())
                temp = [value - temp_min_diff for value in list(node.differences.values())]
                #TODO change thresh_uncertainty such that only patches which are completely in the target region
                #     get assigned the priority value 1.0 (but keep in mind it is used elsewhere)
                node_uncertainty = len([val for (i, val) in enumerate(temp) if val < thresh_uncertainty])

            # if the patch is completely in the target region
            else:

                # make all patches that are completely in the source region be the label of the patch
                for y_compare in range(max(node.y_coord - labels_diametar, 0),
                                       min(node.y_coord + labels_diametar, image.width - image.patch_size + 1)):
                    for x_compare in range(max(node.x_coord - labels_diametar, 0),
                                           min(node.x_coord + labels_diametar, image.height - image.patch_size + 1)):

                        patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size]
                        patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                        if patch_compare_mask_overlap_nonzero_elements == 0:
                            patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height, image.patch_size)
                            node.differences[patch_compare_position] = 0
                            node.labels.append(patch_compare_position)

                node_uncertainty = len(node.labels)

            # the higher priority the higher priority :D
            node.priority = len(node.labels) / max(node_uncertainty, 1)
    else:
        raise AssertionError("Inpainting approach has not been properly set.")

    print("\nTotal number of patches: ", len(nodes))
    print("Number of patches to be inpainted: ", nodes_count)



def label_pruning(image, thresh_uncertainty, max_nr_labels):
    global nodes
    global nodes_count
    global nodes_order

    # make a copy of the differences which we can edit and use in this method, and afterwards discard
    for node in nodes.values():
        node.additional_differences = node.differences.copy()

    # for all the patches that have an overlap with the target region (aka nodes)
    for i in range(nodes_count):

        # # find the node with the highest priority that hasn't yet been visited
        node_highest_priority = max(filter(lambda node:not node.committed, nodes.values()),
                                    key=lambda node: node.priority,
                                    default=-1)
        if node_highest_priority == -1:
            err_msg = f'Nodes has no non-committed entries. Make sure the global values are Reset when inpainting new image! {nodes.values}'
            raise AssertionError(err_msg)
            
        highest_priority = node_highest_priority.priority
        node_highest_priority_id = node_highest_priority.node_id
        
        node = nodes[node_highest_priority_id]
        node.committed = True

        node.prune_labels(max_nr_labels)

        print('Highest priority node {0:3d}/{1:3d}: {2:d}'.format(i + 1, nodes_count, node_highest_priority_id))
        nodes_order.append(node_highest_priority_id)

        # get the neighbors if they exist and have overlap with the target region
        node_neighbor_up, node_neighbor_down, node_neighbor_left, node_neighbor_right = get_neighbor_nodes(
            node, image)

        if image.inpainting_approach == Image2BInpainted.USING_RBG_VALUES:
            update_neighbors_priority_rgb(node, node_neighbor_up, UP, image, thresh_uncertainty)
            update_neighbors_priority_rgb(node, node_neighbor_down, DOWN, image, thresh_uncertainty)
            update_neighbors_priority_rgb(node, node_neighbor_left, LEFT, image, thresh_uncertainty)
            update_neighbors_priority_rgb(node, node_neighbor_right, RIGHT, image, thresh_uncertainty)
        else:
            raise AssertionError("Inpainting approach has not been properly set.")


# using the rgb values of the patches for comparison, as opposed to their descriptors
def update_neighbors_priority_rgb(node, neighbor, side, image, thresh_uncertainty):

    # if neighbor is a node that hasn't been committed yet
    if neighbor is not None and not neighbor.committed:
        min_additional_difference = sys.maxsize
        additional_differences = {}

        for neighbors_label_id in neighbor.labels:

            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

            patch_neighbors_label_rgb = image.rgb[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                        neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
            patch_neighbors_label_rgb_half = get_half_patch_from_patch(patch_neighbors_label_rgb, image.stride, opposite_side(side))

            for node_label_id in node.pruned_labels:

                node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size,
                                   node_label_y_coord: node_label_y_coord + image.patch_size, :]
                patchs_label_rgb_half = get_half_patch_from_patch(patchs_label_rgb, image.stride, side)

                difference = np.sum((patch_neighbors_label_rgb_half - patchs_label_rgb_half).astype(np.int32)**2)

                if difference < (min_additional_difference):    
                    min_additional_difference = difference

            additional_differences[neighbors_label_id] = min_additional_difference
            min_additional_difference = sys.maxsize

        for key in additional_differences.keys():
            if key in neighbor.additional_differences:
                neighbor.additional_differences[key] += additional_differences[key]
            else:
                neighbor.additional_differences[key] = additional_differences[key]
                # print("Will it ever come to this? (2) (TODO delete if unnecessary)")

        temp_min_diff = min(neighbor.additional_differences.values())
        temp = [value - temp_min_diff for value in
                list(neighbor.additional_differences.values())]
        neighbor_uncertainty = [value < thresh_uncertainty for (i, value) in enumerate(temp)].count(True)

        neighbor.priority = len(neighbor.additional_differences) / neighbor_uncertainty #len(patch_neighbor.differences)?


def get_neighbor_nodes(node, image):

    neighbor_up_id = node.get_up_neighbor_position(image)
    if neighbor_up_id is None:
        neighbor_up = None
    else:
        neighbor_up = nodes.get(neighbor_up_id) 

    neighbor_down_id = node.get_down_neighbor_position(image)
    if neighbor_down_id is None:
        neighbor_down = None
    else:
        neighbor_down = nodes.get(neighbor_down_id)

    neighbor_left_id = node.get_left_neighbor_position(image)
    if neighbor_left_id is None:
        neighbor_left = None
    else:
        neighbor_left = nodes.get(neighbor_left_id)

    neighbor_right_id = node.get_right_neighbor_position(image)
    if neighbor_right_id is None:
        neighbor_right = None
    else:
        neighbor_right = nodes.get(neighbor_right_id)

    return neighbor_up, neighbor_down, neighbor_left, neighbor_right


def compute_pairwise_potential_matrix(image, max_nr_labels):

    global nodes
    global nodes_count

    if image.inpainting_approach == Image2BInpainted.USING_RBG_VALUES:

        # calculate pairwise potential matrix for all pairs of nodes (pathces that have overlap with the target region)
        for node in nodes.values():

            # get the neighbors if they exist and have overlap with the target region (i.e. if they're nodes)
            neighbor_up, _, neighbor_left, _ = get_neighbor_nodes(node, image)

            if neighbor_up is not None:

                potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                for i, node_label_id in enumerate(node.pruned_labels):

                    node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                    patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size,
                                       node_label_y_coord: node_label_y_coord + image.patch_size, :]

                    patchs_label_rgb_up = get_half_patch_from_patch(patchs_label_rgb, image.stride, UP)

                    for j, neighbors_label_id in enumerate(neighbor_up.pruned_labels):

                        neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

                        patchs_neighbors_label_rgb = image.rgb[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                                     neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]

                        patchs_neighbors_label_rgb_down = get_half_patch_from_patch(patchs_neighbors_label_rgb, image.stride, DOWN)

                        potential_matrix[i, j] = np.sum(np.subtract(patchs_label_rgb_up, patchs_neighbors_label_rgb_down, dtype=np.int32) ** 2)


                node.potential_matrix_up = potential_matrix
                neighbor_up.potential_matrix_down = potential_matrix

            if neighbor_left is not None:

                potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                for i, node_label_id in enumerate(node.pruned_labels):

                    node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                    patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size,
                                       node_label_y_coord: node_label_y_coord + image.patch_size, :]

                    patchs_label_rgb_left = get_half_patch_from_patch(patchs_label_rgb, image.stride, LEFT)

                    for j, neighbors_label_id in enumerate(neighbor_left.pruned_labels):

                        neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

                        patchs_neighbors_label_rgb = image.rgb[
                                                     neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                                     neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size,
                                                     :]

                        patchs_neighbors_label_rgb_right = get_half_patch_from_patch(patchs_neighbors_label_rgb, image.stride,
                                                                                    RIGHT)

                        potential_matrix[i, j] = np.sum(np.subtract(patchs_label_rgb_left, patchs_neighbors_label_rgb_right, dtype=np.int32) ** 2)


                node.potential_matrix_left = potential_matrix
                neighbor_left.potential_matrix_right = potential_matrix
    else:
        raise AssertionError("Inpainting approach has not been properly set.")


def compute_label_cost(image, max_nr_labels):
    global nodes
    global nodes_count

    if image.inpainting_approach == Image2BInpainted.USING_RBG_VALUES:

        for node in nodes.values():

            node.label_cost = [0 for _ in range(max_nr_labels)]

            if node.overlap_source_region:

                patch_rgb = image.rgb[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
                mask_3ch = np.repeat(mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))
                patch_rgb = patch_rgb * (1 - mask_3ch)

                for i, node_label_id in enumerate(node.pruned_labels):

                    node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
                    patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size, node_label_y_coord: node_label_y_coord + image.patch_size, :]
                    patchs_label_rgb = patchs_label_rgb * (1 - mask_3ch)

                    node.label_cost[i] = np.sum(np.subtract(patch_rgb, patchs_label_rgb, dtype=np.int32) ** 2)

            node.local_likelihood = [math.exp(-cost * (1/100000)) for cost in node.label_cost]
            node.mask = node.local_likelihood.index(max(node.local_likelihood))
    else:
        raise AssertionError("Inpainting approach has not been properly set.")



def neighborhood_consensus_message_passing(image, max_nr_labels, max_nr_iterations):
    global nodes
    for node in nodes.values():
        node.messages = np.ones(max_nr_labels)
        node.beliefs = np.zeros(max_nr_labels)
        node.beliefs[node.mask] = 1

        node.beliefs_new = node.beliefs.copy()


    for i in range(max_nr_iterations):
        for node in nodes.values():
            neighbor_up, neighbor_down, neighbor_left, neighbor_right = get_neighbor_nodes(node, image)

            if neighbor_up is not None:
                node.messages = np.matmul(node.potential_matrix_up, neighbor_up.beliefs.reshape((max_nr_labels, 1)))
            if neighbor_down is not None:
                node.messages = np.matmul(node.potential_matrix_down, neighbor_down.beliefs.reshape((max_nr_labels, 1)))
            if neighbor_left is not None:
                node.messages = np.matmul(node.potential_matrix_left, neighbor_left.beliefs.reshape((max_nr_labels, 1)))
            if neighbor_right is not None:
                node.messages = np.matmul(node.potential_matrix_right, neighbor_right.beliefs.reshape((max_nr_labels, 1)))

            node.messages = np.array([math.exp(-message * (1 / 100000)) for message in
                                       node.messages.reshape((max_nr_labels, 1))]).reshape(1, max_nr_labels)

            node.beliefs_new = np.multiply(node.messages, node.local_likelihood)
            node.beliefs_new = node.beliefs_new / node.beliefs_new.sum()  # normalise to sum up to 1

        # update the mask and beliefs for all the nodes
        for node in nodes.values():
            node.mask = node.beliefs_new.argmax()
            node.beliefs = node.beliefs_new


def generate_inpainted_image(image, blend_method=1, mask_type=1):
    """
    :param image:
    :param blend_method: Either 0 or 1
    :param mask_type: Either 0 or 1
    :return:
    """

    global nodes
    global nodes_count
    global nodes_order
    
    assert blend_method in [0, 1], 'blend_method should be either 0 or 1'
    assert mask_type in [0, 1], 'mask_type should be either 0 or 1'

    target_region = np.copy(image.mask).astype('bool')
    original_mask = np.copy(image.mask).astype('bool')
    
    cyan = np.reshape([0, 255, 255], (1, 1, 3))
    image.inpainted = np.copy(image.rgb)
    image.inpainted[target_region, :] = cyan    # To make clear in debugging mode

    if mask_type == 0:
        filter_size = max(2, image.patch_size // 2)  # should be > 1
        smooth_filter = generate_smooth_filter(filter_size)
        
        blend_mask = generate_blend_mask(image.patch_size)
        blend_mask = signal.convolve2d(blend_mask, smooth_filter, boundary='symm', mode='same')
    elif mask_type == 1:
        blend_mask = generate_linear_diamond_mask(image.patch_size)
    else:
        blend_mask = None
    
    blend_mask_rgb = np.repeat(blend_mask[..., None], 3, axis=2)
    for i in range(len(nodes_order)):

        node_id = nodes_order[i]
        node = nodes[node_id]

        node_mask_patch_x_coord, node_mask_patch_y_coord =  position_to_coordinates(node.pruned_labels[node.mask], image.height, image.patch_size)

        node_rgb = image.inpainted[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]

        node_rgb_new = image.inpainted[node_mask_patch_x_coord: node_mask_patch_x_coord + image.patch_size, node_mask_patch_y_coord: node_mask_patch_y_coord + image.patch_size, :]

        if blend_method == 0:
            image.inpainted[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :] =\
                node_rgb*blend_mask_rgb + node_rgb_new*(1 - blend_mask_rgb)
        
        # Only inpaint/update pixels belonging to mask
        elif blend_method == 1:
            mask_new = target_region[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
            mask_new_orig = original_mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]

            # only inpaint the mask part
            image.inpainted[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :][mask_new]=\
                (node_rgb_new)[mask_new]   # TODO

            # average out with previous values
            mask_prev = np.logical_and(mask_new_orig, np.logical_not(mask_new))
            
            image.inpainted[node.x_coord: node.x_coord + image.patch_size,
                            node.y_coord: node.y_coord + image.patch_size, :][mask_prev] = \
                (node_rgb*blend_mask_rgb + node_rgb_new*(1 - blend_mask_rgb))[mask_prev]
                
        else:
            ValueError(f'Unknown inpainting strategy: {blend_method}')

        target_region[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size] = False
    image.inpainted = image.inpainted.astype(np.uint8)



def generate_smooth_filter(kernel_size):
    if kernel_size <= 1:
        raise Exception('Kernel size for the smooth filter should be larger than 1, but is {}.'.format(kernel_size))

    kernel_1D = np.array([0.5, 0.5]).transpose()

    for i in range(kernel_size - 2):
        kernel_1D = np.convolve(np.array([0.5, 0.5]).transpose(), kernel_1D)

    kernel_1D = kernel_1D.reshape((kernel_size, 1))
    kernel_2D = np.matmul(kernel_1D, kernel_1D.transpose())
    return kernel_2D



def generate_blend_mask(patch_size):
    blend_mask = np.zeros((patch_size, patch_size))
    blend_mask[:patch_size // 3, :] = 1
    blend_mask[:, :patch_size // 3] = 1
    return blend_mask


def generate_linear_diamond_mask(patch_size):
    blend_mask = np.zeros((patch_size, patch_size))
    patch_size_half = int(np.ceil(patch_size/2.))
    
    for i in range(patch_size_half):
        for j in range(patch_size_half):
            val = (i + j) / (patch_size_half - 1 + patch_size_half - 1)

            blend_mask[i, j] = val
            blend_mask[patch_size-1-i, j] = val
            blend_mask[i, patch_size-1-j] = val
            blend_mask[patch_size-1-i, patch_size-1-j] = val

    return blend_mask




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


def inpaint_image(img=None, mask=None, patch_size=16, stride=8, thresh_uncertainty=10360, max_nr_labels=10, max_nr_iterations=10, thresh=128):
    """
    :param thresh: value between 0 and 255 where a high values means the mask has to be extremely certain before it is inpainted.
    :return:
    """
 
    if len(mask.shape) == 3:
        mask = mask[:, :,0]
    mask = np.greater_equal(mask, thresh).astype(np.uint8)

    cyan = [0, 255, 255]
    img[mask.astype(bool), :] = cyan
    image = Image2BInpainted(img, mask, patch_size=patch_size, stride=stride)
    image.inpainting_approach = Image2BInpainted.USING_RBG_VALUES
    
    print("\nNumber of pixels to be inpainted: " + str(np.count_nonzero(image.mask)))

    print("\n... Initialization ...")
    initialization(image, thresh_uncertainty)

    print("\n... Label pruning ...")
    label_pruning(image, thresh_uncertainty, max_nr_labels)

    print("\n... Computing pairwise potential matrix ...")
    compute_pairwise_potential_matrix(image, max_nr_labels)

    print("\n... Computing label cost ...")
    compute_label_cost(image, max_nr_labels)

    print("\n... Neighborhood consensus message passing ...")
    neighborhood_consensus_message_passing(image, max_nr_labels, max_nr_iterations)

    print("\n... Generating inpainted image ...")
    generate_inpainted_image(image)

    return image.inpainted

    # filename_inpainted =  './data/output/markov-random-field/' + image_inpainted_name + '_res.jpg'
    # imageio.imwrite(filename_inpainted, image.inpainted)
    # plt.imshow(image.inpainted, interpolation='nearest')
    # plt.show()
