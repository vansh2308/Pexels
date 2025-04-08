import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filters import sobel
from scipy.ndimage import binary_dilation

class MRFInpainting:
    def __init__(self, patch_size=9, search_window=30, alpha=0.8, max_iterations=10):
        """
        Initialize the MRF-based image inpainting algorithm.
        
        Parameters:
        - patch_size: size of patches used for filling (must be odd)
        - search_window: size of the search window for similar patches
        - alpha: weight between structure and texture similarity
        - max_iterations: maximum number of iterations
        """
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.search_window = search_window
        self.alpha = alpha
        self.max_iterations = max_iterations
        
    def load_image_and_mask(self, input_img=None, mask_img=None, image_path=None, mask_path=None, mask_value=None):
        """
        Load the image and mask.
        
        Parameters:
        - image_path: path to the input image
        - mask_path: path to the mask image (optional)
        - mask_value: value in the image that represents the masked region (optional)
        
        Returns:
        - image: loaded image
        - mask: binary mask where 1 represents the inpainting region
        """
        if image_path:
            self.original_image = cv2.imread(image_path)
        elif input_img is not None:
            self.original_image = input_img
        if self.original_image is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        self.image = self.original_image.copy()
        
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.mask = (mask > 0).astype(np.uint8)
        elif mask_value is not None:
            mask = np.all(self.image == mask_value, axis=2)
            self.mask = mask.astype(np.uint8)
        elif mask_img is not None:
            # mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            self.mask = (mask_img > 0).astype(np.uint8)
        else:
            raise ValueError("Either mask_path or mask_value must be provided")
            
        # Dilate the mask slightly to ensure all damaged pixels are included
        self.mask = binary_dilation(self.mask, iterations=2).astype(np.uint8)
        return self.image, self.mask
    
   
    def compute_structure_tensor(self, gray_image):
        """
        Compute the structure tensor for the given grayscale image.
        
        Parameters:
        - gray_image: grayscale image
        
        Returns:
        - structure_tensor: tensor representing local structure
        - direction: direction of principal orientation
        - coherence: measure of structural coherence
        """
        # Compute gradients
        grad_x = sobel(gray_image, axis=1)
        grad_y = sobel(gray_image, axis=0)
        
        # Compute structure tensor components
        j11 = ndimage.gaussian_filter(grad_x * grad_x, sigma=1.0)
        j12 = ndimage.gaussian_filter(grad_x * grad_y, sigma=1.0)
        j22 = ndimage.gaussian_filter(grad_y * grad_y, sigma=1.0)
        
        # Calculate eigenvalues
        trace = j11 + j22
        det = j11 * j22 - j12 * j12
        
        # Avoid numerical issues
        det = np.maximum(det, 1e-10)
        
        # Calculate eigenvalues lambda1 >= lambda2
        temp = np.sqrt(trace * trace / 4 - det)
        lambda1 = trace / 2 + temp
        lambda2 = trace / 2 - temp
        
        # Calculate direction (orientation angle in radians)
        direction = np.arctan2(2 * j12, j22 - j11) / 2
        
        # Calculate coherence (anisotropy measure)
        lambda1 = np.maximum(lambda1, 1e-10)
        coherence = np.divide(lambda1 - lambda2, lambda1 + lambda2, 
                             out=np.zeros_like(lambda1), where=lambda1+lambda2 > 1e-10)
        
        return (j11, j12, j22), direction, coherence
    
    def compute_direction_histogram(self, direction, coherence, mask):
        """
        Compute a histogram of directions weighted by coherence.
        
        Parameters:
        - direction: direction map (in radians)
        - coherence: coherence map
        - mask: binary mask of the region to analyze
        
        Returns:
        - hist: direction histogram
        - bin_edges: histogram bin edges
        """
        # Convert directions to degrees
        degrees = np.degrees(direction) % 180
        
        # Define histogram bins (18 bins of 10 degrees each)
        bins = 18
        hist, bin_edges = np.histogram(degrees[mask > 0], bins=bins, range=(0, 180),
                                      weights=coherence[mask > 0])
        
        # Normalize histogram
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        return hist, bin_edges
    
    def find_fill_front(self):
        """
        Find the fill front (boundary between known and unknown regions).
        
        Returns:
        - fill_front: binary mask of the fill front
        - confidence: confidence map for the fill front
        """
        # Dilate the mask to find boundary pixels
        dilated = binary_dilation(self.mask, iterations=1).astype(np.uint8)
        fill_front = dilated - self.mask
        
        # Compute confidence for each pixel on the fill front
        confidence = np.zeros_like(fill_front, dtype=float)
        indices = np.where(fill_front > 0)
        
        for i, j in zip(indices[0], indices[1]):
            # Extract patch around the pixel
            patch_mask = self.get_patch_mask(i, j)
            # Confidence is the ratio of known pixels in the patch
            confidence[i, j] = np.sum(1 - self.mask[patch_mask]) / np.sum(patch_mask)
        
        return fill_front, confidence
    
    def get_patch_mask(self, i, j):
        """
        Get a binary mask for the patch centered at (i, j).
        
        Parameters:
        - i, j: center coordinates
        
        Returns:
        - patch_mask: binary mask with 1s inside the patch region
        """
        h, w = self.mask.shape
        patch_mask = np.zeros((h, w), dtype=bool)
        
        # Define patch boundaries
        top = max(0, i - self.half_patch)
        bottom = min(h, i + self.half_patch + 1)
        left = max(0, j - self.half_patch)
        right = min(w, j + self.half_patch + 1)
        
        patch_mask[top:bottom, left:right] = True
        return patch_mask
    
    def get_patch(self, img, i, j):
        """
        Extract a patch from the image centered at (i, j).
        
        Parameters:
        - img: input image
        - i, j: center coordinates
        
        Returns:
        - patch: extracted patch
        """
        h, w = img.shape[:2]
        
        # Define patch boundaries
        top = max(0, i - self.half_patch)
        bottom = min(h, i + self.half_patch + 1)
        left = max(0, j - self.half_patch)
        right = min(w, j + self.half_patch + 1)
        
        # Extract the patch
        if len(img.shape) == 3:  # Color image
            patch = np.zeros((self.patch_size, self.patch_size, img.shape[2]), dtype=img.dtype)
            patch_content = img[top:bottom, left:right]
            # Place the extracted content in the center of the patch
            start_i = self.half_patch - (i - top)
            start_j = self.half_patch - (j - left)
            end_i = start_i + (bottom - top)
            end_j = start_j + (right - left)
            patch[start_i:end_i, start_j:end_j] = patch_content
        else:  # Grayscale image
            patch = np.zeros((self.patch_size, self.patch_size), dtype=img.dtype)
            patch_content = img[top:bottom, left:right]
            # Place the extracted content in the center of the patch
            start_i = self.half_patch - (i - top)
            start_j = self.half_patch - (j - left)
            end_i = start_i + (bottom - top)
            end_j = start_j + (right - left)
            patch[start_i:end_i, start_j:end_j] = patch_content
            
        return patch
    
    def find_best_patch(self, target_i, target_j, direction_hist):
        """
        Find the best matching patch for inpainting.
        
        Parameters:
        - target_i, target_j: coordinates of the target pixel
        - direction_hist: direction histogram of the target region
        
        Returns:
        - best_i, best_j: coordinates of the best matching patch center
        - min_dist: distance/similarity score of the best match
        """
        h, w = self.image.shape[:2]
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Get the target patch
        target_patch = self.get_patch(self.image, target_i, target_j)
        target_patch_mask = self.get_patch(1 - self.mask, target_i, target_j).astype(bool)
        
        # Structure tensor for target patch (use only known pixels)
        target_gray_patch = self.get_patch(gray_image, target_i, target_j)
        
        # Search window boundaries
        top = max(self.half_patch, target_i - self.search_window)
        bottom = min(h - self.half_patch, target_i + self.search_window)
        left = max(self.half_patch, target_j - self.search_window)
        right = min(w - self.half_patch, target_j + self.search_window)
        
        best_i, best_j = target_i, target_j
        min_dist = float('inf')
        
        # Search for the best matching patch
        for i in range(top, bottom):
            for j in range(left, right):
                # Skip if patch contains unknown pixels
                patch_mask = self.get_patch_mask(i, j)
                if np.any(self.mask[patch_mask]):
                    continue
                
                # Get source patch
                source_patch = self.get_patch(self.image, i, j)
                source_gray_patch = self.get_patch(gray_image, i, j)
                
                # Compute structure tensor for source patch
                _, source_dir, source_coh = self.compute_structure_tensor(source_gray_patch)
                source_dir_hist, _ = self.compute_direction_histogram(source_dir, source_coh, np.ones_like(source_dir))
                
                # Compute direction histogram similarity
                hist_dist = np.sum((source_dir_hist - direction_hist) ** 2)
                
                # Compute pixel-wise similarity (only for known pixels in target)
                pixel_dist = np.sum(((target_patch - source_patch) ** 2) * np.expand_dims(target_patch_mask, -1))
                pixel_dist /= np.sum(target_patch_mask)
                
                # Combined distance
                dist = self.alpha * hist_dist + (1 - self.alpha) * pixel_dist
                
                if dist < min_dist:
                    min_dist = dist
                    best_i, best_j = i, j
        
        return best_i, best_j, min_dist
    
    def priority_inpaint(self):
        """
        Perform MRF-based inpainting with direction structure distribution analysis.
        
        Returns:
        - inpainted_image: the resulting inpainted image
        """
        # Convert image to grayscale for structure analysis
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Initialize the fill order priority
        self.confidence_map = 1 - self.mask.astype(float)
        
        # Compute global structure tensor
        structure_tensor, direction, coherence = self.compute_structure_tensor(gray_image)
        
        # Create a copy of the original mask for tracking progress
        original_mask_sum = np.sum(self.mask)
        
        # Iteratively fill the unknown regions
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration+1}/{self.max_iterations}")
            current_mask_sum = np.sum(self.mask)
            progress = 100 * (1 - current_mask_sum / original_mask_sum) if original_mask_sum > 0 else 100
            print(f"Progress: {progress:.2f}% completed")
            
            # Check if we're done
            if np.sum(self.mask) == 0:
                print("Inpainting complete - all masked regions filled")
                break
                
            # Find the fill front
            fill_front, front_confidence = self.find_fill_front()
            if np.sum(fill_front) == 0:
                print("No fill front pixels found")
                break
            
            # Compute the priority for each pixel on the fill front
            priority = np.zeros_like(fill_front, dtype=float)
            fill_indices = np.where(fill_front > 0)
            
            for i, j in zip(fill_indices[0], fill_indices[1]):
                # Get neighborhood around the pixel for structure analysis
                neighborhood = self.get_patch_mask(i, j)
                known_mask = (1 - self.mask) * neighborhood
                
                # Skip if no known pixels in neighborhood
                if np.sum(known_mask) == 0:
                    continue
                
                # Compute direction histogram for the known part of the neighborhood
                local_dir_hist, _ = self.compute_direction_histogram(
                    direction, coherence, known_mask)
                
                # Priority is a combination of confidence and structure coherence
                data_term = coherence[i, j]
                confidence_term = self.confidence_map[i, j]
                priority[i, j] = confidence_term * data_term
            
            # Find the pixel with the highest priority
            if np.max(priority) == 0:
                # If no priority, just pick the first pixel on the fill front
                max_i, max_j = fill_indices[0][0], fill_indices[1][0]
                print("Using default pixel selection (no priorities > 0)")
            else:
                max_idx = np.argmax(priority[fill_indices])
                max_i, max_j = fill_indices[0][max_idx], fill_indices[1][max_idx]
            
            # Compute direction histogram for the target neighborhood
            target_neighborhood = self.get_patch_mask(max_i, max_j)
            known_mask = (1 - self.mask) * target_neighborhood
            target_dir_hist, _ = self.compute_direction_histogram(
                direction, coherence, known_mask)
            
            # Find the best matching patch
            best_i, best_j, _ = self.find_best_patch(max_i, max_j, target_dir_hist)
            
            # Copy the patch
            target_patch_mask = self.get_patch_mask(max_i, max_j)
            source_patch_mask = self.get_patch_mask(best_i, best_j)
            
            # Only copy to unknown pixels in the target patch
            copy_mask = target_patch_mask & self.mask.astype(bool)
            target_indices = np.where(copy_mask)
            
            # Calculate offsets
            offset_i = best_i - max_i
            offset_j = best_j - max_j
            
            # FIX: Process each pixel individually to avoid dimension mismatch
            for idx in range(len(target_indices[0])):
                ti = target_indices[0][idx]
                tj = target_indices[1][idx]
                
                # Calculate source pixel location
                si = ti + offset_i
                sj = tj + offset_j
                
                # Check bounds
                if 0 <= si < self.image.shape[0] and 0 <= sj < self.image.shape[1]:
                    # Copy pixel-by-pixel to avoid dimension errors
                    self.image[ti, tj, :] = self.image[si, sj, :]
                    self.mask[ti, tj] = 0
                    self.confidence_map[ti, tj] = self.confidence_map[max_i, max_j]
            
            # Update structure tensor after filling a region
            if np.sum(self.mask) > 0 and (iteration % 2 == 0 or iteration == self.max_iterations - 1):
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                structure_tensor, direction, coherence = self.compute_structure_tensor(gray_image)
        
        print("Inpainting process finished")
        return self.image
    
    