import numpy as np
import cv2

def apply_custom_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies a given 2D filter/kernel to a grayscale or RGB image.
    Uses cv2.filter2D for execution speed while proving the fundamental concept.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # For RGB imagery, filter applies across all channels independently
        pass
    elif len(image.shape) != 2:
        raise ValueError("Image must be 2D array (grayscale) or 3D array (RGB).")
        
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

def get_sobel_kernels():
    """Returns horizontal and vertical Sobel edge-detection kernels."""
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=np.float32)
                        
    sobel_y = np.array([[-1, -2, -1], 
                        [ 0,  0,  0], 
                        [ 1,  2,  1]], dtype=np.float32)
    return sobel_x, sobel_y

def get_gaussian_blur_kernel():
    """Returns a simplified 3x3 Gaussian Blur kernel."""
    return np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]], dtype=np.float32) / 16.0
