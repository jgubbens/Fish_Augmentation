import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

def __init__(self, displacement_strength = 30, displacement_density = 5):
    self.displacement_strength = displacement_strength
    self.displacement_density = displacement_density

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]  # (height, width)

    # Random affine transformation
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, 
                       [center_square[0] + square_size, center_square[1] - square_size], 
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # Reduce the randomness and increase the deformation intensity
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)  # No deformation along the color axis for RGB images

    # 3D meshgrid for RGB images
    if len(shape) == 3:
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    else:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        z = np.zeros_like(x)

    # Adjust the displacement generation to create fewer but larger displacements
    displacement_strength = 30
    displacement_density = 5
    
    # Generate sparse displacement field: fewer large displacements
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma * displacement_density) * alpha * displacement_strength
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma * displacement_density) * alpha * displacement_strength
    
    # Create new coordinates for the deformation
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    # Map the coordinates to the original image and apply the transformation
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

# Load image
image = cv2.imread('ElasticDeformation/original.png', cv2.IMREAD_COLOR)

# Apply elastic transformation with larger, fewer deformations
transformed_image = elastic_transform(image, alpha=100, sigma=9, alpha_affine=10)

cv2.imwrite('ElasticDeformation/transformed.png', transformed_image)

# Display the original and transformed images
cv2.imshow("Original Image", image)
cv2.imshow("Elastic Transformed Image", transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
