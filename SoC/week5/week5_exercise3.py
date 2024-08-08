import scipy
import scipy.signal
import numpy as np
import cv2
import matplotlib.pyplot as plt

conv2d = scipy.signal.convolve2d # assigning a shorter name for this function.

# looks for horizontal edges
horizontal_edge_detector = np.array(
  [
      [-1, 0, 1]
  ]
)

box_blur_size = 15
box_blur = np.ones((box_blur_size, box_blur_size)) / (box_blur_size ** 2)
sharpen_kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0,  -1, 0]
    ]
)

all_edge_detector = np.array(
    [
        [0, -1, 0],
        [-1, 4, -1],
        [0,  -1, 0]
    ]
)

def prep_to_draw(img):
  """ Function which takes in an image and processes it to display it.
  """
  # Scale to 0,255
  prepped = img * 255
  # Clamp to [0, 255]
  prepped = np.clip(prepped, 0, 255) # clips values < 0 to 0 and > 255 to 255.
  prepped = prepped.astype(np.uint8)
  return prepped

# Write your solution below.
def imshow(image, *args, **kwargs):
    if len(image.shape) == 3:
      # Height, width, channels
      # Assume BGR, do a conversion since
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
      # Height, width - must be grayscale
      # convert to RGB, since matplotlib will plot in a weird colormap (instead of black = 0, white = 1)
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Draw the image
    plt.imshow(image, *args, **kwargs)
    # We'll also disable drawing the axes and tick marks in the plot, since it's actually an image
    plt.axis('off')
    # Make sure it outputs
    plt.show()
    
phoenix_image = cv2.imread('phoenix.jpg' )
phoenix_gray = cv2.cvtColor(phoenix_image, cv2.COLOR_BGR2GRAY)

phoenix_gray_float = phoenix_gray.astype(np.float32) / 255.0

horizontal_edge_convolved = conv2d(phoenix_gray_float, horizontal_edge_detector, mode='same', boundary='wrap')
box_blur_convolved = conv2d(phoenix_gray_float, box_blur, mode='same', boundary='wrap')
sharpen_convolved = conv2d(phoenix_gray_float, sharpen_kernel, mode='same', boundary='wrap')
all_edge_convolved = conv2d(phoenix_gray_float, all_edge_detector, mode='same', boundary='wrap')

horizontal_edge_convolved_final = prep_to_draw(horizontal_edge_convolved)
box_blur_convolved_final = prep_to_draw(box_blur_convolved)
sharpen_convolved_final = prep_to_draw(sharpen_convolved)
all_edge_convolved_final = prep_to_draw(all_edge_convolved)
imshow(horizontal_edge_convolved_final)
imshow(box_blur_convolved_final)
imshow(sharpen_convolved_final)
imshow(all_edge_convolved_final)