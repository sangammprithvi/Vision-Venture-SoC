import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
# Let's download the image and video we'll be using!
#import requests

phoenix_image = cv2.imread('phoenix.jpg' )
phoenix_rgb = cv2.cvtColor(phoenix_image, cv2.COLOR_BGR2RGB)
phoenix_gray = cv2.cvtColor(phoenix_image, cv2.COLOR_BGR2GRAY)
#imshow(phoenix_image)
#imshow(phoenix_gray)
#imshow(phoenix_rgb)

empty_arr = np.zeros(phoenix_gray.shape, dtype = np.uint8)

magenta_phoenix = np.stack([ phoenix_gray, empty_arr, phoenix_gray, ], axis=2)
print("Created image of shape",magenta_phoenix.shape)
#imshow(magenta_phoenix)

bigger_magenta_phoenix = cv2.resize(magenta_phoenix, (720, 720))
imshow(bigger_magenta_phoenix)

output_path = "./output_pinkphoenix.png"
cv2.imwrite(output_path, bigger_magenta_phoenix)

test_read_output = cv2.imread(output_path)
print("Read file of shape:",test_read_output.shape, "type",test_read_output.dtype)
imshow(test_read_output)

# function to crop a given frame
def crop_frame(frame, crop_size):
  # We're given a frame, either gray or RGB, and a crop-size (w,h)
  crop_w, crop_h = crop_size
  # This is an array! We can slice it
  # Take the first pixels along the height, and along the width
  cropped = frame[:crop_h, :crop_w]
  return cropped

capture = cv2.VideoCapture('sample_video.mp4')

print(capture)

crop_size = (600,400) # (w,h)
output_path = 'output_cropped.mp4'
# Use the MJPG format
output_format = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
output_fps = 30
cropped_output = cv2.VideoWriter(output_path, output_format, output_fps, crop_size)

n = 0
while True:
  successful, next_frame = capture.read()
  if not successful:
    # No more frames to read
    print("Processed %d frames" % n)
    break
  # We have an input frame. Use our function to crop it.
  output_frame = crop_frame(next_frame, crop_size)
  # Write the output frame to the output video
  cropped_output.write(output_frame)
  n += 1

  # We have to give up the file at the end.
capture.release()
cropped_output.release()