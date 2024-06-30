#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing
"""

import math

from PIL import Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, row, col, boundary_behavior = "zero"):             #row and col start numbering from 0
    """
    Given row index and column index, this function
    returns the pixel from image["pixels"] 
    based on the boundary behavior
    """
    if len(image["pixels"]) != image["height"] * image["width"]:
        raise ValueError("The size of the pixels list does not match the dimensions of the image.")
    if row in range(0, image["height"]) and col in range(0, image["width"]):
        return image["pixels"][(row*image["width"]) + col]
    
    if boundary_behavior == "zero":
        return 0
    
    if boundary_behavior == "extend":
        if row in range(0, image["height"]) and col >= image["width"]:
            #return image["pixels"][row*image["width"] + image["width"] - 1]
            return get_pixel(image, row, image["width"] - 1)
        if row in range(0, image["height"]) and col < 0:
            #return image["pixels"][row*image["width"]]
            return get_pixel(image, row, 0)
        
        if col in range(0, image["width"]) and row >= image["height"]:
            return get_pixel(image, image["height"] - 1, col)
        if col in range(0, image["width"]) and row < 0:
            return get_pixel(image, 0, col)
        
        if row < 0 and col < 0:
            return get_pixel(image, 0, 0)
        if row < 0 and col >= image["width"]:
            return get_pixel(image, 0, image["width"] - 1)
        if row >= image["height"] and col < 0:
            return get_pixel(image, image["height"] - 1, 0)
        if row >= image["height"] and col >= image["width"]:
            return get_pixel(image, image["height"] - 1, image["width"] - 1)
        
    if boundary_behavior == "wrap":
        return get_pixel(image, row%image["height"], col%image["width"])


def set_pixel(image, row, col, color):
    image["pixels"][row*image["width"] + col] = color


def apply_per_pixel(image, func):
    
    """
    Apply the given function to each pixel in an image and return the resulting image.

    Parameters:
    image (dict): A dictionary representing an image, with keys:
                    - "height" (int): The height of the image.
                    - "width" (int): The width of the image.
                    - "pixels" (list): A list of pixel values.
    func (callable): A function that takes a single pixel value and returns a new pixel value.

    Returns:
    dict: A new image dictionary with the same dimensions as the input image, but with each
            pixel value modified by the given function.
    """
    new_pixels = [func(x) for x in image["pixels"]]
    new_image = {
        "height" : image["height"],
        "width" : image["width"],
        "pixels" : new_pixels
    }

    return new_image    
    #raise NotImplementedError


def inverted(image):
    return apply_per_pixel(image, lambda color: 255-color)  #lambda is used to define a function


# HELPER FUNCTIONS

def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.
    
    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE

    The kernel will be represented as a dict having keys "size" describing the
    side length of the square kernel, and "values" which is a list whose values are 
    the values of each pixel of the kernel in row major order.
    """
    if boundary_behavior not in ("zero", "extend", "wrap"):
        return None
    new_image = {}
    new_image["height"] = image["height"]
    new_image["width"] = image["width"]
    new_image["pixels"] = []
    midvalue = (kernel["size"] - 1)*0.5
    midvalue = int(midvalue)
    for row in range(0, image["height"]):
        for col in range(0, image["width"]):
            i = 0
            sum = 0
            #while i < (kernel["size"])*(kernel["size"]):
            for rowdash in range(row - midvalue, row + midvalue + 1):
                for coldash in range(col - midvalue, col + midvalue + 1):
                    sum += get_pixel(image, rowdash, coldash, boundary_behavior)*kernel["values"][i]
                    i += 1
            new_image["pixels"].append(sum)

    return new_image
    #raise NotImplementedError


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for i in range(0, image["height"]*image["width"]):
        if image["pixels"][i] > 255: 
            image["pixels"][i] = 255
        elif image["pixels"][i] < 0:
            image["pixels"][i] = 0
        else:
            image["pixels"][i] = round(image["pixels"][i])
    return None   
    #raise NotImplementedError



# FILTERS


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    kernel = {}
    kernel["size"] = kernel_size
    value = 1/(kernel_size*kernel_size)
    #value = round(value, 3)
    kernel["values"] = [value]*(kernel_size*kernel_size)
    
    new_image = correlate(image, kernel, "extend")
    round_and_clip_image(new_image)

    return new_image
    #raise NotImplementedError

def sharpened(image, n):
    """
    Return a new image which is sharper than the input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    Output list should be clipped.
    """
    kernel = {}
    kernel["size"] = n
    value = 1/(n*n)
    kernel["values"] = [value]*(n*n)
    
    blur_image = correlate(image, kernel, "extend")
    
    final_image = {}
    final_image["height"] = image["height"]
    final_image["width"] = image["width"]
    final_image["pixels"] = []
    for i in range(0, image["height"]*image["width"]):
        final_image["pixels"].append(2*image["pixels"][i] - blur_image["pixels"][i])
    round_and_clip_image(final_image)

    return final_image
    #raise NotImplementedError

def edges(image):
    """
    Return a new image with all the edges distincly and clearly detectable.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    Output list should be clipped.
    """
    #First, defining K1:
    kernel_1 = {}
    kernel_1["size"] = 3
    kernel_1["values"] = [-1, -2, -1,
                           0, 0, 0,
                           1, 2, 1]
    #Second, defining K2:
    kernel_2 = {}
    kernel_2["size"] = 3
    kernel_2["values"] = [-1, 0, 1,
                          -2, 0, 2,
                          -1, 0, 1]
    #Defining O1 and O2:
    output_1 = correlate(image, kernel_1, "extend")
    output_2 = correlate(image, kernel_2, "extend")

    #Final image:
    final_image = {}
    final_image["height"] = image["height"]
    final_image["width"] = image["width"]
    final_image["pixels"] = []
    for i in range(0, len(image["pixels"])):
        final_image["pixels"].append(pow((output_1["pixels"][i]**2 + output_2["pixels"][i]**2),0.5))
    
    round_and_clip_image(final_image)
    return final_image
    #raise NotImplementedError

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the "mode" parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    #pass
    im = load_greyscale_image("C:/Users/prith/XYZ/SoC/test_images/mushroom.png")
    #save_greyscale_image(im, "C:/Users/prith/XYZ/SoC/test_images/mushroom_blacknwhite.png")
    mushroom_edges = edges(im)
    save_greyscale_image(mushroom_edges, r"C:/Users/prith/XYZ/SoC/test_images/mushroom_edges.png")