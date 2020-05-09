import scipy.misc
import numpy as np
from matplotlib.image import imread
import matplotlib
import cv2
from PIL import Image

def pad_with(vector, pad_width, iaxis, kwargs):
     pad_value = kwargs.get('padder', 255.0)
     vector[:pad_width[0]] = pad_value
     vector[-pad_width[1]:] = pad_value

image_paths = ["images/mask.png", "images/MASK1.png", "images/sand.jpg"]
dest_image_path = ["images/mask_white.png","images/MASK1_white.png","images/sand_white.jpg"]


for i,image_path in enumerate(image_paths):
    img = cv2.imread(image_path)    
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage.fill(0.0)
    #print(image_path, " --- B --- ", grayImage.shape)
    grayImage = np.pad(grayImage, 40, pad_with)
    #print(image_path, " --- A --- ", grayImage.shape)
    cv2.imwrite(dest_image_path[i], grayImage)
    


for image_path in image_paths:
    img = cv2.imread(image_path)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = grayImage * 0.5
    grayImage = np.pad(grayImage, 40, pad_with)
    cv2.imwrite(image_path, grayImage)
    
from PIL import Image

im = Image.open('images/citymap.png')
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

im_new = add_margin(im, 40, 40, 40, 40, (255, 255, 255))
im_new.save('images/citymap.png', quality=100)


