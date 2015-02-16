'''
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
'''

'''def main():
    # sequence = map(str, raw_input('Please enter your sequence ').split(' '))
    img = cv2.imread('fist-center.jpg',0)
    k = cv2.waitKey(0)

    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messigray.png',img)
        cv2.destroyAllWindows()

if __name__ == "__main__": main()

'''

import cv2
import numpy as np
import os
# import matplotlib.pyplot as plt

gray_img = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE) # default has color
cv2.imwrite('test_image_grayscale.jpg',gray_img)

imageformat=".JPG"
path="./images"
imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
for el in imfilelist:
        print el
        image = cv2.imread(el, cv2.CV_LOAD_IMAGE_COLOR)
        cv2.imshow('Image', image) #Show the image
        cv2.waitKey(1000)
'''
plt.imshow(bgr_img, cmap = plt.get_cmap('gray'))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF    # 0xFF? To get the lowest byte.
    if k == 27: break            # Code for the ESC key

cv2.destroyAllWindows()
'''