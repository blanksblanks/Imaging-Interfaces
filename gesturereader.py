import cv2, os, sys
import numpy as np
# import matplotlib.pyplot as plt

gray_img = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE) # default has color
cv2.imwrite('test_image_grayscale.jpg',gray_img)

def showImageWithDelay(img, wait):
	cv2.waitKey(wait)
	cv2.imshow('Image', img)

imageformat=".JPG"
path = "./" + sys.argv[1]
# path="./images"
if os.path.exists(path):
    imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
    for el in imfilelist:
        print el
        image = cv2.imread(el, cv2.IMREAD_COLOR)
        # resize image - calculate aspect ratio first
        r = 300.0 / image.shape[1]
        dim = (300, int(image.shape[0] * r))
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        # rotate image by 180 degrees - calculate center first
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 270, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        showImageWithDelay(image, 1000)
        # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        showImageWithDelay(image, 1000)
        # invert colors
        image = (255-image)
        showImageWithDelay(image, 1000)
        # find otsu's threshold value with median blurring -> bw
        blur = cv2.medianBlur(image, 5) # cv2.GaussianBlur(image,(5,5),0)
        ret,thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image = thresh
        showImageWithDelay(image, 1000)
        # apply morphological closing to close holes
        kernel = np.ones((5,5), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        showImageWithDelay(image, 1000)
        # canny edge detection: performs gaussian filter, intensity gradient,
        # non-max suppression, hysteresis thresholding all at once
        image = cv2.Canny(image,100,200) # params: min, max vals
        showImageWithDelay(image, 1000) # finally, show the image

else:
	sys.exit("The path name does not exist")

'''
plt.imshow(bgr_img, cmap = plt.get_cmap('gray'))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF    # 0xFF? To get the lowest byte.
    if k == 27: break            # Code for the ESC key

cv2.destroyAllWindows()
'''

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