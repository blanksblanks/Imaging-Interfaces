import cv2, os, sys
import numpy as np
# import matplotlib.pyplot as plt

# resize image to 300 x 300 pixels
def resize(image):
    r = 300.0 / image.shape[1] # calculate aspect ratio
    dim = (300, int(image.shape[0] * r))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

# rotate image 90 degrees counterclockwise
def rotate(image):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2) # find center
    M = cv2.getRotationMatrix2D(center, 270, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    show(image, 1000)
    return image

# convert color image to grayscale
def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show(image, 1000)
    return image

# invert image colors
def invert(image):
    image = (255-image)
    show(image, 1000)
    return image

# find otsu's threshold value with median blurring to make image black and white
def binarize(image):
    blur = cv2.medianBlur(image, 5) # better for spotty noise than cv2.GaussianBlur(image,(5,5),0)
    ret,thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = thresh
    show(image, 1000)
    return image

def close(image):
    # apply morphological closing to close holes
    kernel = np.ones((5,5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    show(image, 1000)
    return image

def findEdges(image):
    # canny edge detection: performs gaussian filter, intensity gradient,
    #non-max suppression, hysteresis thresholding all at once
    image = cv2.Canny(image,100,200) # params: min, max vals
    show(image, 1000)
    return image

# find contours and convex hull
def contourify(image):
    temp = image # store because findContours modifes source
    contours,hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0,255,0), 3)

    cnt = contours[0]
    M = cv2.moments(cnt)
    # print M

    # check curve for convexity defects and correct it
    # points[]: contours passed in
    # hull[]: output
    # clockwise[]: true or false
    # returnPoints: if false, return indices of contour points
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull) # array

    image = temp # revert to original

    # cv2.drawContours(image, contours, 0, (0,255,0), 5)

    # find centroid
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = (cx, cy)
    cv2.circle(image, centroid, 20, (255,255,0), -1)

    if len(hull) > 3 and len(cnt) > 3:
        if (defects is not None):
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(image,start,end,[0,255,0],2)
                cv2.circle(image,far,5,[0,0,255],-1)
    # cnt = contours[4]
    # cv2.drawContours(image, [cnt], -1, (255,0,0), 3)
    show(image, 1000)
    return image

def show(image, wait):
    cv2.waitKey(wait)
    cv2.imshow('Image', image)

def main():
    imageformat=".JPG"
    path = "./" + sys.argv[1]

    # load image sequence
    if os.path.exists(path):
        imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
        for el in imfilelist:
            print el
            # load original image
            image = cv2.imread(el, cv2.IMREAD_COLOR)
            image = resize(image)
            image = rotate(image)
            image = grayscale(image)
            image = binarize(image)
            image = close(image)
            image = contourify(image)
    else:
        sys.exit("The path name does not exist")

if __name__ == "__main__": main()

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