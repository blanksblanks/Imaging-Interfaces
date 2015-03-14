import cv2, os, sys, time
import PIL
from PIL import Image
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

# ============================================================
# Constants
# ============================================================

BINS = 32
BINSIZE = int(256/BINS)
BLK_THRESH = 40

# ============================================================
# Helper Functions
# ============================================================

def save(image, name):
    cv2.imwrite(name, image)

def show(image, wait):
    cv2.waitKey(wait)
    cv2.imshow('Image', image)

# ============================================================
# 3D histogram
# ============================================================

def hexencode(rgb):
    r = rgb[0]*BINSIZE
    g = rgb[1]*BINSIZE
    b = rgb[2]*BINSIZE
    return '#%02x%02x%02x' % (r,g,b)

def ret_3dhistogram(image):
    colors = []
    h = len(image)
    w = len(image[0])
    print 'height and width', h, w
    hist = np.zeros(shape=(BINS, BINS, BINS)) # 8 the number of color bins
    for i in xrange(0, h): # height
        for j in xrange(0, w): # width
            pixel = image[i][j]
            if pixel[0] > BLK_THRESH and pixel[1] > BLK_THRESH and pixel[2] > BLK_THRESH:
                r_bin = pixel[0] / BINSIZE
                g_bin = pixel[1] / BINSIZE
                b_bin = pixel[2] / BINSIZE
                # 'RGB bins + 1', r_bin, g_bin, b_bin
                hist[r_bin][g_bin][b_bin] += 1
                if (r_bin,g_bin,b_bin) not in colors:
                    colors.append( (r_bin,g_bin,b_bin) )

    colors = sorted(colors, key=lambda c: -hist[(c[0])][(c[1])][(c[2])])

    for idx, c in enumerate(colors):
        r = c[0]
        g = c[1]
        b = c[2]
        # print 'c var', c
        print 'count', hist[r][g][b]
        print 'color', hexencode(c)
        plt.bar(idx, hist[r][g][b], color=hexencode(c), edgecolor=hexencode(c))
    plt.show()
    print '3d histogram:\n', hist

# ============================================================
# Main Method
# ============================================================

def main():
    if len(sys.argv) < 2:
        sys.exit("Need to specify a path from which to read images")

    imageformat=".ppm"
    path = "./" + sys.argv[1]

    images = []
    titles = []

    # load image sequence
    if os.path.exists(path):
        imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
        if len(imfilelist) < 1:
        	sys.exit ("Need to specify a path containing .ppm files")
        for el in imfilelist:
            sys.stdout.write(el)
            image = cv2.imread(el, cv2.IMREAD_UNCHANGED) # load original
            # print el, '\n', image, '\n\n'
            # pixels = list(image)
            ret_3dhistogram(image)
            # titles.append(el[9:-4])
        #     images.append(image)
        # for i in xrange(40):
        #     plt.subplot(5,8,i+1),plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)) # row, col
        #     plt.title(titles[i], size=12)
        #     plt.xticks([]),plt.yticks([])
        # plt.show()

    else:
        sys.exit("The path name does not exist")


    # print combination
    # decision = authenticate(combination)
    # print decision

    time.sleep(5)

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