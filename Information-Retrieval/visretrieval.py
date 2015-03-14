import cv2, os, sys, time
import PIL
from PIL import Image
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

# ============================================================
# Constants
# ============================================================

BINS = 8
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
                r_bin = pixel[2] / BINSIZE # openCV loads as BGR
                g_bin = pixel[1] / BINSIZE
                b_bin = pixel[0] / BINSIZE
                # 'RGB bins + 1', r_bin, g_bin, b_bin
                hist[r_bin][g_bin][b_bin] += 1
                if (r_bin,g_bin,b_bin) not in colors:
                    colors.append( (r_bin,g_bin,b_bin) )
                    # colors.append( (pixel[0],pixel[1],pixel[2]) )
    colors = sorted(colors, key=lambda c: -hist[(c[0])][(c[1])][(c[2])])

    for idx, c in enumerate(colors):
        r = c[0]#/BINSIZE
        g = c[1]#/BINSIZE
        b = c[2]#/BINSIZE
        # print 'c var', c
        print 'count', hist[r][g][b]
        print 'color', hexencode(c)
        plt.subplot(1,2,1).bar(idx, hist[r][g][b], color=hexencode(c), edgecolor=hexencode(c))
        plt.xticks([]),plt.xlabel('color bins'),plt.ylabel('frequency count')
        plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.xticks([]),plt.yticks([])
    # plt.show()
    # print '3d histogram:\n', hist
    return hist

def l1_color_norm(h1, h2):
    diff = 0
    sum = 0
    for r in xrange(0, BINS):
        for g in xrange(0, BINS):
            for b in range(0, BINS):
                diff += abs(h1[r][g][b] - h2[r][g][b])
                sum += h1[r][g][b] + h2[r][g][b]
    distance = diff / 2.0 / sum
    print 'diff, count and distance:', diff, sum, distance
    return distance

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
    chists = []
    cresults = []

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
            chist = ret_3dhistogram(image)
            titles.append(el[9:-4])
            images.append(image)
            chists.append(chist)
        # for i in xrange(40):
        #     plt.subplot(5,8,i+1),plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)) # row, col
        #     plt.title(titles[i], size=12)
        #     plt.xticks([]),plt.yticks([])
        # plt.show()

    else:
        sys.exit("The path name does not exist")

    # deterine 4 closest and 4 farthest matches for all images
    for k in xrange(len(imfilelist)):
        results = {} # reinitialize
        cbest = []
        cworst = []
        for i in xrange(0, len(imfilelist)):
            d = l1_color_norm(chists[k], chists[i]);
            results[i] = d;

        results = sorted([(v, k) for (k, v) in results.items()])
        print results
        # return a list of tuples (similarity, index)
        # ordered from most similar to least similar
        # ignore first value - it will be the same image as the one being compared to

        for i in xrange(0,4):
            b = (results[i])[1]
            cbest.append(b)
            if i > 0:
                w = (results[-i])[1]
                cworst.append(w)
            # print "b,w: ", b,w

        print "cbest, cworst", cbest, cworst
        results = cbest
        results.extend(cworst)
        print "results", results
        cresults.extend(results)
        print "cresults", cresults

    for i in xrange(7*10): # only do first 10 images for now
        index = cresults[i]
        plt.subplot(10,7,i+1),plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB)) # row, col
        if index > 8:
            index = 'i' + str(index+1)
        else:
            index = 'i0' + str(index+1)
        plt.title(index, size=12)
        plt.xticks([]),plt.yticks([])
    plt.show()


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