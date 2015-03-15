import os
import sys
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

# import PIL
# from PIL import Image
# from numpy import linalg as la


# ============================================================
# Constants
# ============================================================

NUM_IM = 40
COL_RANGE = 256
BLK_THRESH = 40
BINS = 8
BIN_SIZE = int(COL_RANGE/BINS)

# ============================================================
# Helper Functions
# ============================================================

def save(image, name):
    cv2.imwrite(name, image)

def show(image, wait):
    cv2.waitKey(wait)
    cv2.imshow('Image', image)

# NO LONGER NEEDED
def entitle(idx):
    """Convert image index and return image title."""
    if idx > 8:
        idx = 'i' + str(idx+1)
    else:
        idx = 'i0' + str(idx+1)
    return idx

def display_all(images, titles):
    for i in xrange(NUM_IM):
        plt.subplot(5,8,i+1),plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)) # row, col
        plt.title(titles[i], size=12)
        plt.xticks([]),plt.yticks([])
    plt.show()

def visualize_hist(image, hist, colors):
    colors = sorted(colors, key=lambda c: -hist[(c[0])][(c[1])][(c[2])])
    for idx, c in enumerate(colors):
        r = c[0]
        g = c[1]
        b = c[2]
        # print 'color, count:', hexencode(c, BIN_SIZE), hist[r][g][b]
        plt.subplot(1,2,1).bar(idx, hist[r][g][b], color=hexencode(c, BIN_SIZE), edgecolor=hexencode(c, BIN_SIZE))
        plt.xticks([]),plt.xlabel('color bins'),plt.ylabel('frequency count')
        plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.xticks([]),plt.yticks([])
    # plt.show()
    # print '3d histogram:\n', hist


# ============================================================
# Gross Color Matching
# ============================================================

def hexencode(rgb, factor):
    """Convert RGB tuple to hexadecimal color code."""
    r = rgb[0]*factor
    g = rgb[1]*factor
    b = rgb[2]*factor
    return '#%02x%02x%02x' % (r,g,b)

def color_histogram(image):
    colors = []
    h = len(image)
    w = len(image[0])
    hist = np.zeros(shape=(BINS, BINS, BINS))
    for i in xrange(h):
        for j in xrange(w):
            pixel = image[i][j]
            if pixel[0] > BLK_THRESH and pixel[1] > BLK_THRESH and pixel[2] > BLK_THRESH:
                r_bin = pixel[2] / BIN_SIZE # OpenCV loads as BGR
                g_bin = pixel[1] / BIN_SIZE
                b_bin = pixel[0] / BIN_SIZE
                hist[r_bin][g_bin][b_bin] += 1
                if (r_bin,g_bin,b_bin) not in colors:
                    colors.append( (r_bin,g_bin,b_bin) )
    visualize_hist(image, hist, colors)
    return hist

def l1_color_norm(h1, h2):
    diff = 0
    total = 0
    for r in xrange(0, BINS):
        for g in xrange(0, BINS):
            for b in range(0, BINS):
                diff += abs(h1[r][g][b] - h2[r][g][b])
                total += h1[r][g][b] + h2[r][g][b]
    distance = diff / 2.0 / total
    similarity = 1 - distance
    # print 'diff, sum and distance:', diff, sum, distance
    return distance

def color_matches(k, chists):
    """Find images most like and unlike an image based on color distribution.
    k -- the original image for comparison
    chists -- the list of color histograms for analysis
    """
    results = {}
    cbest = []
    cworst = []

    for i in xrange(0, NUM_IM):
        d = l1_color_norm(chists[k], chists[i]);
        results[i] = d;
    # Ordered list of tuples (dist, idx) from most to least similar
    # -- first value will be the original image with diff of 0
    results = sorted([(v, k) for (k, v) in results.items()])
    # print results

    for i in xrange(0,4):
        b = (results[i])[1]
        cbest.append(b)
        if i > 0:
            w = (results[-i])[1]
            cworst.append(w)

    # List of indices for original, 3 most, and 3 least similar image(s)
    results = cbest
    results.extend(cworst)
    # print "cbest, cworst", cbest, cworst
    # print "results", results
    return results

# ============================================================
# Main Method
# ============================================================

def main():

    if len(sys.argv) < 2:
        sys.exit("Need to specify a path from which to read images")

    format = ".ppm"
    path = "./" + sys.argv[1]

    images = []
    titles = []
    chists = []
    cresults = []

    # load image sequence
    if os.path.exists(path):
        imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(format)]
        if len(imfilelist) < 1:
        	sys.exit ("Need to specify a path containing .ppm files")
        NUM_IM = len(imfilelist)
        for el in imfilelist:
            print(el)
            image = cv2.imread(el, cv2.IMREAD_UNCHANGED)
            chist = color_histogram(image)
            titles.append(el[9:-4])
            images.append(image)
            chists.append(chist)
    else:
        sys.exit("The path name does not exist")

    # determine 4 closest and 4 farthest matches for all images
    for k in xrange(NUM_IM):
        results = color_matches(k, chists)
        cresults.extend(results)

    print "Gross color matching results", cresults

    for i in xrange(7*10): # only do first 10 images for now
        index = cresults[i]
        plt.subplot(10,7,i+1),plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB)) # row, col
        plt.title(titles[index], size=12)
        plt.xticks([]),plt.yticks([])
    plt.savefig('foo.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__": main()
