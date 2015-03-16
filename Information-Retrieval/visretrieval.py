import os
import sys
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from PIL import Image

# import PIL
# from numpy import linalg as la


# ============================================================
# Constants
# ============================================================

NUM_IM = 40
COL_RANGE = 256
BLK_THRESH = 40
BINS = 8
BIN_SIZE = int(COL_RANGE/BINS)
IMG_H = 60
IMG_W = 89

# ============================================================
# Helper Functions
# ============================================================

def save(image, name):
    cv2.imwrite(name, image)

def show(image, wait):
    cv2.waitKey(wait)
    cv2.imshow('Image', image)

def display_all(images, titles):
    for i in xrange(NUM_IM):
        plt.subplot(5,8,i+1),plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)) # row, col
        plt.title(titles[i], size=12)
        plt.xticks([]),plt.yticks([])
    plt.show()

def visualize_hist(image, hist, colors, title):
    colors = sorted(colors, key=lambda c: -hist[(c[0])][(c[1])][(c[2])])
    plt.rcParams['font.family']='Aller Light'
    for idx, c in enumerate(colors):
        r = c[0]
        g = c[1]
        b = c[2]
        # print 'color, count:', hexencode(c, BIN_SIZE), hist[r][g][b]
        plt.subplot(1,2,1).bar(idx, hist[r][g][b], color=hexencode(c, BIN_SIZE), edgecolor=hexencode(c, BIN_SIZE))
        plt.xticks([])
        # plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.xticks([]),plt.yticks([])
    dir_name = './color_hist/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    title = dir_name+title+'.png'
    plt.savefig(title, bbox_inches='tight')
    plt.clf()
    plt.close('all')
    print title
    plot = cv2.imread(title, cv2.IMREAD_UNCHANGED)
    return plot
    # show(plot, 100)
    # clear image

    # print '3d histogram:\n', hist

def pic_stitch(series, images, titles):
    n = len(series)
    # for k in xrange(0, n, 7):
    #     title = titles[k]
    #     for i in xrange(1, 7):
    #         idx = series[i]
    #         if i == 1:
    #             prior = series[i-1]
    #             img = np.concatenate((images[prior], images[idx]), axis=1)
    #         else:
    #             img = np.concatenate((img, images[idx]))
    #     dir_name = './color_matches/'
    #     if not os.path.exists(dir_name):
    #         os.makedirs(dir_name)
    #     cv2.imwrite(dir_name+title+'.png', img)


    im = Image.new('RGB',(IMG_W*7,IMG_H))
    for i in xrange(n):
        title = titles[i]
        idx = series[i]
        img = images[i] # Image.open('./images/'+title+'.ppm')
        im.paste(img, (i*IMG_W,0))
    dir_name = './color_matches/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    im.save(dir_name+title+'.jpg','JPEG')

def septuple_stitch(images, chist_images, titles, dir_name, cresults, cdistances):
    plt.rcParams['font.family']='Aller Light'
    gs1 = gridspec.GridSpec(2,7)
    gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
    for k in xrange(0, NUM_IM*7, 7):
        for i in xrange(7):
            idx = cresults[k+i]
            ax = plt.subplot(gs1[i])
            ax.imshow(cv2.cvtColor(chist_images[idx], cv2.COLOR_BGR2RGB))
            ax.set_xticks([]),ax.set_yticks([])

            ax = plt.subplot(gs1[i+7])
            plt.axis('on')
            # plt.subplot(10,7,i+1)
            plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)) # row, col
            plt.title(titles[idx], size=12)
            # plt.title(titles[idx], size=12)
            plt.xticks([]),plt.yticks([])
            if i == 0:
                plt.xlabel('similarity:')
            else:
                sim = 1 - round(cdistances[k+i], 5)
                plt.xlabel(sim)
            ax.set_aspect('equal')

        title = titles[k/7]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        title = dir_name+title+'.png'
        plt.savefig(title, bbox_inches='tight')
        print title
        # plt.savefig('firstseries.png', bbox_inches='tight')
        # plt.show()
        plt.clf()
        plt.close('all')


# ============================================================
# Gross Color Matching
# ============================================================

def hexencode(rgb, factor):
    """Convert RGB tuple to hexadecimal color code."""
    r = rgb[0]*factor
    g = rgb[1]*factor
    b = rgb[2]*factor
    return '#%02x%02x%02x' % (r,g,b)

def color_histogram(image, title):
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
    plot = visualize_hist(image, hist, colors, title)
    return hist, plot

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

def calc_distance(chists):
    chist_dis = {}
    for i in xrange(NUM_IM):
        for j in xrange(i, NUM_IM):
            if (i,j) not in chist_dis:
                d = l1_color_norm(chists[i], chists[j])
                chist_dis[(i,j)] = d
    print chist_dis
    return chist_dis


def color_matches(k, chist_dis):
    """Find images most like and unlike an image based on color distribution.
    k -- the original image for comparison
    chists -- the list of color histograms for analysis
    """
    results = {}
    indices = []
    distances = []
    bestdiff = 0
    worstdiff = 0

    for i in xrange(0, NUM_IM):
        if k > i: # because tuples always begin with lower index
            results[i] = chist_dis[(i,k)]
        else:
            results[i] = chist_dis[(k,i)]
    # Ordered list of tuples (dist, idx) from most to least similar
    # -- first value will be the original image with diff of 0
    results = sorted([(v, k) for (k, v) in results.items()])
    # print 'results for image', k, results
    seven = results[:4]
    seven.extend(results[-3:])
    # print 'last seven for image', k, seven

    distances, indices = zip(*seven)
    # print 'distances:',distances
    # print 'indices:',indices
    bestdiff = reduce(lambda x, y: x+y, distances[:4])
    worstdiff = reduce(lambda x, y: x+y, distances[-3:])

    # for i in xrange(0,4):
    #     b = (results[i])
    #     cbest.append(b[1])
    #     cbestd.append(b[0])
    #     bestdiff += b[0]
    #     if i > 0:
    #         w = (results[-i])
    #         cworst.append(w[1])
    #         worstdiff += w[0]

    # List of indices for original, 3 most, and 3 least similar image(s)
    # indices.extend(cbest)
    # indices.extend(cworst)
    # print "cbest, cworst", cbest, cworst
    # print "indices", indices
    return indices, distances, bestdiff, worstdiff

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
    chist_images = []
    cresults = []
    cdistances = []
    like = 1
    unlike = 0

    # load image sequence
    if os.path.exists(path):
        imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(format)]
        if len(imfilelist) < 1:
        	sys.exit ("Need to specify a path containing .ppm files")
        NUM_IM = len(imfilelist)
        for el in imfilelist:
            print(el)
            image = cv2.imread(el, cv2.IMREAD_UNCHANGED)
            title = el[9:-4]
            chist, plot = color_histogram(image, title)
            titles.append(title)
            images.append(image)
            chists.append(chist)
            chist_images.append(plot)
    else:
        sys.exit("The path name does not exist")

    # calculate lookup table for distances between image color histograms
    chist_dis = calc_distance(chists)

    # determine 4 closest and 4 farthest matches for all images
    for k in xrange(NUM_IM):
        results, distances, bestdiff, worstdiff = color_matches(k, chist_dis)
        cresults.extend(results)
        cdistances.extend(distances)
        if bestdiff < like:
            like = k
        if worstdiff > unlike:
            unlike = k

    print "Gross color matching results:", cresults
    print "Most like, most unlike:", like, unlike

    # for i in xrange(NUM_IM):
    #     pic_stitch(cresults[i], images, titles)
    septuple_stitch(images, chist_images, titles, './color_sim_hist/', cresults, cdistances)
    # septuple_stitch(chist_images, titles, './color_hist_sim/', cresults, cdistances)


if __name__ == "__main__": main()
