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
NUM_CLUSTERS = 7
COL_RANGE = 256
BLK_THRESH = 40
BINS = 8
BIN_SIZE = int(COL_RANGE/BINS)
IMG_H = 60
IMG_W = 89
LAP_BINS = 128
LAP_BIN_SZ = int(COL_RANGE*8)/LAP_BINS


# ============================================================
# Gross Color Matching
# ============================================================

def hexencode(rgb, factor):
    """Convert RGB tuple to hexadecimal color code."""
    r = rgb[0]*factor
    g = rgb[1]*factor
    b = rgb[2]*factor
    return '#%02x%02x%02x' % (r,g,b)

def visualize_chist(image, hist, colors, title):
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
    # plot = visualize_chist(image, hist, colors, title)
    return hist

def l1_color_norm(h1, h2):
    diff = 0
    total = 0
    for r in xrange(0, BINS):
        for g in xrange(0, BINS):
            for b in range(0, BINS):
                diff += abs(h1[r][g][b] - h2[r][g][b])
                total += h1[r][g][b] + h2[r][g][b]
    l1_norm = diff / 2.0 / total
    similarity = 1 - l1_norm
    # print 'diff, sum and distance:', diff, sum, distance
    return l1_norm

def calc_cdistance(chists):
    chist_dis = {}
    for i in xrange(NUM_IM):
        for j in xrange(i, NUM_IM):
            if (i,j) not in chist_dis:
                d = l1_color_norm(chists[i], chists[j])
                chist_dis[(i,j)] = d
    # print chist_dis
    return chist_dis

def color_matches(k, chist_dis):
    """
    Find images most like and unlike an image based on color distribution.
    k -- the original image for comparison
    chists -- the list of color histograms for analysis
    """
    results = {}
    indices = []
    distances = []

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
    return indices, distances

def find_four(chist_dis):
    results = {}
    for a in xrange(NUM_IM):
        for b in xrange(NUM_IM): #TODO: change to (a,NUM_IM)
            for c in xrange(NUM_IM):
                for d in xrange(NUM_IM):
                    if (a<b and b<c and c<d):
                        results[(a,b,c,d)] = chist_dis[(a,b)] + chist_dis[(a,c)] + chist_dis[(a,d)] + chist_dis[(b,c)] + chist_dis[(b,d)] + chist_dis[(c,d)]
    results = sorted([(v, k) for (k, v) in results.items()])
    best = results[0]
    worst = results[-1]
    indices = list(best[1])
    indices.extend(list(worst[1]))
    # print "results: ", len(results), results
    # print "best, worst", best, worst
    return indices

# ============================================================
# Gross Texture Matching
# ============================================================

def grayscale(image, title):
    """Convert image into black and white with (R+G+B)/3"""
    gray = np.zeros(shape=(IMG_H, IMG_W))
    for row in xrange(IMG_H):
        for col in xrange(IMG_W):
            # compose gray image pixel by pixel
            pixel = image[row][col]
            rgb = int(pixel[0]) + int(pixel[1]) + int(pixel[2])
            g = int(round((rgb / 3.0),0)) # round up
            gray[row][col] = g
    # gray_img = cv2.imread(gray, cv2.IMREAD_UNCHANGED)
    cv2.imwrite('./gray/'+title+'.png', gray)
    return gray

def laplacian(gray, title):
    """Make laplacian image by calculating sum of neighbors for every pixel"""
    # gray = grayscale(image)
    laplacian = np.zeros(shape=(IMG_H, IMG_W))
    # laplacian.fill(COL_RANGE) # white would be 255+255+255/3 = 255
    for i in xrange(0,IMG_H):
        for j in xrange(0,IMG_W):
            # if gray[i][j] > int(BLK_THRESH/2):
            if gray[i][j]:
                neighbors = []
                for row in xrange(i-1,i+2):
                    for col in xrange(j-1,j+2):
                        if not (row is i and col is j) and index_valid(row,col):
                            neighbors.append(gray[row][col])
                neighbor_sum = reduce(lambda x, y: x+y, neighbors)
                n = len(neighbors)
                laplacian[i][j] = gray[i][j]*n - neighbor_sum
                # print 'calculations per pixel:', laplacian[i][j], gray[i][j], n, neighbor_sum
    # print 'laplacian',laplacian
    cv2.imwrite('./laplacian/'+title+'.png', laplacian)
    return laplacian

def index_valid(row,col):
    if (row < 0 or col < 0 or row >= IMG_H or col >= IMG_W):
        return False
    return True

def visualize_thist(image, hist, bins, title):
    # bins = sorted(bins, key=lambda c: -hist[c])
    plt.rcParams['font.family']='Aller Light'
    for idx, c in enumerate(bins):
        plt.subplot(1,2,1).bar(c, hist[c], color='#D3D3D3')
    dir_name = './texture_hist/'
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

def texture_histogram(lap, gray, title):
    hist = np.zeros(shape=LAP_BINS)
    bins = []
    for i in range(IMG_H):
        for j in range(IMG_W):
            pixel = lap[i][j]
            # Only count pixels outside the black threshold
            # to avoid image similarity due to shared black background
            if gray[i][j] > BLK_THRESH:
                if pixel < 256:
                    bin = abs(pixel/LAP_BIN_SZ)
                    hist[bin] += 1
                    if bin not in bins:
                        bins.append(bin)
    # plot = visualize_thist(gray, hist, bins, title)
    # print 'hist', hist
    return hist#, plot

def l1_texture_norm(h1, h2):
    diff = 0
    total = 0
    for bin in xrange(0,LAP_BINS):
        diff += abs(h1[bin] - h2[bin])
        total += h1[bin] + h2[bin]
    l1_norm = diff / 2.0 / total
    similarity = 1 - l1_norm
    return l1_norm

def calc_tdistance(thists):
    thist_dis = {}
    for i in xrange(NUM_IM):
        for j in xrange(i, NUM_IM):
            if (i,j) not in thist_dis:
                d = l1_texture_norm(thists[i], thists[j])
                thist_dis[(i,j)] = d
    return thist_dis

def texture_matches(k, thist_dis):
    results = {}
    indices = []
    distances = []
    for i in xrange(0, NUM_IM):
        if k > i: # because tuples always begin with lower index
            results[i] = thist_dis[(i,k)]
        else:
            results[i] = thist_dis[(k,i)]
    results = sorted([(v, k) for (k, v) in results.items()])
    seven = results[:4]
    seven.extend(results[-3:])
    distances, indices = zip(*seven)
    return indices, distances

# ============================================================
# Combine Similarities and Cluster
# ============================================================

def combine_similarities(chist_dis, thist_dis):
    similarities = {}
    distances = {}
    closest = 1
    r = 0.2 # because color > texture
    for i in xrange(NUM_IM):
        for j in xrange(i+1, NUM_IM):
            if (i,j) not in similarities:
                t_sim = 1 - thist_dis[(i,j)]
                c_sim = 1 - chist_dis[(i,j)]
                s = r*(t_sim) + (1-r)*(c_sim)
                d = 1 - s
                # d = r*thist_dis[(i,j)] + (1-r)*chist_dis[(i,j)]
                similarities[(i,j)] = s
                distances[(i,j)] = d
                # if (d < closest):
                #     closest = (i,j)
    return similarities, distances#, closest

def cluster(images, distances):
    clusters = {}
    for idx in xrange(0,NUM_IM):
        clusters[(idx,)] = (idx,)
    print clusters
    counter = 0
    while len(clusters) > NUM_CLUSTERS:
        nearest_pair = (None, None)
        nearest_dist = 1.0
        counter += 1
        for a in clusters:
            for b in clusters:
                if a is not b:
                    # Between multi-element clusters, nearness is the farthest
                    # distance between any two of their elements
                    dist = 0.0
                    for i in clusters[a]:
                        for j in clusters[b]:
                            if i < j:
                                k = (i,j)
                            elif i > j:
                                k = (j,i)
                            else:
                                continue
                            curr_dist = distances[k]
                            if curr_dist > dist:
                                dist = curr_dist
                    # Find out if that distance is the nearest pair so far
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_pair = (a,b)
                        nearest_pair_values = clusters[a]+clusters[b]
        clusters.pop(nearest_pair[0])
        clusters.pop(nearest_pair[1])
        new_key = nearest_pair[0] + nearest_pair[1]
        clusters[nearest_pair_values] = nearest_pair_values
        print counter, clusters
    return clusters.keys()

    # clusters = {}
    # for idx in xrange(NUM_IM):
    #     clusters[str(idx)] = (idx,)
    # print clusters
    # while len(clusters) > NUM_CLUSTERS:
    #     nearest_pair = (None, None)
    #     nearest_dist = 1.0
    #     for i in xrange(NUM_IM):
    #         for j in xrange(i+1,NUM_IM):
    #             if i in clusters and j in clusters:
    #                 cluster_a = clusters[i]
    #                 cluster_b = clusters[j]
    #                 # Between multi-element clusters, nearness is the farthest
    #                 # distance between any two of their elements
    #                 dist = 0.0
    #                 for a in cluster_a:
    #                     for b in cluster_b:
    #                         if a < b:
    #                             k = (a,b)
    #                         elif a > b:
    #                             k = (b,a)
    #                         else:
    #                             continue
    #                         curr_dist = distances[k]
    #                         if curr_dist > dist:
    #                             dist = curr_dist
    #                 # Find out if that distance is the nearest pair so far
    #                 if dist < nearest_dist:
    #                     nearest_dist = dist
    #                     nearest_pair = (i,j)
    #     if str(nearest_pair[1]) in clusters:
    #         clusters.pop(str(nearest_pair[1]))
    #     clusters[str(nearest_pair[0])] = nearest_pair[0] + nearest_pair[1]
    # print clusters
    # return clusters


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

def pair_stitch_v(images1, images2, titles):
    # note: must be same width
    n = len(images1)
    for i in xrange(0, n):
        img = np.concatenate((images1[i], images2[i]), axis=0)
        path = './'+titles[i]+'.png'
        cv2.imwrite(path, img)

def septuple_stitch_h(images, titles, dir_name, cresults, cdistances):
    plt.rcParams['font.family']='Aller Light'
    gs1 = gridspec.GridSpec(1,7)
    gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
    for k in xrange(0, NUM_IM*7, 7):
        for i in xrange(7):
            idx = cresults[k+i]
            ax = plt.subplot(gs1[i])
            plt.axis('on')
            plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)) # row, col
            plt.xticks([]),plt.yticks([])
            if cdistances:
                if i == 0:
                    plt.xlabel('similarity:')
                else:
                    sim = 1 - round(cdistances[k+i], 5)
                    plt.xlabel(sim)
                plt.title(titles[idx], size=12)
            ax.set_aspect('equal')

        title = titles[k/7]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        title = dir_name+title+'.png'
        plt.savefig(title, bbox_inches='tight')
        print title
        plt.clf()
        plt.close('all')

def four_stitch_h(images, titles, cresults):
    plt.rcParams['font.family']='Aller Light'
    gs1 = gridspec.GridSpec(1,4)
    gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
    for k in xrange(0,8,4):
        for i in xrange(4):
            idx = cresults[i+k]
            ax = plt.subplot(gs1[i])
            plt.axis('on')
            plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)) # row, col
            plt.xticks([]),plt.yticks([])
            plt.title(titles[idx], size=12)
            ax.set_aspect('equal')
        if k is 0:
            title = './texture_best_match.png'
        else:
            title = './texture_worst_match.png'
        plt.savefig(title, bbox_inches='tight')
        print title
        plt.clf()
        plt.close('all')

# ============================================================
# Where All the Magic Gets Invoked
# ============================================================

def main():

    # Check if user has provided a directory argument
    if len(sys.argv) < 2:
        sys.exit("Need to specify a path from which to read images")

    format = ".ppm"
    path = "./" + sys.argv[1]

    # Part 1: gross color matching
    images = []
    titles = []
    chists = []
    chist_images = []
    cresults = []
    cdistances = []

    # Part 2: gross texture matching
    gray_images = []
    lap_images = []
    thists = []
    thist_images = []
    tresults = []
    tdistances = []

    # Part 3: combine similarities and cluters
    similarities = []
    distances = []

    # Process images from user-provided directory
    if os.path.exists(path):
        imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(format)]
        if len(imfilelist) < 1:
            sys.exit ("Need to specify a path containing .ppm files")
        NUM_IM = len(imfilelist) # default is 40
        for el in imfilelist:
            print(el)
            # Update images and titles list
            image = cv2.imread(el, cv2.IMREAD_UNCHANGED)
            title = el[9:-4]
            images.append(image)
            titles.append(title)
            # Generate color histogram
            chist = color_histogram(image, title)
            chists.append(chist)
            # chist_images.append(plot)
            # Generate texture histogram
            gray = grayscale(image, title)
            lap = laplacian(gray, title)
            thist = texture_histogram(lap, gray, title)
            gray_images.append(gray)
            lap_images.append(lap)
            thists.append(thist)
            # thist_images.append(plot)
    else:
        sys.exit("The path name does not exist")

    # Calculate lookup table for distances based on color histograms
    chist_dis = calc_cdistance(chists)
    thist_dis = calc_tdistance(thists)

    # Determine 3 closest and 3 farthest matches for all images
    for k in xrange(NUM_IM):
        # By color
        results, distances = color_matches(k, chist_dis)
        cresults.extend(results)
        cdistances.extend(distances)
        # By texture
        res, dis = texture_matches(k, thist_dis)
        tresults.extend(res)
        tdistances.extend(dis)

    # Find set of 4 most different and 4 most similar images
    # By color
    cfour = find_four(chist_dis)
    # four_stitch_h(images, titles, cfour) # TODO: specify dir_name
    # By texture
    tfour = find_four(thist_dis)
    # four_stitch_h(images, titles, tfour)

    # septuple_stitch_h(images, titles, './texture_sim/', tresults, tdistances)
    # for i in xrange(NUM_IM):
    #     pic_stitch(cresults[i], images, titles)
    # septuple_stitch_h(images, titles, './color_sim/', cresults, cdistances)
    # septuple_stitch_h(chist_images, titles, './color_hist_sim_untitled/', cresults, None)
    # pair_stitch_v(images, chist_images, titles, './color_sims')

    # Complete and single link clustering
    # TODO: add closest?
    similarities, distances = combine_similarities(chist_dis, thist_dis)
    complete_clusters = cluster(images, distances)


'''
    if len(sys.argv) < 2:
        sys.exit("Need to specify a path from which to read images")

    format = ".ppm"
    path = "./" + sys.argv[1]

    images = []
    gray_images = []
    lap_images = []
    titles = []
    thists = []
    thist_images = []
    tresults = []
    tdistances = []

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
            gray = grayscale(image)
            cv2.imwrite('./gray/'+title+'.png', gray)
            lap = laplacian(gray)
            cv2.imwrite('./laplacian/'+title+'.png', lap)
            thist = texture_histogram(lap, gray, title)
            titles.append(title)
            image = cv2.imread(el, cv2.IMREAD_UNCHANGED)
            images.append(image)
            gray_images.append(gray)
            lap_images.append(lap)
            thists.append(thist)
            # chist_images.append(plot)
    else:
        sys.exit("The path name does not exist")

    # calculate lookup table for distances between image color histograms
    thist_dis = calc_tdistance(thists)

    # determine 4 closest and 4 farthest matches for all images
    for k in xrange(NUM_IM):
        results, distances = texture_matches(k, thist_dis)
        tresults.extend(results)
        tdistances.extend(distances)

    # print "Gross color matching results:", cresults
    tfour = find_four(thist_dis)
    # cfour = [0,9,37,15,2,30,5,14]
    four_stitch_h(images, titles, tfour)

    septuple_stitch_h(images, titles, './texture_sim/', tresults, tdistances)

'''

'''    if len(sys.argv) < 2:
        sys.exit("Need to specify a path from which to read images")

    format = ".ppm"
    path = "./" + sys.argv[1]

    images = []
    titles = []
    chists = []
    chist_images = []
    cresults = []
    cdistances = []

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
            chist = color_histogram(image, title)
            titles.append(title)
            images.append(image)
            chists.append(chist)
            # chist_images.append(plot)
    else:
        sys.exit("The path name does not exist")

    # calculate lookup table for distances between image color histograms
    chist_dis = calc_cdistance(chists)

    # determine 4 closest and 4 farthest matches for all images
    for k in xrange(NUM_IM):
        results, distances = color_matches(k, chist_dis)
        cresults.extend(results)
        cdistances.extend(distances)

    # print "Gross color matching results:", cresults
    cfour = find_four(chist_dis)
    # cfour = [0,9,37,15,2,30,5,14]
    four_stitch_h(images, titles, cfour)

    # for i in xrange(NUM_IM):
    #     pic_stitch(cresults[i], images, titles)
    # septuple_stitch_h(images, titles, './color_sim/', cresults, cdistances)
    # septuple_stitch_h(chist_images, titles, './color_hist_sim_untitled/', cresults, None)
    # pair_stitch_v(images, chist_images, titles, './color_sims')

'''
if __name__ == "__main__": main()
