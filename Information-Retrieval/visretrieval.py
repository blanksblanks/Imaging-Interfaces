import os
import sys
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from scipy.misc import comb
# from PIL import Image
# from numpy import linalg as la


# ============================================================
# Constants
# ============================================================

NUM_IM = 40
NUM_CLUSTERS = 7
COL_RANGE = 256
BINS = 8
BIN_SIZE = int(COL_RANGE/BINS)
BLK_THRESH = 40
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
    '''
    Calculate the 3D color histogram of an image by counting the number
    of RGB values in a set number of bins
    image -- pre-loaded image using cv2.imread function
    title -- image title
    (optional: visualize the histogram as a bar graph)
    '''
    colors = []
    h = len(image)
    w = len(image[0])
    # Create a 3D array - if BINS is 8, there are 8^3 = 512 total bins
    hist = np.zeros(shape=(BINS, BINS, BINS))
    # Traverse each pixel in the image matrix and increment the appropriate
    # hist[r_bin][g_bin][b_bin] - we know which one by floor dividing the
    # original RGB values / BIN_SIZE
    for i in xrange(h):
        for j in xrange(w):
            pixel = image[i][j]
            # If the pixel is below black threshold, do not count it
            if pixel[0] > BLK_THRESH and pixel[1] > BLK_THRESH \
            and pixel[2] > BLK_THRESH:
                # Note: pixel[i] is descending since OpenCV loads BGR
                r_bin = pixel[2] / BIN_SIZE
                g_bin = pixel[1] / BIN_SIZE
                b_bin = pixel[0] / BIN_SIZE
                hist[r_bin][g_bin][b_bin] += 1
                # Generate list of color keys for visualization
                if (r_bin,g_bin,b_bin) not in colors:
                    colors.append( (r_bin,g_bin,b_bin) )
    # plot = visualize_chist(image, hist, colors, title)
    # return plot, hist
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
    # ensure that a<b, b<c and c<d as order does not matter
    for a in xrange(NUM_IM):
        for b in xrange(a+1,NUM_IM):
            for c in xrange(b+1,NUM_IM):
                for d in xrange(c+1,NUM_IM):
                    results[(a,b,c,d)] = \
                    chist_dis[(a,b)] + chist_dis[(a,c)] + \
                    chist_dis[(a,d)] + chist_dis[(b,c)] + \
                    chist_dis[(b,d)] + chist_dis[(c,d)]
    results = sorted([(v, k) for (k, v) in results.items()])
    best = results[0]
    worst = results[-1]
    indices = list(best[1])
    indices.extend(list(worst[1]))
    # print "results: ", len(results), #results
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
    # return hist, plot
    return hist

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

def combine_similarities(chist_dis, thist_dis, r):
    '''where r is the ratio'''
    similarities = {}
    distances = {}
    closest = 1
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

# 0 is COMPLETE LINK, 1 is SINGLE
def cluster(distances, link):
    clusters = {}
    for idx in xrange(0,NUM_IM):
        clusters[(idx,)] = (idx,)
    # print clusters
    counter = 0
    while len(clusters) > NUM_CLUSTERS:
        nearest_pair = (None, None)
        nearest_dist = 1 # total dissimilarity
        counter += 1
        # Go through every cluster-pair to find nearest pair
        for a in clusters:
            for b in clusters:
                # Do not compare with same cluster
                if a is not b:
                    dist = link
                    # For each element in both clusters, determine "nearness"
                    # Complete nearness: farthest distance bt any 2 el in 2 clusters 
                    # Single nearness: nearest distance bt any 2 el in 2 clusters
                    for i in clusters[a]:
                        for j in clusters[b]:
                            k = (i,j)
                            if i > j:
                                k = (j,i) # tuples always in ascending order
                            curr_dist = distances[k]
                            if (link is 0 and curr_dist > dist) or (link is 1 and curr_dist < dist):
                                # print counter, ': New distance for', clusters[a], clusters[b], dist, '->', curr_dist
                                dist = curr_dist
                    # Find out if this is the nearest pair so far in the iteration
                    if dist < nearest_dist:
                        # print counter, ': Replace distance with ', (a,b), nearest_dist, '->', dist
                        nearest_dist = dist
                        nearest_pair = (a,b)
                        nearest_pair_values = clusters[a]+clusters[b] # add elements in a tuple
        # combine the nearest pair of clusters and remove old clusters
        clusters.pop(nearest_pair[0])
        clusters.pop(nearest_pair[1])
        # new_pair = nearest_pair[0] + nearest_pair[1]
        clusters[nearest_pair_values] = nearest_pair_values
        # clusters[new_pair] = clusters[new_pair]
        # print counter, clusters.keys()

    # print clusters
    clusters = clusters.keys()
    return clusters

# ============================================================
# Performance Measure
# ============================================================

def loadCSV(filename):
    results = []
    infile = open(filename, 'rU')
    for idx in xrange(NUM_IM):
        results.append(idx)
        data = infile.readline().strip().replace(' ', '').split(',')
        # append img idx to results list, then append friend results
        # for best color, worst color, best texture, worst texture
        # results.extend(data)
        for j in xrange(4):
            results.append(int(data[j])-1)
    return results

def print_results(tp, fp, fn):
    print "TP: %d, FP: %d, FN: %d" % (tp, fp, fn)

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

    print "Precision : %f" % precision
    print "Recall    : %f" % recall
    print "F1        : %f" % ((2.0 * precision * recall) / (precision + recall))

def match_eval(cresults, tresults, hresults):
    '''
    tp -- if human guess is in sys results
    fn -- if human guess is missing from sys results
    fp -- if human guess is in opposite of sys results
    '''
    # best,worst color and best,worst texture measures
    bc_tp, bc_fp, bc_fn = 0,0,0
    wc_tp, wc_fp, wc_fn = 0,0,0
    bt_tp, bt_fp, bt_fn = 0,0,0
    wt_tp, wt_fp, wt_fn = 0,0,0

    counter = 0

    for i in xrange(0,NUM_IM*7,7):

        # range indices for best/worst system results
        b_from = i+1
        b_to = i+4
        w_from = i+4
        w_to = i+7

        # human results for best color, worst color,
        # best texture, worst texture
        # note: need var j because human list increments by 5
        # whereas system list increments by 7
        j = i - (2*counter)
        c_best = hresults[j+1]
        c_worst = hresults[j+2]
        t_best = hresults[j+3]
        t_worst = hresults[j+4]
        counter += 1

        # check best color match
        if c_best in cresults[b_from:b_to]:
            bc_tp += 1
        else:
            bc_fn += 1
        # since we are measuring computer against human perf:
        # program messed up and made a big false positive
        if c_best in cresults[w_from:w_to]:
            wc_fp += 1
            print 'Human best color match in system worst!', c_best

        # check worst color match
        if c_worst in cresults[w_from:w_to]:
            wc_tp += 1
        else:
            wc_fn += 1
        if c_worst in cresults[b_from:b_to]:
            bc_fp += 1
            print'Human worst color match in system best!', c_worst

        # check best texture match
        if t_best in tresults[b_from:b_to]:
            bt_tp += 1
        else:
            bt_fn += 1
        # since we are measuring computer against human perf:
        # program messed up and made a big false positive
        if t_best in tresults[w_from:w_to]:
            wt_fp += 1
            print 'Human best texture match in system worst!', t_best

        # check worst texture match
        if t_worst in tresults[w_from:w_to]:
            wt_tp += 1
        else:
            wt_fn += 1
        if t_worst in tresults[b_from:b_to]:
            bt_fp += 1
            print'Human worst texture match in system best!', t_worst

    # Calculate total tp,fp,fn values
    tp = bc_tp + wc_tp + bt_tp + wt_tp
    fp = bc_fp + wc_fp + bt_fp + wt_fp
    fn = bc_fn + wc_fn + bt_fn + wt_fn

    print "1: BEST COLOR MATCH RESULTS"
    print_results(bc_tp, bc_fp, bc_fn)

    print "2: WORST COLOR MATCH RESULTS"
    print_results(wc_tp, wc_fp, wc_fn)

    print "3: BEST TEXTURE MATCH RESULTS"
    print_results(bt_tp, bt_fp, bt_fn)

    print "4: WORST TEXTURE MATCH RESULTS"
    print_results(wt_tp, wt_fp, wt_fn)

    print "5: OVERALL COLOR AND TEXTURE MATCH RESULTS"
    print_results(tp, fp, fn)

def cluster_id(list_of_tups, key):
    '''
    Take list of cluster tups and return list of cluster id's
    where each list index is the image and the el is its set id
    key -- 1 if its starting index needs to be decremented to 0
    '''
    cluster_ids = [0] * 40
    for i in xrange(7):
        tup = list_of_tups[i]
        for j in tup:
            if key is 1:
                cluster_ids[j-1] = i
            else:
                cluster_ids[j] = i
    return cluster_ids

def cluster_eval(system, human):
    # Change list of tuples -> list of lists
    hum = cluster_id(human, 1)
    syst = cluster_id(system, 0)

    # Initialize True-Pos, True-Neg False-Pos, False-Neg measures
    tp, tn, fp, fn = 0,0,0,0
    for i in xrange(NUM_IM):
        for j in xrange(i+1,NUM_IM):
            if hum[i] == hum[j] and syst[i] == syst[j]:
                tp += 1
            if hum[i] == hum[j] and syst[i] != syst[j]:
                fp += 1
            if hum[i] != hum[j] and syst[i] == syst[j]:
                fn += 1
            if hum[i] != hum[j] and syst[i] != syst[j]:
                tn += 1
    print "TP: %d, FP: %d, TN: %d, FN: %d" % (tp, fp, tn, fn)

    rand_idx = (float(tp + tn) / (tp + fp + fn + tn))
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

    # Print results:
    print "Rand index: %f" % rand_idx
    print "Precision : %f" % precision
    print "Recall    : %f" % recall
    print "F1        : %f" % ((2.0 * precision * recall) / (precision + recall))

    return rand_idx

# ============================================================
# Helper and Display Functions
# ============================================================

def save(image, name):
    cv2.imwrite(name, image)

def show(image, wait):
    cv2.waitKey(wait)
    cv2.imshow('Image', image)

def display_all(images, titles):
    plt.rcParams['font.family']='Aller Light'
    for i in xrange(NUM_IM):
        plt.subplot(5,8,i+1),plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)) # row, col
        plt.title(titles[i], size=12)
        plt.xticks([]),plt.yticks([])
    title = './all_im.png'
    plt.savefig(title, bbox_inches='tight')
    print title

def pair_stitch_v(images1, images2, titles):
    # note: must be same width
    n = len(images1)
    for i in xrange(0, n):
        img = np.concatenate((images1[i], images2[i]), axis=0)
        path = './'+titles[i]+'.png'
        cv2.imwrite(path, img)

def septuple_stitch_h(images, titles, dir_name, cresults, cdistances, cvt):
    plt.rcParams['font.family']='Aller Light'
    gs1 = gridspec.GridSpec(1,7)
    gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
    for k in xrange(0, NUM_IM*7, 7):
        for i in xrange(7):
            idx = cresults[k+i]
            ax = plt.subplot(gs1[i])
            plt.axis('on')
            if cvt is 0:
                plt.imshow(images[idx], cmap="Greys_r")
            elif cvt is -1:
                plt.imshow(images[idx], cmap="binary")
            else:
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

def four_stitch_h(images, titles, cresults, dir_name):
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
            title = 'best_match.png'
        else:
            title = 'worst_match.png'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(dir_name+title, bbox_inches='tight')
        print title
        plt.clf()
        plt.close('all')

def cluster_stitch_h(images, titles, clusters, link, dir_name):
    # set n to the length of the largest cluster
    n = 1
    for cluster in clusters:
        if len(cluster) > n:
            n = len(cluster)
    plt.rcParams['font.family']='Aller Light'
    gs1 = gridspec.GridSpec(7,n)
    gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
    for k in xrange(0, n*7, n):
        cluster = clusters[(k/n)]
        for i in xrange(len(cluster)):
            idx = cluster[i]
            ax = plt.subplot(gs1[i+k])
            plt.axis('on')
            plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)) # row, col
            plt.xticks([]),plt.yticks([])
            plt.title(titles[idx], size=12)
            ax.set_aspect('equal')
    if link is 0:
        title = 'cluster_complete'
    else:
        title = 'cluster_single'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    title = dir_name+title+'.png'
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

    # ==================================
    # Parts 1 and 2: gross color and texture matching
    # ==================================

    # 1: color
    images = []
    titles = []
    chists = []
    chist_images = []
    cresults = []
    cdistances = []

    # 2: texture
    gray_images = []
    lap_images = []
    thists = []
    thist_images = []
    tresults = []
    tdistances = []

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
            # thist = texture_histogram(lap, gray, title)
            thist = texture_histogram(lap, gray, title)
            # thist,plot = texture_histogram(lap, gray, title)
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
    cfour = find_four(chist_dis)
    tfour = find_four(thist_dis)

    # ==================================
    # Note: Commented out image saving and display
    # ==================================

    # Display all images
    # display_all(images,titles)

    # Display septuples
    # septuple_stitch_h(images, titles, './part1/color_sim/', cresults, cdistances, 1)
    # septuple_stitch_h(chist_images, titles, './part1/color_hist_sim_untitled/', cresults, None, 1)
    # for i in xrange(NUM_IM):
    #     pic_stitch(cresults[i], images, titles)
    # septuple_stitch_h(images, titles, './part2/tpics/', tresults, tdistances, 1)
    # septuple_stitch_h(thist_images, titles, './part2/thistograms/', tresults, None, 0)
    # septuple_stitch_h(gray_images, titles, './part2/gray_images/', tresults, None, 0)
    # septuple_stitch_h(lap_images, titles, './part2/lap_images/', tresults, None, -1)

    # Display four best and four worst, by color and by texture
    # four_stitch_h(images, titles, cfour, './part1/')
    # four_stitch_h(images, titles, tfour, './part2/')

    # ==================================
    # Part 3: combine similarities and cluster
    # ==================================

    similarities = []
    distances = []
    complete = []
    single = []

    # Testing combined similarities
    similarities, distances = combine_similarities(chist_dis, thist_dis, 0.2)
    combo_four = find_four(distances)
    # four_stitch_h(images, titles, combo_four, './part3/')

    complete = cluster(distances,0)
    single = cluster(distances,1)
    cluster_stitch_h(images, titles, complete, 0, './part3/')
    cluster_stitch_h(images, titles, single, 1, './part3/')

    # Testing additional r values for combined similarity to see their effect
    sim, dis = combine_similarities(chist_dis, thist_dis, 0.5)
    comp = cluster(dis,0)
    sim2, dis2 = combine_similarities(chist_dis, thist_dis, 0.8)
    comp2 = cluster(dis2,0)

    # ==================================
    # Part 4: creative step
    # ==================================

    robert = loadCSV('./part4/Robert.csv')
    jacky = loadCSV('./part4/Jacky.csv')
    alex = loadCSV('./part4/Alex.csv')
    ashley = loadCSV('./part4/Ashley.csv')

    # clusters
    # TODO: load the CSV file instead of this manual stuff
    jacky_c = \
    [(2,17,23), \
    (25,33,26,28), \
    (18,19,20,21,22), \
    (31,32,27), \
    (13,30,29,34), \
    (1,3,4,8,10,24,36,37,38,16), \
    (5,6,7,9,11,12,14,15,35,39,40)]
    robert_c = \
    [(37,38,19,24), \
    (2,13,11,10,16,7,5,6,15,9,14,12), \
    (3,1,8,4), \
    (27,31,32,), \
    (29,34,17,30,36,39), \
    (35,33,40,20,23), \
    (25,26,28,18)]
    alex_c = \
    [(8,1,4,3,24,10,37,38,16,19),
    (2,39,21,22),
    (13,12,14),
    (11,7,6,9),
    (15,20,23,33,5,6),
    (40,25,17,34,35,18,26,28),
    (31,32,29,27,30)]
    ashley_c = \
    [(1,3,4,8,10,16),
    (11,15,5,6,23,40,20,33,7),
    (34,2),
    (28,18,21,39,26,17,35,25),
    (12,9,13,14),
    (27,32,31),
    (37,19,22,36,29,24,38,30)]

    # print 'Robert', robert, robert_c
    # print 'Jacky', jacky, jacky_c
    # print 'Alex', alex, alex_c
    # print 'Ashley', ashley, ashley_c

    print '\n\n========================'
    print 'Matching Evaluation:'
    print '========================'
    print '\nRobert:'
    match_eval(cresults, tresults, robert)
    print '\nJacky:'
    match_eval(cresults, tresults, jacky)
    print '\nAlex:'
    match_eval(cresults, tresults, alex)
    print '\nAshley:'
    match_eval(cresults, tresults, ashley)

    print '\n\n========================'
    print 'Clustering Evaluation:'
    print '========================'

    print '\n\nUsing r=0.2:'
    print '\nRobert:'
    cluster_eval(complete, robert_c)
    print '\nJacky:'
    cluster_eval(complete, jacky_c)
    print '\nAlex:'
    cluster_eval(complete, alex_c)
    print '\nAshley:'
    cluster_eval(complete, ashley_c)

    print '\n\nUsing r=0.5:'
    print '\nRobert:'
    cluster_eval(comp, robert_c)
    print '\nJacky:'
    cluster_eval(comp, jacky_c)
    print '\nAlex:'
    cluster_eval(comp, alex_c)
    print '\nAshley:'
    cluster_eval(comp, ashley_c)

    print '\n\nUsing r=0.8:'
    print '\nRobert:'
    cluster_eval(comp2, robert_c)
    print '\nJacky:'
    cluster_eval(comp2, jacky_c)
    print '\nAlex:'
    cluster_eval(comp2, alex_c)
    print '\nAshley:'
    cluster_eval(comp2, ashley_c)

if __name__ == "__main__": main()
