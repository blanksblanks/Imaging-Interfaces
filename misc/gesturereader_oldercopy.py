import cv2, os, sys, time
import numpy as np
# import matplotlib.pyplot as plt

# ============
# DATA REDUCTION
# ============

# resize image to 300 x 300 pixels
def resize(image):
    r = 300.0 / image.shape[1] # calculate aspect ratio
    dim = (300, int(image.shape[0] * r))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image
s
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

def colorize(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
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

def edge_finder(image):
    # canny edge detection: performs gaussian filter, intensity gradient,
    #non-max suppression, hysteresis thresholding all at once
    image = cv2.Canny(image,100,200) # params: min, max vals
    show(image, 1000)
    return image

# find contours and convex hull
def contour_reader(image):
    temp = image # store because findContours modifes source

    contours,hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    # check curve for convexity defects and correct it
    # pass in contour points, hull, !returnPoints return indices
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull) # array

    image = temp # revert to original
    cv2.drawContours(image, contours, 0, (255,255,255), 1)
    image = colorize(image) # prep for drawing in color

    # defects returns four arrays: start point, end point, farthest
    # point (indices of cnt), and approx distance to farthest point
    interdigitalis = 0 # representing tip of finger to convext points
    if len(hull) > 3 and len(cnt) > 3 and (defects is not None):
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            # print start, end, far
            cv2.line(image,start,end,[0,255,0],1)
            cv2.circle(image,far,3,[255,0,255],-1)
            # print d
            if d > 6000: # trial and error
                interdigitalis += 1
    print 'interdigitalis:', interdigitalis

    # find centroid
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = (cx, cy)
    cv2.circle(image, centroid, 6, (255,255,0), -1)
    print 'centroid:', centroid

    gesture = classify(interdigitalis)
    print 'gesture', gesture

    # calculate size for height as image is a numpy array
    h = len(image)
    location,lst = locate(h,cx,cy)
    print location

    # draw visual grid
    for i in xrange(0, len(lst), 2):
        a = lst[i]
        b = lst[i+1]
        # print a,b
        cv2.line(image,a,b,[128,128,128],1)
    show(image, 1000)

    pair = (gesture, location)
    return image,pair


def classify(num):
    if num is 0:
        return 'fist'
    if num is 1:
        return 'two'
    if num is 2:
        return 'three'
    if num is 3:
        return 'four'
    if num is 4:
        return 'five'
    else:
        return 'unknown'

def locate(h, cx, cy):
    w = h # square image
    h1 = int(h/3)
    h2 = h - int(h/3)
    v1 = int(w/3)
    v2 = w - int(w/3)
    lst = [(0,h1), (w,h1), (0,h2), (w,h2), (v1,0), (v1,h), (v2,0), (v2,h)]

    p1 = (v1,h1) # [0] = x, [1] = y
    p2 = (v2,h1)
    p3 = (v1,h2)
    p4 = (v2,h2)

    if cx > p1[0] and cx < p2[0] and cy < p3[1] and cy > p1[1]:
        return 'center', lst
    elif cx < p1[0] and cy < p1[1]:
        return 'top-left', lst
    elif cx > p2[0] and cy < p2[1]:
        return 'top-right', lst
    elif cx < p3[0] and cy > p3[1]:
        return 'bottom-left', lst
    elif cx > p4[0] and cy > p4[1]:
        return 'bottom-right', lst
    elif cx > p1[0] and cx < p2[0] and cy < p2[1]:
        return 'top-center', lst
    elif cx > p1[0] and cx < p2[0] and cy > p4[1]:
        return 'bottom-center', lst
    else:
        return 'unknown', lst

def authenticate(sequence):
    denial = 'Entered wrong password combination. Access denied.'
    admittance = 'Thank you for entering your password combination. Access granted.'
    confusion = 'There is an unknown value in your input. Access denied.'

    for tup in sequence:
        if 'unknown' in tup:
            return confusion

    # check three sequences
    tup1 = sequence[0]
    if tup[0] is not 'center' and tup[1] is not 'fist':
         return denial
    tup2 = sequence[1]
    if tup[0] is not 'five' and tup[1] is not 'bottom-left':
        return denial
    tup3 = sequence[2]
    if tup[0] is not 'five' and tup[2] is not 'top-right':
        return denial

    return admittance

def save(image, name):
    cv2.imwrite(name, image)

def show(image, wait):
    cv2.waitKey(wait)
    cv2.imshow('Image', image)

def main():

    if len(sys.argv) < 2:
        sys.exit("Need to specify a path from which to read images")

    imageformat=".JPG"
    path = "./" + sys.argv[1]

    combination = []

    # load image sequence
    if os.path.exists(path):
        imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
        if len(imfilelist) < 1:
        	sys.exit ("Need to specify a path containing .JPG files")
        for el in imfilelist:
            print el
            image = cv2.imread(el, cv2.IMREAD_COLOR) # load original
#            image = rotate(image)
 #           save(image, el)
            image = resize(image)
            save(image, 'post-processing/'+el[:-4]+'_resized.png')
            image = grayscale(image)
            save(image, el[:-4]+'_grayscale.png')
            image = binarize(image)
            save(image, el[:-4]+'_binarized.png')
            image,combo = contour_reader(image)
            combination.append(combo)
            save(image, el[:-4]+'_contours.png')
    else:
        sys.exit("The path name does not exist")

    print combination
    decision = authenticate(combination)
    print decision

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