import cv2
import numpy as np
import sys
import os

def resize(image, sz):
    r = float(sz) / image.shape[1] # calculate aspect ratio
    dim = (int(sz), int(image.shape[0] * r))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

def main():

    if len(sys.argv) < 2:
        sys.exit("Need to specify a path from which to read images")

    format = ".png"
    path = "./" + sys.argv[1]

    images = []

    # load image sequence
    if os.path.exists(path):
        imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(format)]
        if len(imfilelist) < 1:
            sys.exit ("Need to specify a path containing .png files")
        n = len(imfilelist)
        for el in imfilelist:
            print(el)
            image = cv2.imread(el, cv2.IMREAD_UNCHANGED)
            if len(sys.argv) == 3:
                image = resize(image, sys.argv[2]) # option to add second arg as size
            title = el[9:-4]
            images.append(image)
    else:
        sys.exit("The path name does not exist")

    for i in xrange(1, n):
        if i == 1:
            img = np.concatenate((images[i-1], images[i]), axis=0)
        else:
            img = np.concatenate((img, images[i]), axis=0)

    dir_name = './color_matches/'
    cv2.imwrite(path+'.png', img)

if __name__ == "__main__": main()

