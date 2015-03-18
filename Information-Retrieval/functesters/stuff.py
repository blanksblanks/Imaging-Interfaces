
main:
    #Step 1
    print "COLOR:"
    colorhist = readfiles1()
    Cvals = compimcolor(colorhist)


TOTAL_IM = 40 #i01.ppm - i40.ppm
#Step 1 bins (COLOR)
BINS = 52
BIN_SIZE = int((255.0/BINS)+1)

def compimcolor(hist):
    '''Step 1: Compare all the images using L1-norm of the color hist array'''
    overallmin = 1.0
    overallmax = 0.0
    overallminim = 0
    overallmaxim = 0
    values = np.ones((TOTAL_IM, TOTAL_IM))

    #iterate through the images
    for im1 in range(0,TOTAL_IM):
        maxim = 0
        minim = 0
        maxval = 0
        minval = 10000
        total1 = sum(sum(sum(hist[im1])))
        for im2 in range(0,TOTAL_IM):
            #don't compare the image to itself
            if im1 != im2:
                #compute the l1-norm
                total2 = sum(sum(sum(hist[im2])))
                diff = abs((hist[im1]/float(total1)) - (hist[im2]/float(total2)))

                l1norm = 1 - sum(sum(sum(diff)))/2.0

                values[im1][im2] = l1norm
                #update the max and min images
                if maxval < l1norm:
                    maxval = l1norm
                    maxim = im2+1
                if minval > l1norm:
                    minval = l1norm
                    minim = im2+1
        #display the max and min images for this image
        #print im1+1, "farthest = ", minim,"\t", minval
        #print im1+1, "closest = ", maxim,"\t", maxval
        #update the overall max and min images
        if overallmin > minval:
            overallmin = minval
            overallminim = [im1+1, minim]
        if overallmax < maxval:
            overallmax = maxval
            overallmaxim = [im1+1, maxim]
    #display the max and min images
    print "Overall farthest = ", overallmin,"\t", overallminim
    print "Overall closest = ", overallmax, "\t", overallmaxim
    return values

def fillhist(pixels, inum, hist):
    '''Step 1: Color Fill the hist array for inum image given pixels'''
    for pixel in pixels:
        #ignore black pixels
        if pixel[0] < 40 and pixel[1] < 40 and pixel[2] < 40:
            'ignore me'
        else:
            r = pixel[0]/BINS
            g = pixel[1]/BINS
            b = pixel[2]/BINS
            hist[inum-1][r][g][b] = hist[inum-1][r][g][b] + 1
    return hist

def readfiles1():
    '''Step 1: read in each file and compute the color histogram'''
    filepre = "images/i" #prefix
    filepost = ".ppm"    #postfix

    # for the histogram: [image#-1][red][green][blue]
    hist = np.zeros((TOTAL_IM, BIN_SIZE, BIN_SIZE, BIN_SIZE))
    for i in range(1, TOTAL_IM+1):
        if i < 10:
            filename = filepre + "0" + str(i) + filepost
        else:
            filename = filepre + str(i) + filepost
        p = readfile(filename)
        hist = fillhist(p, i, hist)
    return hist

def readfile(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    pixels = list(image)
    print pixels
    return pixels