import cv2
import numpy as np

# ============================================================
# Globals
# ============================================================

# set such that full image array is printed out
np.set_printoptions(threshold=np.nan)

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
map_labeled = cv2.imread('ass3-labeled.pgm', 0) # load map_labeled as grayscale
map_campus = cv2.imread('ass3-campus.pgm', 1) # load map_campus as color
map_binary = cv2.cvtColor(map_campus,cv2.COLOR_BGR2GRAY) # load map_campus as grayscale

def print_imarray(im):
    w = len(im[0])
    for i in xrange(len(im)):
        print im[i][0], im[i][1], im[i][3]

def grayscale(image):
    """Convert image into black and white with (R+G+B)/3"""
    h = len(image)
    w = len(image[0])
    gray = np.zeros(shape=(h,w))
    for row in xrange(h):
        for col in xrange(w):
            # compose gray image pixel by pixel
            pixel = image[row][col]
            rgb = int(pixel[0]) + int(pixel[1]) + int(pixel[2])
            g = int(round((rgb / 3.0),0)) # round up
            gray[row][col] = g
    # gray_img = cv2.imread(gray, cv2.IMREAD_UNCHANGED)
    cv2.imwrite('./gray.png', gray)
    cv2.imwrite('./unchanged.png', image)
    return gray

# print gray

# ============================================================
# User Interface
# ============================================================

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        idx = which_building(ix,iy)
        print 'Mouse clicked', ix,iy, 'building', idx

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                # cv2.rectangle(map_campus,(ix,iy),(x,y),(0,255,0),-1)
                cv2.circle(map_campus,(x,y),5,(0,255,0),-1)
            else:
                cv2.circle(map_campus,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            # cv2.rectangle(map_campus,(ix,iy),(x,y),(0,255,0),-1)
            cv2.circle(map_campus,(x,y),5,(0,255,0),-1)
        else:
            cv2.circle(map_campus,(x,y),5,(0,0,255),-1)

# ============================================================
# The "What"
# ============================================================

def which_building(x,y):
    idx = int(map_labeled[y][x][2])
    return idx

def load_names(filename):
    names = {}
    infile = open(filename, 'rU')
    while True:
        try:
            line = infile.readline().replace('"', '').split('=')
            n = line[0]
            name = line[1].rstrip('\r\n')
            names[n] = name
        except IndexError:
            break
    return names

def id_building(cnt):
    """Identify what building a contour represents by its pixel value"""

    # To get all the points which comprise an object
    # Numpy function gives coordinates in (row, col)
    # OpenCV gives coordinates in (x,y)
    # Note row = x and col = y
    mask = np.zeros(map_binary.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    #pixelpoints = cv2.findNonZero(mask)

    # Use color to determine index, which will give us name
    color = cv2.mean(map_labeled,mask=mask)
    # print color
    if (color[0] > 0.9):
        idx = int(round(color[0], 0))
        return idx
    else:
        return None

def measure_building(cnt, print_rect=False):
    # Let (x,y) be top-left coordinate and (w,h) be width and height
    # Find min, max value of x, min, max value of y
    x,y,w,h = cv2.boundingRect(cnt)
    mbr = [(x,y),(x+w,y+h)] # mbr[0] = T-L corner, mbr[1] = B-R corner
    # print " Minimum Bounding Rectangle: ({0},{1}), ({2},{3})".format(x,y,(x+w),(y+h))
    roi = map_campus[y:y+h,x:x+w]
    # cv2.imwrite(str(idx) + '.jpg', roi)
    # To draw a rectangle, you need T-L corner and B-R corner
    if print_rect:
        cv2.rectangle(map_campus,(x,y),(x+w,y+h),(200,0,0),2)

    # Image moments help you to calculate center of mass, area of object, etc.
    # cv2.moments() gives dictionary of all moment values calculated
    M = cv2.moments(cnt)
    # Centroid is given by the relations
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = (cx, cy)
    # print ' Center of Mass:', centroid
    # To draw a circle, you need its center coordinates and radius
    cv2.circle(map_campus, centroid, 3, (255,255,0), -1)

    # Contour area is given by the function cv2.contourArea(cnt) or
    area = M['m00']
    # print ' Area:', area
    # area = cv2.contourArea(cnt)
    # x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    # print ' Extent:', round(extent, 3)

    # label = str(idx) + ' : ' + str(area) + ' : ' + str(extent)
    # cv2.putText(map_campus, str(idx), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
    # check curve for convexity defects and correct it
    # pass in contour points, hull, !returnPoints return indices
    # hull = cv2.convexHull(cnt,returnPoints = False)
    # defects = cv2.convexityDefects(cnt,hull) # array
    # if len(hull) > 3 and len(cnt) > 3 and (defects is not None):
    #     for i in range(defects.shape[0]):
    #         s,e,f,d = defects[i,0]
    #         start = tuple(cnt[s][0])
    #         end = tuple(cnt[e][0])
    #         far = tuple(cnt[f][0])
    #         # print start, end, far
    #         cv2.line(map_campus,start,end,[0,255,0],1)
    #         cv2.circle(map_campus,far,3,[255,0,255],-1)

    # this just draws the rect again
    #cv2.drawContours(map_campus, contours, 0, (0,0,255), 1)

    # find corners - this method is buggy
    # dst = cv2.cornerHarris(map_binary,3,3,0.2)
    # dst = cv2.dilate(dst,None)
    # map_campus[dst>0.01*dst.max()]=[0,0,255]

    return mbr, centroid, area, extent

def analyze_buildings(names):
    """Find information about buildings and save in list of dicts"""
    num_buildings = len(names)
    buildings = list(np.zeros(num_buildings))

    # Find contours in binary campus map image
    # Contours is a Python list of all the contours in the image
    # Each contour is a np array of (x,y) boundary points of each object
    contours,hierarchy = cv2.findContours(map_binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        building = {}
        idx = id_building(cnt)
        if idx is None:
            continue
        mbr, centroid, area, extent = measure_building(cnt)
        building['number'] = idx
        building['name'] = names[str(idx)]
        building['mbr'] = mbr
        building['centroid'] = centroid
        building['area'] = area
        building['extent'] = extent
        building['cnt'] = cnt
        buildings[(idx-1)] = building

    max_area, min_area = analyze_areas(buildings) # add True arg to print results
    for building in buildings:
        size = describe_size(building, max_area)
        describe_shape(building)
        # shape = describe_shape(building)
        # multiple = describe_multiplicity
    # analyze_extents(buildings)
    # analyze_shapes(buildings)

    return buildings

def analyze_extents(buildings):
    """Sort buildings by extent and determine cutoff for rectangles"""
    print 'Analyzing building extents (area/mbr) and convexity...'
    num_buildings = len(buildings)
    sorted_buildings = sorted(buildings, key=lambda k:-k['extent'])
    indices = [(sorted_buildings[i]['number']-1) for i in range(num_buildings)]
    for i in indices:
        building = buildings[i]
        convex = cv2.isContourConvex(building['cnt'])
        print round(building['extent'],4), '\t', convex, '\t', i+1, building['name']

def analyze_areas(buildings, print_results=False):
    """Sort buildings by area, determine cutoff for size and return max"""
    num_buildings = len(buildings)
    sorted_buildings = sorted(buildings, key=lambda k:-k['area'])
    indices = [(sorted_buildings[i]['number']-1) for i in range(num_buildings)]
    areas = [(sorted_buildings[i]['area']) for i in range(num_buildings)]

    max_area = areas[0]
    avg_area = sum(areas)/num_buildings
    min_area = areas[-1]

    # Print results to analyze cutoffs for size categories
    if (print_results):
        print 'Analyzing building areas...'
        ratios = [round(areas[i]/max_area,3) for i in range(num_buildings)]
        ratio_diffs = [round((ratios[i+1]-ratios[i]),3) for i in range(num_buildings-1)]
        ratio_diffs.insert(0,0)
        max_area_ratios = [round(max_area/areas[i],3) for i in range(num_buildings)]

        print 'Max Area:', max_area, '\nAverage:', round(avg_area, 3)
        print 'Area\tRatio r\tDiff r\tMax r\tBuilding'
        for i in xrange(num_buildings):
            idx = indices[i]
            print areas[i], '\t', ratios[i], '\t', ratio_diffs[i], '\t', max_area_ratios[i], '\t', idx+1, buildings[idx]['name']

    return max_area, min_area

def analyze_shapes(buildings):
    """Sort building by shape similarity (not very good results)"""
    print 'Analyzing shape similarity with cv2.matchShapes...'
    num_buildings = len(buildings)
    shape_sim = {}
    for i in xrange(num_buildings):
        for j in xrange(i+1, num_buildings):
            cnt1 = buildings[i]['cnt']
            cnt2 = buildings[j]['cnt']
            ret = cv2.matchShapes(cnt1,cnt2, 1,0.0)
            shape_sim[(i,j)] = ret
    sorted_sim = sorted([(value,key) for (key,value) in shape_sim.items()])
    for sim in sorted_sim[:40]:
        bldg1 = sim[1][0]
        bldg2 = sim[1][1]
        print round(sim[0],4), '\t', buildings[bldg1]['name'], '&', buildings[bldg2]['name']

def describe_size(building, max_area):
    ratio = building['area']/max_area
    if ratio > 0.7: # cutoff at College Walk
        return 'colossal'
    if ratio < 0.4: # cutoff at Journalism & Furnald
        return 'large'
    elif ratio > 0.15: # cutoff at Philosophy
        return 'middling'
    elif ratio > 0.1: # cutoff Earl Hall
        return 'small'
    else:
        return 'miniscule'

def describe_shape(building):
    tolerance = 2
    # map_l = cv2.imread('ass3-labeled.pgm', 0)
    # print map_l
    # Recall mbr = [(x,y),(x+w,y+h)]
    # nw = building['mbr'][0]
    # se = building['mbr'][1]
    # ne = (se[0],nw[1]) # (x+w,y)
    # sw = (nw[0],se[1]) # (x,y+h)
    # # tl = (x+w,y)
    # # br = (x,y+h)

    # # Extract x,y,w,h coordinates
    # x = [0] + tolerance
    # y = nw[1] + tolerance
    # w = se[0] - x - tolerance
    # h = se[1] - y - tolerance

    x,y,w,h = cv2.boundingRect(building['cnt'])
    x += tolerance
    y += tolerance
    w -= 2*tolerance
    h -= 2*tolerance

    # Extract four corners
    nw = (x,y)
    se = (x+w,y+h)
    ne = (x+w,y)
    sw = (x,y+h)

    # Extract midpoints on every wall face
    n = (x+(w/2),y)
    e = (x+w,y+(h/2))
    s = (x+(w/2),y+h)
    w = (x,y+(h/2))

    corners = [nw,se,ne,sw]
    midpoints = [n,e,s,w]
    corners_filled = [] # nw, ne, se, sw
    midpoints_filled = [] # n, e, s, w

    print building['name']

    try :
        for corner in corners:
            # print 'Map labeled corner: ', map_labeled[corner]
            if map_labeled[tuple(reversed(corner))] == building['number']:
                corners_filled.append(1)
                cv2.circle(map_campus, corner, 1, (255,255,0), -1)
            else:
                corners_filled.append(0)
                cv2.circle(map_campus, corner, 1, (0,0,255), -1)

        for midpoint in midpoints:
            # 'Map labeled midpoint: ', map_labeled[midpoint]
            if map_labeled[tuple(reversed(midpoint))] == building['number']:
                midpoints_filled.append(1)
                cv2.circle(map_campus, midpoint, 1, (255,255,0), -1)
            else:
                midpoints_filled.append(0)
                cv2.circle(map_campus, midpoint, 1, (0,0,255), -1)
    except (IndexError):
        print IndexError

    print 'corners', corners_filled
    print 'midpoints', midpoints_filled




def print_info(buildings):
    for building in buildings:
        print building['number'], ':', building['name']
        print ' Minimum Bounding Rectangle:', building['mbr'][0], ',', building['mbr'][1]
        print ' Center of Mass:', building['centroid']
        print ' Area:', building['area']
        # print ' Description', building['description']


# ============================================================
# Main Invocation
# ============================================================

def main():

    # map_campus = np.zeros((512,512,3), np.uint8)

    # Analyze image
    names = load_names('ass3-table.txt')
    buildings = analyze_buildings(names)
    print_info(buildings)

    cv2.namedWindow('Columbia Campus Map')
    cv2.setMouseCallback('Columbia Campus Map', draw_circle)
    print "Showing image..."

    # cv2.waitKey(0)

    while(1):
        cv2.imshow('Columbia Campus Map', map_campus)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            print 'pressed m'
            mode = not mode
        elif k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__": main()
