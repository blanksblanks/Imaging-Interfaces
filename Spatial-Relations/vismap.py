import cv2
import numpy as np
from matplotlib.path import Path

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
    find_extrema(buildings)

    for building in buildings:
        location = describe_location(building)
        size = describe_size(building, max_area)
        description = describe_shape(building)
        description.insert(0,size)
        description.extend(location)
        building['description'] = description

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
    elif ratio > 0.4: # cutoff at Journalism & Furnald
        return 'large'
    elif ratio > 0.15: # cutoff at Philosophy
        return 'middling'
    elif ratio > 0.1: # cutoff Earl Hall
        return 'small'
    else:
        return 'tiny'

def describe_shape(building):
    """Describe shape based on corner and midpoint counts"""

    descriptions = []

    x,y,w,h = cv2.boundingRect(building['cnt'])

    # Tolerance based on ratio of min(w,h) as building sizes vary
    tolerance = min(w,h)/10

    # Shift x,y,w,h so corners and midpoints are closer to center
    # Else they may report false negative on the MBR perimeter, esp
    # for bumpy buildings
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
    west = (x,y+(h/2))

    corners = [nw,se,ne,sw]
    midpoints = [n,e,s,west] # west because it overwrites width
    corners_filled = [] # nw, ne, se, sw
    midpoints_filled = [] # n, e, s, west

    for corner in corners:
        if map_labeled[tuple(reversed(corner))] == building['number']:
            corners_filled.append(1)
            cv2.circle(map_campus, corner, 1, (255,255,0), -1)
        else:
            corners_filled.append(0)
            cv2.circle(map_campus, corner, 1, (0,0,255), -1)

    for midpoint in midpoints:
        if map_labeled[tuple(reversed(midpoint))] == building['number']:
            midpoints_filled.append(1)
            cv2.circle(map_campus, midpoint, 1, (0,255,0), -1)
        else:
            midpoints_filled.append(0)
            cv2.circle(map_campus, midpoint, 1, (0,0,255), -1)

    # Count the number of corners and midpoints for each building
    # Not necessary to consider order at this point
    corners_count = corners_filled.count(1)
    midpoints_count = midpoints_filled.count(1)

    # print building['number'], building['name']
    # print ' Tolerance', tolerance
    # print '', corners_filled, 'Corners Count', corners_count
    # print '', midpoints_filled, 'Midpoints Count', midpoints_count

    # Difference between height and width should be small enough
    # Decided not to use absolute value as differnce is relative
    # Also check that building fills out most of the MBR
    # Ruling out Journalism & Furnald, and Chandler & Havemeyer
    if (abs(h-w) <= max(h,w)/5) and (building['extent'] > 0.7):
        is_square = True
    else:
        is_square = False

    # Used this method to check accuracy of my rectangle check
    # if (cv2.isContourConvex(building['cnt'])):
    #     print 'Rectangle'

    # Check shape conditions:
    # [] must have all corners and midpoints filled
    # + should have empty corners and all midpoints
    # I should have all corners but only 2 midpoints
    # C should have all corners but one midpoint missing
    # L should have 3 corners and only 2 midpoints
    # T should have 2 corners but all midpoints
    # Anything else is classified as 'irregular'
    if (corners_count == 4 and midpoints_count == 4):
        # because if it square, rectangular would be redundant
        if (is_square):
            descriptions.append('square')
        else:
            descriptions.append('rectangular')
    elif (corners_count == 0 and midpoints_count == 4):
        if (is_square):
            descriptions.append('squarish')
        descriptions.append('cross-shaped')
    elif (corners_count == 4 and midpoints_count == 2):
        descriptions.append('I-shaped')
    elif (corners_count == 4 and midpoints_count == 3):
        descriptions.append('C-shaped')
    elif (corners_count == 3 and midpoints_count == 2):
        descriptions.append('L-shaped')
    elif (corners_count == 2 and midpoints_count == 4):
        descriptions.append('T-shaped')
    else:
        descriptions.append('irregularly shaped')

    # Check orientation conditions:
    # If width is > 1.5 * height, "wide", E-W oriented
    # If height is > 1.5 * width, "tall", N-S oriented
    # Decided not to include symmetrically oriented
    if (w > 1.5 * h):
        descriptions.append('oriented East-West')
    elif (h > 1.5 * w):
        descriptions.append('oriented North-South')

    # print ' Description', descriptions
    return descriptions

def describe_location(building):
    # TODO: fix to buildings[21]
    college_walk = (137,322)
    marker = college_walk[1]
    map_h = len(map_binary)
    map_w = len(map_binary[0])

    h = building['mbr'][1][1] - building['mbr'][0][1]
    w = building['mbr'][1][0] - building['mbr'][0][0]

    # Reduce h/w shift so buildings are positioned properly
    h = h * 0.7
    w = w * 0.7

    cx = building['centroid'][0]
    cy = building['centroid'][1]

    location = []

    # Locate buildings on borders or central axis
    if (cx < w) and (cy < h):
        location.append('northwest corner')
    elif (cx > map_w-w) and (cy < h):
        location.append('northeast corner')
    elif (cx > map_w-w) and (cy > map_h-h):
        location.append('southeast corner')
    elif (cx < w) and (cy > map_h-h):
        location.append('southwest corner')
    elif (cx < w):
        location.append('western border')
    elif (cy < h):
        location.append('northern border')
    elif (cx > map_w-w):
        location.append('eastern border')
    elif (cy > map_h-h):
        location.append('southern border')

    # For buildings not on north/south borders, locate whether on
    # upper/central/lower campus
    if (cy > marker) and (cy < map_h-h):
        location.append('lower campus')
    elif (cy < marker) and (cy > (map_h-marker)/2):
        location.append('central campus')
    elif (cy > h) and (cy < (map_h-marker)/2):
        location.append('upper campus')

    # For buildings not on east/west borders
    if (cx > (map_w/2)-w) and (cx < (map_w/2)+w) and (cx > w) and (cx < map_w-w):
        location.append('on central axis')

    return location

def find_extrema(buildings):
    """Find extrema and return as list of tuple pairs of building index and extrema description"""
    num_buildings = len(buildings)
    extrema = []

    # Find largest and smallest by MBR area
    # We use MBR area instea of area because it's more apparent to the human eye
    # when a structure takes up more space in the bigger scheme of things
    sorted_buildings = sorted(buildings, key=lambda k:-(k['area']/k['extent']))
    areas = [(sorted_buildings[i]['number']-1) for i in range(num_buildings)]
    extrema.append(('biggest', areas[0]))
    extrema.append(('smallest', areas[-1]))
    return extrema


def print_info(buildings):
    for building in buildings:
        print building['number'], ':', building['name']
        print ' Minimum Bounding Rectangle:', building['mbr'][0], ',', building['mbr'][1]
        print ' Center of Mass:', building['centroid']
        print ' Area:', building['area']
        print ' Description', building['description']

# ============================================================
# The "Where"
# ============================================================

def analyze_relations(buildings):
    """Find all binary spatial relationships for every pair,
    and apply transitive reduction."""

    num_buildings = len(buildings)

    # Lookup tables
    n_table = np.zeros((num_buildings, num_buildings),bool)
    e_table = np.zeros((num_buildings, num_buildings),bool)
    s_table = np.zeros((num_buildings, num_buildings),bool)
    w_table = np.zeros((num_buildings, num_buildings),bool)
    near_table = np.zeros((num_buildings, num_buildings),bool)

    for source in xrange(0, num_buildings):
        for target in xrange(0, num_buildings):
            if source != target:
                s = buildings[source]
                t = buildings[target]
                # n_array[s][t] = is_north(s,t)

def is_north(s,t):
    """Find out if 'North of S is T'
    m1 = arctan(angle) left line
    m2 = arctan(angle) right line
    y  = mx + b
    Create fov triangle with 3 points
    Check if t is within the triangle
    # Draw triangle points first
    """
    map_h = len(map_binary)
    map_w = len(map_binary[0])

    # Experiment with this value
    # How about theta?
    angle = 95

    # 1. Calculate slopes m1 and m2
    m1 = np.arctan(angle)
    m2 = -np.arctan(angle)
    print "m1, m2", m1, m2

    # 2. Find b = y - mx using origin
    p0 = s['centroid'] # x,y
    b1 = p0[1] - m1*p0[0]
    b2 = p0[1] - m2*p0[0]
    print "b1, b2", b1, b2

    # 3. Calculate 2 other points in FOV triangle
    y = 100
    x1 = int((y-b1)/m1)
    x2 = int((y-b2)/m2)
    print "x1, x2", x1, x2
    cv2.circle(map_campus, (x1,y), 6, (255,0,255), -1)
    cv2.circle(map_campus, (x2,y), 6, (255,0,255), -1)


# def same_side(p1,p2,a,b):

# def is_in_triangle(p1):

# def is_near(s,t):
#     """Find out if 'Near to S is T'"""

# def transitive_reduce():
#     """Output should use building names rather than numbers"""


def is_index_valid(xy):
    x = xy[0]
    y = xy[1]
    map_h = len(map_binary)
    map_w = len(map_binary[0])
    if (x > 0) and (x < map_w) and (y > 0) and (y < map_h):
        return True
    else:
        return False

# ============================================================
# Main Invocation
# ============================================================

def main():

    # map_campus = np.zeros((512,512,3), np.uint8)

    # Analyze image
    names = load_names('ass3-table.txt')
    buildings = analyze_buildings(names)
    print_info(buildings)

    # Generate lookup table for building relations
    # relations = analyze_relations(buildings)
    # Try 11 Lowe and then 21 Journalism
    source = 11
    num_buildings = len(buildings)
    for target in xrange(0, num_buildings):
        if source != target:
            s = buildings[source]
            t = buildings[target]
            is_north(s,t)

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
