import cv2
import numpy as np
import sys
import math
# from matplotlib.path import Path

# ============================================================
# Globals
# ============================================================

np.set_printoptions(threshold=np.nan) # Set such that full image array is printed out
sys.setrecursionlimit(150000) # Reset python's default recursion limit (1000)

# 1. Basic Infrastructure
map_labeled = cv2.imread('ass3-labeled.pgm', 0) # Load labeled map as grayscale
map_campus = cv2.imread('ass3-campus.pgm', 1) # Load campus map(for display) as color
map_binary = cv2.cvtColor(map_campus,cv2.COLOR_BGR2GRAY) # Convert campus map to grayscale for contouring
MAP_H = len(map_binary)
MAP_W = len(map_binary[0])

buildings = []
num_buildings = 0
monument = {}

# 2. Spatial Relationships
n_table = []
e_table = []
s_table = []
w_table = []
near_table = []

# 3a. User Interface
drawing = False # true if mouse is pressed
mode = True # if True, generate path. Press 'm' to toggle to curve
ix,iy = -1,-1
click_count = -1
clicks = []
# Green, Red, Blue, Teal, Yellow, Orange, Magenta
# colors = [(0,255,0),(0,0,255),(255,0,0), (255,255,0), (0,255,255),(0,128,255),(255,0,255)]
# All Blue (user clicks)
colors = [(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0)]
color = colors[0] # Default

# 3b. Cloud Ambiguity
cloud = {}
called = {}
recursive_calls = 0
pix = 2 # Number of pixels to check in each direction for cloud generation

# 4. Path Generation
# S1G1: Broadway Gates -^ Mudd
# S2G2: Pupin -v Alma Mater
# S3G3: Carman -> Hartley
# S4G4: Kent <- Mathematics
# S5G5: Butler ^- Physical Fitness Center
# S6G6: Journalism -^ Uris
# S7G7: Avery ^- Shapiro
# S8G8: Lawn ^- Low
S_LIST = [(8,320),(35,4),(78,477),(232,285),(132,443),(52,398),(203,160),(135,369)]
G_LIST = [(205,51),(137,291),(257,374),(36,178),(88,68),(134,97),(143,37),(172,212)]
paths = [] # Will contain all the sequences of instructions for each 8 paths
path_parens = []
path_no_parens = []
itinerary_num = 0
user_responses = []
counter = 0

# ============================================================
# The "What"
# ============================================================

def load_names(filename):
    """Load files from text file in order"""
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

def analyze_what(names):
    """Find information about buildings and save in list of dicts"""
    global num_buildings, buildings
    num_buildings = len(names)
    buildings = list(np.zeros(num_buildings))
    areas = measure_areas()
    # print areas

    # Find contours in binary campus map image
    # Contours is a Python list of all the contours in the image
    # Each contour is a np array of (x,y) boundary points of each object
    contours,hierarchy = cv2.findContours(map_binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        building = {}
        idx = id_building(cnt)
        if idx is None:
            continue
        building['number'] = idx
        building['name'] = names[str(idx)]
        building['area'] = areas[str(idx)]
        mbr, centroid, extent, xywh = measure_building(cnt,building['area'])
        building['mbr'] = mbr
        building['centroid'] = centroid
        building['extent'] = extent
        building['xywh'] = xywh
        # Note: this was used by analyze_shapes and analyze_extents
        # building['cnt'] = cnt
        buildings[(idx-1)] = building

    max_area, min_area = analyze_areas(buildings) # add True arg to print results

    find_monument()

    for building in buildings:
        location = describe_location(building)
        size = describe_size(building, max_area)
        shape = describe_shape(building)
        if 'description' not in building:
            description = []
        else:
            description = building['description']
        if building['area'] is min_area: # replace with extrema
            description.append('smallest')
        else:
            description.append(size)
        description.extend(shape)
        description.extend(location)
        building['description'] = description

    # Reduce descriptions
    find_extrema()
    find_ambiguity()

    # multiple = describe_multiplicity
    # analyze_extents(buildings)
    # analyze_shapes(buildings)

def measure_areas():
    """Count areas for each building"""
    areas = {}
    for x in xrange(MAP_W):
        for y in xrange(MAP_H):
            pixel = map_labeled[(y,x)]
            if str(pixel) in areas:
                areas[str(pixel)] += 1
            else:
                areas[str(pixel)] = 1
    return areas

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

def measure_building(cnt, area, print_rect=False):
    """Use OpenCV to create a bounding rectangle and find center of mass"""
    # Let (x,y) be top-left coordinate and (w,h) be width and height
    # Find min, max value of x, min, max value of y
    x,y,w,h = cv2.boundingRect(cnt)
    xywh = (x,y,w,h)
    mbr = [(x,y),(x+w,y+h)]
    roi = map_campus[y:y+h,x:x+w]
    # To draw a rectangle, you need T-L corner and B-R corner
    # We have mbr[0] = T-L corner, mbr[1] = B-R corner
    if print_rect:
        cv2.rectangle(map_campus,(x,y),(x+w,y+h),(200,0,0),2)
    # print " Minimum Bounding Rectangle: ({0},{1}), ({2},{3})".format(x,y,(x+w),(y+h))

    # Calculate centroid based on bounding rectangle
    cx = x+(w/2)
    cy = y+(h/2)
    centroid = (cx, cy)

    # DRAW CENTROIDS!
    # cv2.circle(map_campus, centroid, 3, (255,255,0), -1)
    # To draw a circle, you need its center coordinates and radius
    # print ' Center of Mass:', centroid

    rect_area = w*h
    extent = float(area)/rect_area

    # Discarded methods
    # Image moments help you to calculate center of mass, area of object, etc.
    # cv2.moments() gives dictionary of all moment values calculated
    # M = cv2.moments(cnt)
    # Centroid is given by the relations
    # cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])
    # centroid = (cx, cy)

    # Contour area is given by the function cv2.contourArea(cnt) or
    # area = M['m00']
    # print ' Area:', area
    # area = cv2.contourArea(cnt)
    # x,y,w,h = cv2.boundingRect(cnt)
    # rect_area = w*h
    # extent = float(area)/rect_area
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

    return mbr, centroid, extent, xywh

def analyze_areas(buildings, print_results=False):
    """Sort buildings by area, determine cutoff for size and return max"""
    # num_buildings = len(buildings)
    sorted_buildings = sorted(buildings, key=lambda k:-k['area'])
    indices = [(sorted_buildings[i]['number']-1) for i in range(num_buildings)]
    areas = [(sorted_buildings[i]['area']) for i in range(num_buildings)]

    max_area = areas[0]
    avg_area = sum(areas)/num_buildings
    min_area = areas[-1]

    # Print results to analyze cutoffs for size categories
    if (print_results):
        print 'Analyzing building areas...'
        ratios = [round(float(areas[i])/max_area,3) for i in range(num_buildings)]
        ratio_diffs = [round((ratios[i+1]-ratios[i]),3) for i in range(num_buildings-1)]
        ratio_diffs.insert(0,0)
        max_area_ratios = [round(max_area/areas[i],3) for i in range(num_buildings)]

        print 'Max Area:', max_area
        print 'Average:', avg_area
        print 'Min Area:', min_area
        print 'Area\tRatio r\tDiff r\tMax r\tBuilding'
        for i in xrange(num_buildings):
            idx = indices[i]
            print areas[i], '\t', ratios[i], '\t', ratio_diffs[i], '\t', max_area_ratios[i], '\t', idx+1, buildings[idx]['name']

    return max_area, min_area

def analyze_extents():
    """Sort buildings by extent and determine cutoff for rectangles"""
    print 'Analyzing building extents (area/mbr) and convexity...'
    # num_buildings = len(buildings)
    sorted_buildings = sorted(buildings, key=lambda k:-k['extent'])
    indices = [(sorted_buildings[i]['number']-1) for i in range(num_buildings)]
    for i in indices:
        building = buildings[i]
        convex = cv2.isContourConvex(building['cnt'])
        print round(building['extent'],4), '\t', convex, '\t', i+1, building['name']

def analyze_shapes(buildings):
    """Sort building by shape similarity (not very good results)"""
    print 'Analyzing shape similarity with cv2.matchShapes...'
    # num_buildings = len(buildings)
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
    ratio = float(building['area'])/max_area
    if ratio > 0.7: # cutoff at College Walk
        return 'colossal'
    elif ratio > 0.4: # cutoff at Journalism & Furnald
        return 'large'
    elif ratio > 0.16: # cutoff at Philosophy
        return 'midsized'
    elif ratio > 0.1: # cutoff Earl Hall
        return 'small'
    else:
        return 'tiny'

def describe_shape(building,draw_points=False):
    """Describe shape based on corner and midpoint counts"""

    descriptions = []

    xywh = building['xywh']
    corners_count, midpoints_count, xywh2 = count_points(building,xywh,draw_points)

    # print building['number'], building['name']
    # print ' Tolerance', tolerance
    # print '', corners_filled, 'Corners Count', corners_count
    # print '', midpoints_filled, 'Midpoints Count', midpoints_count

    # Difference between height and width should be small enough
    # Decided not to use absolute value as differnce is relative
    # Also check that building fills out most of the MBR
    # Ruling out Journalism & Furnald, and Chandler & Havemeyer
    x,y,w,h = unpack(xywh)
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
            descriptions.append('squarish cross-shaped')
        else:
            cc, mc, xywh2 = count_points(building,xywh2,draw_points)
            if (cc%2 == 1): # Not symmetrical
                descriptions.append('bell-shaped')
            else:
                descriptions.append('cross-shaped')
    elif (corners_count == 4 and midpoints_count == 2):
        descriptions.append('I-shaped')
    elif (corners_count == 4 and midpoints_count == 3):
        descriptions.append('U-shaped')
    elif (corners_count == 3 and midpoints_count == 2):
        descriptions.append('L-shaped')
    elif (corners_count == 2 and midpoints_count == 4):
        descriptions.append('almost rectangular')
    else:
        descriptions.append('irregularly shaped')

    # Check orientation conditions:
    # If width is > 1.5 * height, "wide", E-W oriented
    # If height is > 1.5 * width, "tall", N-S oriented
    # Decided not to include symmetrically oriented
    # if (w > 1.5 * h):
    #     descriptions.append('oriented East-West')
    # elif (h > 1.5 * w):
    #     descriptions.append('oriented North-South')

    # print ' Description', descriptions
    return descriptions

def unpack(tup):
    if len(tup) is 4:
        return tup[0],tup[1],tup[2],tup[3]
    elif len(tup) is 5:
        return tup[0],tup[1],tup[2],tup[3],tup[4]

def count_points(building,xywh,draw_points):
    x,y,w,h = unpack(xywh)

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
            if draw_points:
                cv2.circle(map_campus, corner, 1, (255,255,0), -1)
        else:
            corners_filled.append(0)
            if draw_points:
                cv2.circle(map_campus, corner, 1, (0,0,255), -1)

    for midpoint in midpoints:
        if map_labeled[tuple(reversed(midpoint))] == building['number']:
            midpoints_filled.append(1)
            if draw_points:
                cv2.circle(map_campus, midpoint, 1, (0,255,0), -1)
        else:
            midpoints_filled.append(0)
            if draw_points:
                cv2.circle(map_campus, midpoint, 1, (0,0,255), -1)

    # Count the number of corners and midpoints for each building
    # Not necessary to consider order at this point
    corners_count = corners_filled.count(1)
    midpoints_count = midpoints_filled.count(1)

    return corners_count, midpoints_count, (x,y,w,h)

def find_monument():
    global monument, buildings
    for idx in xrange(num_buildings):
        if buildings[idx]['xywh'][2] > MAP_W - 10:
            monument = buildings[idx]
            buildings[idx]['description'] = ['longest']
            return

def describe_location(building):
    if building['number'] is monument['number']:
        return []

    location = []
    marker = monument['centroid'][1] # cy for College Walk

    h = building['mbr'][1][1] - building['mbr'][0][1]
    w = building['mbr'][1][0] - building['mbr'][0][0]

    # Reduce h/w shift so buildings are positioned properly
    h = int(h * 0.7)
    w = int(w * 0.7)

    cx = building['centroid'][0]
    cy = building['centroid'][1]

    # Draw lines
    # if building['number'] is 10:
    #     cv2.line(map_campus,(0,marker/2),(MAP_W,marker/2),[0,255,0],2)
    #     cv2.line(map_campus,(0,marker),(MAP_W,marker),[0,255,0],2)
    #     cv2.line(map_campus,(0,h),(MAP_W,h),[0,255,0],2)
    #     cv2.line(map_campus,(0,MAP_H-h),(MAP_W,MAP_H-h),[0,255,0],2)
    #     cv2.line(map_campus,(int((MAP_W/2)-w),0),(int((MAP_W/2)-w),MAP_H),[0,255,0],2)
    #     cv2.line(map_campus,(int((MAP_W/2)+w),0),(int((MAP_W/2)+w),MAP_H),[0,255,0],2)
    #     cv2.line(map_campus,(w,0),(w,MAP_H),[0,255,0],2)
    #     cv2.line(map_campus,(MAP_W-w,0),(MAP_W-w,MAP_H),[0,255,0],2)

    # Locate buildings on borders or central axis
    if (cx < w) and (cy < h):
        location.append('northwest corner')
    elif (cx > MAP_W-w) and (cy < h):
        location.append('northeast corner')
    elif (cx > MAP_W-w) and (cy > MAP_H-h):
        location.append('southeast corner')
    elif (cx < w) and (cy > MAP_H-h):
        location.append('southwest corner')
    elif (cy < h):
        location.append('northernmost')
    elif (cy > MAP_H-h):
        location.append('southernmost')
    elif (cx > MAP_W-w):
        location.append('easternmost')
    elif (cx < w):
        location.append('westernmost')

    # For buildings not on north/south borders, locate whether on
    # upper/central/lower campus
    if (cy > marker) and (cy < MAP_H-h): # southernmost already weeded out
        location.append('lower campus')
    elif (cy > h) and (cy < marker/2):
        location.append('upper campus')
    elif (cy < marker) and (cy > marker/2) and (cx < MAP_W-w) and (cx > w): # central_axis(cx,w):
        location.append('central campus')

    # For buildings not on east/west borders
    # if (cx > (MAP_W/2)-w) and (cx < (MAP_W/2)+w) and (cx > w) and (cx < MAP_W-w):
    #     location.append('on central axis')

    return location

def central_axis(cx,w):
    if (cx > (MAP_W/2)-w) and (cx < (MAP_W/2)+w) and (cx > w) and (cx < MAP_W-w):
        return True
    return False

def counting_dict(dic,key):
    if key in dic:
        dic[key] += 1
    else:
        dic[key] = 1
    return dic

def find_extrema():
    """Find singularly defining characteristics and remove other details"""
    global buildings
    characteristics = {}
    for idx in xrange(num_buildings):
        bldg1 = buildings[idx]
        description = bldg1['description']
        for characteristic in description:
            # print characteristic
            count = 0
            # Add to counting dictionary
            characteristics = counting_dict(characteristics, characteristic)
            for jdx in xrange(num_buildings):
                bldg2 = buildings[jdx]
                if (idx != jdx) and (characteristic in tuple(bldg2['description'])):
                    count += 1
            if count is 0 and characteristic != 'almost rectangular' and characteristic != 'southernmost':
                # 'Found extrema!', characteristic
                extrema = [characteristic]
                bldg1['description'] = extrema
                buildings[idx] = bldg1
    return characteristics

def find_ambiguity():
    global buildings
    for idx in xrange(num_buildings):
        bldg1 = buildings[idx]
        for jdx in xrange(num_buildings):
            bldg2 = buildings[jdx]
            if idx != jdx and bldg1['description'] == bldg2['description']:
                if is_north(bldg1,bldg2):
                    bldg2['description'].insert(0,'more northern')
                    bldg1['description'].insert(0,'more southern')
                elif is_south(bldg1,bldg2):
                    bldg1['description'].insert(0,'more northern')
                    bldg2['description'].insert(0,'more southern')
                buildings[idx] = bldg1
                buildings[jdx] = bldg2
                # print 'Ambiguity between', bldg1['name'], 'and', bldg2['name']

def print_info():
    """System output for part 1"""
    for building in buildings:
        print building['number'], ':', building['name']
        print '     Minimum Bounding Rectangle:', building['mbr'][0], ',', building['mbr'][1]
        print '     Center of Mass:', building['centroid']
        print '     Area:', building['area']
        print '     Description:', building['description']

# ============================================================
# The "Where"
# ============================================================

def analyze_where(buildings):
    """Find all binary spatial relationships for every pair,
    and apply transitive reduction."""

    global n_table, e_table, s_table, w_table, near_table

    n_table = np.zeros((num_buildings, num_buildings),bool)
    e_table = np.zeros((num_buildings, num_buildings),bool)
    s_table = np.zeros((num_buildings, num_buildings),bool)
    w_table = np.zeros((num_buildings, num_buildings),bool)
    near_table = np.zeros((num_buildings, num_buildings),bool)

    for s in xrange(0, num_buildings):
        for t in xrange(0, num_buildings):
            if s != t:
                source = buildings[s]
                target = buildings[t]
                n_table[s][t] = is_north(source,target)
                s_table[s][t] = is_south(source,target)
                e_table[s][t] = is_east(source,target)
                w_table[s][t] = is_west(source,target)
                near_table[s][t] = is_near(source,target)

    print 'North relationships:'
    count = print_table(n_table, num_buildings)
    print 'South relationships:'
    count += print_table(s_table, num_buildings)
    print 'East relationships:'
    count += print_table(e_table, num_buildings)
    print 'West relationships:'
    count += print_table(w_table, num_buildings)
    print 'Near relationships:'
    count += print_table(near_table, num_buildings)
    print 'Total count:', count

    n_table, s_table, e_table, w_table, near_table = transitive_reduce(n_table, s_table, e_table, w_table, near_table)

    print 'After transitive reduction...'
    print 'North relationships:'
    count = print_table(n_table, num_buildings)
    print 'South relationships:'
    count += print_table(s_table, num_buildings)
    print 'East relationships:'
    count += print_table(e_table, num_buildings)
    print 'West relationships:'
    count += print_table(w_table, num_buildings)
    print 'Near relationships:'
    count += print_table(near_table, num_buildings)
    print 'Total count:', count

    print_table_info(n_table, buildings, 'North')
    print_table_info(s_table, buildings, 'South')
    print_table_info(e_table, buildings, 'East')
    print_table_info(w_table, buildings, 'West')
    print_table_info(near_table, buildings, 'Near')

    return n_table, s_table, e_table, w_table, near_table

def analyze_single_where(source, direction, buildings):
    """Analyze relations for single building"""
    # Try 11 Lowe and then 21 Journalism
    # num_buildings = len(buildings)
    for target in xrange(0, num_buildings):
        if source != target:
            s = buildings[source]
            t = buildings[target]
            if (direction == "north"):
                triangulate_FOV(s,t,-1,0,1,draw=True)
            elif (direction == "east"):
                triangulate_FOV(s,t,MAP_W,-1,1.2,draw=True)
            elif (direction == "south"):
                triangulate_FOV(s,t,-1,MAP_H,1,draw=True)
            elif (direction == "west"):
                triangulate_FOV(s,t,0,-1,1.2,draw=True)

def is_north(s,t):
    """Find out if 'North of S is T'"""
    # Form triangle to north border: (x,0)
    return triangulate_FOV(s,t,-1,0,0.8)

def is_south(s,t):
    """Find out if 'South of S is T'"""
    # Form triangle to south border: (x,MAP_H)
    return triangulate_FOV(s,t,-1,MAP_H,0.8)

def is_east(s,t):
    """Find out if 'East of S is T'"""
    # Form triangle to east border: (MAP_W,y)
    return triangulate_FOV(s,t,MAP_W,-1,1.5)

def is_west(s,t):
    """Find out if 'West of S is T'"""
    # Form triangle to west border: (0,y)
    return triangulate_FOV(s,t,0,-1,1.5)

def triangulate_FOV(s,t,x,y,slope,draw=False):
    """Create a triangle FOV with 3 points and
    check if t is within triangle"""

    # Check if input is a building (if so, leave it)
    # or an int (if so, change to a building)
    if type(s) == int and type(t) == int:
        s = buildings[s]
        t = buildings[t]

    if y is 0:
        fov = 'north_fov'
    elif y is MAP_H:
        fov = 'south_fov'
    elif x is MAP_W:
        fov = 'east_fov'
    elif x is 0:
        fov = 'west_fov'

    if fov not in s:
        # 0. Find (x,y) for source and target
        p0 = s['centroid']
        p4 = t['centroid']

        # 1. Determine slopes m1 and m2
        # if (s['number'] == 21):
        #     slope = 3
        m1 = slope
        m2 = -slope
        # print "m1, m2", m1, m2

        # 2. Find b = y - mx using origin and slope
        b1 = p0[1] - m1*p0[0]
        b2 = p0[1] - m2*p0[0]
        # print "b1, b2", b1, b2

        # 3. Calculate 2 other points in FOV triangle
        # Direction is determined by what x or y values
        # are given for p1 and p2
        if (x == -1): # y given, so North/South direction
            x1 = int((y-b1)/m1)
            x2 = int((y-b2)/m2)
            # print "x1, x2", x1, x2
            p1 = (x1,y)
            p2 = (x2,y)

        elif (y == -1): # x given, so East/West direction
            y1 = int((m1*x) + b1)
            y2 = int((m2*x) + b2)
            # print "y1, y2", x1, y2
            p1 = (x,y1)
            p2 = (x,y2)

        if (draw == True):
            cv2.line(map_campus,p0,p1,(0,255,0),2)
            cv2.line(map_campus,p0,p2,(0,255,0),2)

        # Mandatory: Add new FOV to building dictionary for reuse
        s[fov] = (p0,p1,p2)
        idx = s['number'] - 1
        buildings[idx] = s

    # If FOV has been pre-calculated, just use the points to check
    else:
        p0 = s[fov][0]
        p1 = s[fov][1]
        p2 = s[fov][2]
        p4 = t['centroid']

    # 4. Check whether target centroid is in the field of view
    if is_in_triangle(p4,p0,p1,p2):
        if (draw == True):
            cv2.circle(map_campus, p4, 6, (0,255,0), -1)
        return True

    # Special case for campus-wide College Walk, add centroids
    if (t['number'] == monument['number']):
        mid = t['centroid']
        p5 = (MAP_W/5,mid[1])
        p6 = (MAP_W*4/5,mid[1])
        if is_in_triangle(p5,p0,p1,p2):
            if (draw == True):
                cv2.circle(map_campus, p5, 6, (0,255,0), -1)
            return True
        elif is_in_triangle(p6,p0,p1,p2):
            if (draw == True):
               cv2.circle(map_campus, p6, 6, (0,255,0), -1)
            return True
    return False # if not in FOV, return false

    # 4. Check whether target centroid is in the field of view
    if is_in_triangle(p3,p0,p1,p2):
        if (draw == True):
            cv2.circle(map_campus, p3, 6, (0,255,0), -1)
        return True

    # Special case for campus-wide College Walk, add centroids
    if (t['number'] == monument['number']):
        mid = t['centroid']
        p5 = (MAP_W/5,mid[1])
        p6 = (MAP_W*4/5,mid[1])
        if is_in_triangle(p5,p0,p1,p2):
            if (draw == True):
                cv2.circle(map_campus, p5, 6, (0,255,0), -1)
            return True
        elif is_in_triangle(p6,p0,p1,p2):
            if (draw == True):
               cv2.circle(map_campus, p6, 6, (0,255,0), -1)
            return True

    return False # if not in FOV, return false

def same_side(p1,p2,a,b):
    cp1 = np.cross(np.subtract(b,a), np.subtract(p1,a))
    cp2 = np.cross(np.subtract(b,a), np.subtract(p2,a))
    if np.dot(cp1,cp2) >= 0:
        return True
    else:
        return False

def is_in_triangle(p,a,b,c):
    if same_side(p,a,b,c) and same_side(p,b,a,c) and same_side(p,c,a,b):
        return True
    else:
        return False

def shift_corners(building, shift):
    # Shift should be negative if you want to tuck in points
    x,y,w,h = unpack(building['xywh'])
    # Shift x,y,w,h so corners and midpoints are closer/farther to center
    x -= shift
    y -= shift
    w += 2*shift
    h += 2*shift
    return x,y,w,h

def extract_corners(x,y,w,h):
    nw = (x,y)
    ne = (x+w,y)
    se = (x+w,y+h)
    sw = (x,y+h)
    return nw,ne,se,sw

def draw_rectangle(nw,ne,se,sw):
    cv2.line(map_campus,nw,ne,(0,128,255),2)
    cv2.line(map_campus,ne,se,(0,128,255),2)
    cv2.line(map_campus,se,sw,(0,128,255),2)
    cv2.line(map_campus,sw,nw,(0,128,255),2)
    if (diagonal):
        cv2.line(map_campus,nw,se,(0,128,255),2)

def draw_triangle(p1,p2,p3):
    cv2.line(map_campus,p1,p2,(0,128,255),2)
    cv2.line(map_campus,p2,p3,(0,128,255),2)
    cv2.line(map_campus,p3,p1,(0,128,255),2)

def get_near_points(building,shift):
    if 'near_points' not in building: # or building['number'] > num_buildings:
        # Extract four corners: nw,ne,se,sw
        x1,y1,w1,h1 = shift_corners(building,shift)
        p1,p2,p3,p4 = extract_corners(x1,y1,w1,h1)
        p0 = building['centroid']
        points = (p1,p2,p3,p4,p0)
        # draw_rectangle(p1,p2,p3,p4)
        # Add new points to source
        building['near_points'] = points
        idx = building['number'] - 1
        buildings[idx] = building
    else:
        points = building['near_points']
    return points

def is_near(s,t,draw=False):
    """Near to S is T"""
    if type(s) == int and type(t) == int:
        s = buildings[s]
        t = buildings[t]

    shift = 15 # Empirically chosen
    s_points = get_near_points(s,shift)
    t_points = get_near_points(t,shift)

    s1,s2,s3,s4,s0 = unpack(s_points)
    t1,t2,t3,t4,t0 = unpack(t_points)

    # Check whether any corner in expanded target rectangle
    # lies inside one of the two triangles that form the
    # source rectangle
    for pt in t_points:
        if is_in_triangle(pt,s1,s2,s3) or is_in_triangle(pt,s3,s4,s1):
            # Optional
            if (draw):
                if is_in_triangle(pt,s1,s2,s3):
                    draw_triangle(s1,s2,s3)
                else:
                    draw_triangle(s3,s4,s1)
                # draw_rectangle(t1,t2,t3,t4)
                cv2.circle(map_campus, s0, 6, (0,128,255), -1)
                cv2.circle(map_campus, t0, 6, (0,128,255), -1)
                cv2.circle(map_campus, pt, 6, (0,128,255), -1)
                cv2.circle(map_campus, pt, 3, (0,255,255), -1)
            # Mandatory
            return True
    return False

def transitive_reduce(n_table, s_table, e_table, w_table, near_table):
    """Output should use building names rather than numbers"""
    # TODO: Uncomment these and explain
    for t in range(0, num_buildings):
        for s in range(0, num_buildings):
            if n_table[s][t]:
                for u in range(0, num_buildings):
                    if n_table[t][u]:
                        n_table[s][u] = False
            if s_table[s][t]:
                for u in range(0, num_buildings):
                    if s_table[t][u]:
                        s_table[s][u] = False
            if w_table[s][t]:
                for u in range(0, num_buildings):
                    if w_table[t][u]:
                        w_table[s][u] = False
            if e_table[s][t]:
                for u in range(0, num_buildings):
                    if e_table[t][u]:
                        e_table[s][u] = False

    # If t is north of s we no longer need to say s is south of t
    # Similarly, east west relationships can be inferred
    for s in range(0, num_buildings):
        for t in range(0, num_buildings):
            if n_table[s][t] and s_table[t][s]:
                s_table[t][s] = False
            if e_table[s][t] and w_table[t][s]:
                w_table[t][s] = False

    # If relationship is reflexive, keep the smaller building's relationship
    for s in xrange(0, num_buildings):
        for t in xrange(0, num_buildings):
            source = buildings[s]
            target = buildings[t]
            if near_table[s][t] and near_table[t][s]:
                if source['area'] > target['area']:
                    near_table[s][t] = False
                else:
                    near_table[t][s] = False
            elif near_table[s][t] and not near_table[t][s]:
                print 'Near to', source['name'], 'is', target['name'], 'but not other way around'

    return n_table, s_table, e_table, w_table, near_table

def print_table_info(table, buildings, direction):
    # num_buildings = len(buildings)
    # Track printed source indices so they are only printed once
    printed = 0
    for s in xrange(0, num_buildings):
        for t in xrange(0, num_buildings):
            if table[s][t]:
                target = buildings[t]
                source = buildings[s]
                if printed < s:
                    printed += 1
                    if direction is 'Near':
                        print 'Near to', source['name'], 'is:'
                    else:
                        print direction, 'of', source['name'], 'is:'
                print '    ', target['name']

def print_table(table,num_buildings):
    count = 0
    print '  ',
    for s in xrange(num_buildings):
        if s < 9:
            print '', s+1,
        elif s == 9:
            print '', s+1,
        else:
            print s+1,
    print ''
    for s in xrange(num_buildings):
        for t in xrange(num_buildings):
            if t == 0:
                if s < 9:
                    print '', s+1, '',
                else:
                    print s+1, '',
            if table[s][t]:
                count += 1
                print 1, '',
            else:
                print '  ',
            if t == num_buildings-1:
                print '\n',

    print 'Number of true relationships:', count
    return count


# ============================================================
# User Interface
# ============================================================

# mouse callback function
def click_event(event,x,y,flags,param):
    global ix,iy,drawing,mode,click_count,color,itinerary_num,map_campus,counter

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        ix,iy = x,y
        clicks.append((ix,iy))
        # counter += 1
        # ix, iy = intercept_click(ix,iy)

        if mode == True: # User tests
            change_color() # and increment click count
            # print 'iter num: ', itinerary_num
            # print 'counter', counter
            # print '<len(path_parens[itinerary_num])', len(path_parens[itinerary_num])
            # print '==len(path_parens[itinerary_num]-1)', len(path_parens[itinerary_num])-1
            # print '>len(path_parens[itinerary_num])', len(path_parens[itinerary_num])
            if counter < len(path_parens[itinerary_num]):
                print 'Clicked location: ({},{})'.format(ix,iy)
                counter += 1
                # print 'Click count:', len(clicks)
            if counter == len(path_parens[itinerary_num])-1:
                end = G_LIST[itinerary_num]
                print 'Final destination:', end
                clicks.append((ix,iy))
                counter += 1
                print 'Distance: ', get_euclidean_distance(end, clicks[-1])
                cv2.circle(map_campus,end,6,(0,0,255),-1)
                itinerary_num += 1
                user_responses.append(clicks[-1])
                print
                print 'Good job! Next itinerary! Click any white space to begin.'
                print '------'
                # Save results
                cv2.imwrite('iter'+str(itinerary_num)+'.png', map_campus);
            elif counter > len(path_parens[itinerary_num]):
                # Reset counter
                counter = 0
                color = (255,255,255)
                # Reload image
                map_campus = cv2.imread('ass3-campus.pgm', 1)
                cv2.imshow('Columbia Campus Map', map_campus)
                start = S_LIST[itinerary_num]
                # print new start
                cv2.circle(map_campus,start,6,(0,255,0),-1)
                print_instructions()

        else: # Cloud Ambiguity
            # Function to test ALL clouds for largest/smallest
            # test_clouds()
            idx = create_building(ix,iy)
            change_color() # and increment click count
            pixels = pixel_cloud(ix,iy) # Generate cloud of all similar pixels

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.circle(map_campus,(ix,iy),6,color,-1)
        else:
            cv2.circle(map_campus,(ix,iy),pix,color,-1)
            # white dot indicates original click location
            cv2.circle(map_campus,(ix,iy),1,(255,255,255),-1)

        # if mode == True:
        #     # cv2.rectangle(map_campus,(ix,iy),(x,y),(0,255,0),-1)
        # else:
        #     cv2.circle(map_campus,(x,y),pix/2,(255,255,255),-1)

def intercept_click(ix,iy):
    """Helper function that intercepts click values and changes to desired test"""
    if click_count%2 == 0: # Target
        ix,iy = 50,430 # Smallest
        # ix,iy = 90, 400 # New Largest
    else: # Source
        ix,iy = 70,210 # Largest
        # ix,iy = 130, 340 # Second Largest
        # ix,iy = 10, 190 # Small
    return ix,iy

def change_color():
    global color, click_count
    # alternate colors based on clicks
    if click_count >= len(colors)-1: # reset
        click_count = 0
    else:
        click_count += 1
    color = colors[click_count]

# ============================================================
# Source and Target Description
# ============================================================

def create_building(x,y):
    global buildings
    # idx = int(map_labeled[y][x])
    # add new x,y as a new building
    idx = len(buildings)
    building = {}
    building['number'] = len(buildings)+1
    building['name'] = 'Building ' + str(len(buildings)+1)
    building['centroid'] = (x,y)
    building['xywh'] = (x,y,1,1)
    buildings.append(building)
    # num_buildings = len(buildings)
    return idx

def pixel_cloud(x,y):
    global color, cloud, recursive_calls, called
    # Reset cloud every time this function is called
    cloud = {}
    relationships = []
    recursive_calls = 0
    called = {}

    # To copy numpy arrays:
    # a = np.zeros((27,27),bool)
    # b = np.zeros((28,28),bool)
    # b[:-1,:-1] = a

    # for num in xrange(0, num_buildings-1-click_count):
    for num in xrange(num_buildings):
        s = buildings[num]
        t = buildings[-1] # the newly added building
        # Note these methods require xywh, centroid, number
        idx = int(map_labeled[y][x]) - 1
        # near = xy_near(s,x,y)
        # near = is_near(s,t) # Keep smaller (1 pixel) building's relationship
        near = is_near(s,t) or is_near(t,s)
        relationships.append([is_north(s,t), is_south(s,t), is_east(s,t), is_west(s,t),near,num,idx])
        # relationships.append([is_north(s,t), is_east(s,t), is_near(s,t),idx])
    # print "Relationships:", relationships

    relationships, sorted_indices = reduce_by_nearness(relationships)
    # print 'New relationships:', relationships

    # Recursively generate ambiguity cloud based on pruned relationships and sorted indices
    flood_fill(x,y,relationships,sorted_indices)

    # Color in the cloud
    for xy in cloud:
        col = xy[0]
        row = xy[1]
        # map_campus[row][col] = [0,255,0]
        # Draw filled circle with radius of 5
        cv2.circle(map_campus,(col,row),pix/2,color,-1)

    description = ts_description(x,y,relationships,sorted_indices)
    print description

    cloud_size = len(cloud) * pix
    print '     Size of cloud:', cloud_size, '(recursive calls: %d)\n' %recursive_calls

    return cloud_size

def reduce_by_nearness(relationships):
    # Experiment with limit
    # Increasing it does not shrink ambiguity by much
    # Users seem confused by more than 3 descriptions
    limit = 3
    distances_to = {}
    for i in xrange(num_buildings):
        # Only keep near relationships
        if relationships[i][4] == False:
            # Change all values to False (ignore)
            relationships[i][:5] = [False,False,False,False,False]
        else:
        # Of the remaining 'near' relationships, sort by distance
            s = relationships[i][5]
            t = -1 # Last added building to list of buildings
            dist = get_euclidean_distance(s,t)
            distances_to[str(s)] = dist

    # Keep relationships only with three closest structures
    sorted_distances = sorted(distances_to.items(), key=lambda k:k[1])

    # Special case: if click is inside building, its color value - 1
    # (its building index) should be at start of list
    click_idx = relationships[0][-1]
    if click_idx == -1: # Outside
        sorted_indices = [int(tup[0]) for tup in sorted_distances]
    else: # Inside
        sorted_indices = [int(tup[0]) for tup in sorted_distances if int(tup[0]) != click_idx]
        sorted_indices.insert(0,click_idx)
    # If there more than three structures indicated, set rest to be ignored
    if len(sorted_indices) > limit:
        for n in xrange(limit,len(sorted_indices)):
            idx = sorted_indices[n]
            relationships[idx][:5] = [False,False,False,False,False]
        # Prune the list of indices to contain only the limit
        sorted_indices = sorted_indices[:limit]
    # print 'Sorted distances:', sorted_distances
    # print 'Distances:', distances_to
    # print 'Sorted indices:', sorted_indices
    # print 'New relationships:', relationships
    return relationships, sorted_indices

def flood_fill(x, y, rel_table, indices):
    """Recursive algorithm that starts at x and y and changes any
    adjacent pixel that match rel_table"""
    global cloud, called, recursive_calls

    if (x,y) in called:
        return
    else:
        recursive_calls += 1
        called[(x,y)] = ''

    # print recursive_calls, ':', x,y

    rel = []
    # for num in range(0, num_buildings-1-click_count):
    for num in xrange(num_buildings):
        s = buildings[num]
        t = buildings[-1]
        t['centroid'] = (x,y) # change centroid to new x,y
        t['xywh'] = (x,y,100,100)
        if 'near_points' in t:
            del t['near_points']
        buildings[t['number']-1] = t
        idx = int(map_labeled[y][x]) - 1
        # Only check relevant relations
        if num in tuple(indices):
            # near = xy_near(s,x,y)
            # near = is_near(s,t) # Keep smaller (1 pixel) building's relationship
            near = is_near(s,t) or is_near(t,s)
            if (near):
                rel.append([is_north(s,t), is_south(s,t), is_east(s,t), is_west(s,t),near,num,idx])
            else:
                rel.append([False,False,False,False,False,num,idx])
        # Else set all values to default False
        else:
            rel.append([False,False,False,False,False,num,idx])

    # print 'Flood Fill Rel:', rel

    # Base case. If the current x,y is not the right rel do nothing
    if rel != rel_table:
        return

    # Add pixel to list of clouds to be recolored and used later
    cloud[(x,y)] = ''

    # Recursive calls. Make a recursive call as long as we are not
    # on boundary

    if x > (pix-1): # left # originally 0
        flood_fill(x-pix, y, rel_table, indices)

    if y > (pix-1): # up # originally 0
        flood_fill(x, y-pix, rel_table, indices)

    if x < MAP_W-(pix+1): # right # originally MAP_W-1
        flood_fill(x+pix, y, rel_table, indices)

    if y < MAP_H-(pix+1): # down # originall MAP_H-`
        flood_fill(x, y+pix, rel_table, indices)

def test_clouds():
    """Check clouds of every other 10 pixels in the map
    and lists the xy coordinates sorted by cloud size"""
    clouds = []
    min_cloud = (0,0,10)
    max_cloud = (0,0,10)
    for x in xrange(MAP_W):
        for y in xrange(MAP_H):
            if (x%10 == 0) and (y%10 == 0):
                idx = create_building(x,y)
                # change_color() # don't draw
                size = pixel_cloud(x,y)
                if (size < min_cloud[2]):
                    min_cloud = (x,y,size)
                elif (size > max_cloud[2]):
                    max_cloud = (x,y,size)
                clouds.append((x,y,size))
    sorted_clouds = sorted(clouds, key=lambda k:-k[2])
    print 'Max cloud', max_cloud
    print 'Min cloud', min_cloud
    print 'Sorted clouds', sorted_clouds

def index_valid(x,y):
    x = xy[0]
    y = xy[1]
    if (x > 0) and (x < MAP_W) and (y > 0) and (y < MAP_H):
        return True
    else:
        return False

def what_description(idx):
    global buildings
    what = 'the '
    descr = buildings[idx]['description']
    for i in xrange(len(descr)):
        if i < len(descr)-1:
            what += descr[i] + ', '
        else:
            what += descr[i] + ' structure'
    return what

def ts_description(x, y, relationships, sorted_indices):
    coordinates = '     Click (%d,%d)' %(x,y)
    if click_count%2 == 1:
        print 'TARGET: ' #+ coordinates
        # description = 'Then go to the building that is '
    else:
        print 'SOURCE: ' #+ coordinates
        # description = 'Go to the nearby building that is '

    # Check if click point is outside or inside
    if (relationships[0][-1] == -1):
        description = coordinates + ' is '
    else:
        description = coordinates + ' is INSIDE and to the '

    # print 'Sorted indices:', sorted_indices
    # print 'Relationships:', relationships
    # for idx in range(0, num_buildings-1):
    rel_count = 0
    for idx in sorted_indices:
        count = 0
        if relationships[idx][0]:
            description += 'NORTH of '
            count += 1
        if relationships[idx][1]:
            description += 'SOUTH of '
            count += 1
        if relationships[idx][2]:
            if count == 0:
                count += 1
            else:
                description = description[:-4]
            description += 'EAST of '
        if relationships[idx][3]:
            if count == 0:
                count += 1
            else:
                description = description[:-4]
            description += 'WEST of '
        # Implied nearness
        # if relationships[idx][4]:
        #     if count == 0:
        #         descr += "near "
        #         count += 1
        #     else:
        #         desc += "and near "
        if count != 0:
            description += what_description(idx)
            description += ' (%s), ' %buildings[idx]['name']
            rel_count += 1
            if sorted_indices[1] == -1: # Only one descriptor
                break
            if sorted_indices[0] == -1 and rel_count == len(sorted_indices)-2:
                description += 'and '
            elif sorted_indices[0] != -1 and rel_count == len(sorted_indices)-1:
                description += 'and '
    description = description[:-2] + '.'
    return description

# ============================================================
# Path Generation
# ============================================================

def generate_graph():
    graph = {}
    for s in xrange(0, num_buildings):
        distances = {}
        for t in xrange(0, num_buildings):
            near = is_near(s,t) or is_near(t,s)
            # Only generate paths between near nodes
            if s != t and near:
                distances[str(t)] = get_euclidean_distance(s,t)
        graph[str(s)] = distances
    # print dist_table
    return graph

def generate_paths(graph):
    global paths, S_LIST, G_LIST, buildings, path_parens, path_no_parens

    starting_points = []
    starting_indices = []
    terminal_points = []
    # path_descriptions = []
    path_ends = []

    # Find description for starting point
    for xy in S_LIST:
        start = find_closest(xy)
        starting_points.append(start)
        idx = create_building(xy[0],xy[1])
        text = first_step(idx,start,True)
        path_parens.append([text])
        text = first_step(idx,start,False)
        path_no_parens.append([text])
        buildings.pop()

    for xy in G_LIST:
        end = find_closest(xy)
        terminal_points.append(end)
        idx = create_building(xy[0],xy[1])
        description = terminal_guidance(idx,start)
        path_ends.append(description)
        buildings.pop()

    # print "Starting points", starting_points
    # print "Terminal points", terminal_points
    # print "Graph", graph

    for i in xrange(len(starting_points)):
        start = starting_points[i]
        end = terminal_points[i]
        # Convert ints because graph keys are strings
        dijkstra(graph, str(start), str(end),[],{},{})
        # print "\nGraph", graph

    # Example: [[22, 19, 12, 8, 4, 0]] len: 6
    for i in xrange(len(paths)):
        path = paths[i]
        for j in xrange(len(path)-1):
            s = path[j]
            t = path[j+1]
            text = step_guidance(s,t,True) # True)
            path_parens[i].append(text)
            text = step_guidance(s,t,False)
            path_no_parens[i].append(text)
        path_parens[i].append(path_ends[i])
        path_no_parens[i].append(path_ends[i])

    print 'Paths', paths
    # print 'Paths (parens):', path_parens
    # print 'Paths (no parens):', path_no_parens
    # print 'Path endings:', path_ends

    # dijkstra(graph,'0','22')

def get_euclidean_distance(source,target):
    """Find the euclidean distance between two points
    Based on an old Java program of mine:
    private double getEuclideanDistance(Vertex v1, Vertex v2) {
        double base = Math.abs(v1.x - v2.x); // x1 - x2
        double height = Math.abs(v1.y - v2.y); // y1 - y2
        double hypotenuse = Math
                .sqrt((Math.pow(base, 2) + (Math.pow(height, 2))));
        return hypotenuse;
    """

    # Take min(w,h) of source building into account
    # margin = (min(s['xywh'][2],s['xywh'][3])/2)

    if (type(source) == int):
        # Get building from indices
        s = buildings[source]
        x1 = s['centroid'][0]
        y1 = s['centroid'][1]
    else:
        x1 = source[0]
        y1 = source[1]

    if (type(target) == int):
        t = buildings[target]
        x2 = t['centroid'][0]
        y2 = t['centroid'][1]
    else:
        x2 = target[0]
        y2 = target[1]

    base = abs(x1-x2)
    height = abs(y1-y2)
    hypotenuse = math.sqrt(math.pow(base,2)+(math.pow(height,2)))
    return hypotenuse

def find_closest(xy):
    x = xy[0]
    y = xy[1]
    building_idx = int(map_labeled[y][x])-1
    if building_idx is not -1:
        return building_idx
    else:
        distances = np.zeros(num_buildings)
        for i in xrange(num_buildings):
            distances[i] = get_euclidean_distance(xy,i)
        return distances.argmin()

def is_inside(idx):
    building = buildings[idx]
    x = building['centroid'][0]
    y = building['centroid'][1]
    pixel = int(map_labeled[y][x])-1
    if pixel is -1:
        return False
    else:
        return True

def dijkstra(graph,src,dest,visited=[],distances={},predecessors={}):
    """Calculates a shortest path tree routed in src. Based on this tutorial:
    http://geekly-yours.blogspot.com/2014/03/dijkstra-algorithm-python-example-source-code-shortest-path.html
    I could have converted my Dijkstra program in Java into Python, but sorted_indices
    path-finding is not the emphasis of this assignment, I decided to spend more time
    on the visual analysis component"""
    global paths
    # a few sanity checks
    if src not in graph:
        raise TypeError(src, ': the root of the shortest path tree cannot be found in the graph')
    if dest not in graph:
        raise TypeError(dest, ': the target of the shortest path cannot be found in the graph')

    # ending condition
    if src == dest:
        # We build the shortest path and display it
        path=[]
        pred=dest
        while pred != None:
            path.append(int(pred))
            pred=predecessors.get(pred,None)
        if path:
            # print('Shortest Path: '+str(path)+" (Cost: "+str(distances[dest])+')')
            correct_order = []
            for item in reversed(path):
                correct_order.append(item)
            paths.append(correct_order)
            # print paths
    else:
        # if it is the initial  run, initializes the cost
        if not visited:
            distances[src]=0
        # visit the neighbors
        for neighbor in graph[src] :
            if neighbor not in visited:
                new_distance = distances[src] + graph[src][neighbor]
                if new_distance < distances.get(neighbor,float('inf')):
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = src
        # mark as visited
        visited.append(src)
        # now that all neighbors have been visited: recurse
        # select the non visited node with lowest distance 'x'
        # run Dijskstra with src='x'
        unvisited={}
        for k in graph:
            if k not in visited:
                unvisited[k] = distances.get(k,float('inf'))
        x=min(unvisited, key=unvisited.get)
        dijkstra(graph,x,dest,visited,distances,predecessors)

def first_step(start,target,parens,name=False):
    """'Go to the building that is east and near (which is cross-shaped).
    """

    inside = is_inside(start)

    if inside:
        text = 'You are inside a building'
        if parens:
            text += ' (%s)' %what_description(target)
        if name:
            text += ' <%s>' %buildings[target]['name']
        text += ' to the '
    else:
        text = 'You are outside. Go to the nearby building that is '

    s = buildings[start]
    t = buildings[target]

    if is_north(s,t): # north of s is t
        if inside:
            text += 'SOUTH'
        else:
            text += 'NORTH'
    elif is_south(s,t):
        if inside:
            text += 'NORTH'
        else:
            text += 'SOUTH'
    if is_east(s,t):
        if inside:
            text += 'WEST'
        else:
            text += 'EAST'
    elif is_west(s,t):
        if inside:
            text += 'EAST'
        else:
            text += 'WEST'
    if not inside:
        if parens:
            text += ' (%s)' %what_description(target)
        if name:
            text += ' <%s>' %buildings[target]['name']

    text += '.'
    # print 'EAST:', is_east(s,t), e_table[s][t]
    # print 'WEST:', is_west(s,t), e_table[t][s], w_table[s][t]
    # print text
    return text

def step_guidance(s,t,parens,name=False):
    """'Go to the building that is east and near (which is cross-shaped).
    Then go to the building that is north (which is oriented east-to-west).
    Then go to the building that is north and east (which is medium-sized and oriented north-to-south)
    """
    text = 'Now go to the nearby building that is '

    count = 0
    if n_table[s][t]: # north of s is t
        text += 'NORTH'
        count += 1
    elif n_table[t][s] or s_table[t][s]:
        text += 'SOUTH'
        count += 1
    if e_table[s][t]:
        if count == 0:
            text += 'EAST'
            count += 1
        else:
            text == ' and EAST'
    elif e_table[t][s] or w_table[s][t]:
        if count == 0:
            text += 'WEST'
            count += 1
        else:
            text += ' and WEST'
    # if count == 0:
    #     if is_north(s,t):
    #         text += 'north'
    if count != 0:
        if parens:
            text += ' (%s)' %what_description(t)
        if name:
            text += ' <%s>' %buildings[t]['name']

    text += '.'
    # print 'EAST:', is_east(s,t), e_table[s][t]
    # print 'WEST:', is_west(s,t), e_table[t][s], w_table[s][t]
    # print text
    return text

def terminal_guidance(start,target):
    if is_inside(target):
        text = 'Your final destination is inside this building. Go '
    else:
        text = 'Your final destination is outside near this building. Go '

    s = buildings[start]
    t = buildings[target]

    count = 0
    if is_north(s,t): # north of s is t
        text += 'NORTH'
        count += 1
    elif is_south(s,t):
        text += 'SOUTH'
        count += 1
    if is_east(s,t):
        if count == 0:
            text += 'EAST'
        else:
            text == ' and EAST'
    elif is_west(s,t):
        if count == 0:
            text += 'WEST'
        else:
            text += ' and WEST'
    if is_inside(target):
        text += ' within the building'
    text += '.'
    # print 'EAST:', is_east(s,t), e_table[s][t]
    # print 'WEST:', is_west(s,t), e_table[t][s], w_table[s][t]
    # print text
    return text

def print_instructions(parens_first=True):
    global path_parens, path_no_parens

    if parens_first:
        firsthalf = path_parens
        secondhalf = path_no_parens
    else:
        firsthalf = path_no_parens
        secondhalf = path_parens

    print '\nITINERARY ' + str(itinerary_num+1)
    print '------'

    if itinerary_num < 4:
        itinerary = firsthalf[itinerary_num]
        for step in itinerary:
            print step
            print '------'
    else:
        itinerary = secondhalf[itinerary_num]
        for step in itinerary:
            print step
            print '------'

def print_all_instructions(parens_first=False):
    global path_parens, path_no_parens

    if parens_first:
        firsthalf = path_parens
        secondhalf = path_no_parens
    else:
        firsthalf = path_no_parens
        secondhalf = path_parens

    for i in xrange(4):
        print '\nITINERARY ' + str(i+1)
        print '------'
        itinerary = firsthalf[i]
        for step in itinerary:
            print step
            print '------'

    for i in xrange(4):
        print '\nITINERARY ' + str(i+5)
        print '------'
        itinerary = secondhalf[i+4]
        for step in itinerary:
            print step
            print '------'

# ============================================================
# Main Invocation
# ============================================================

def main():

    global buildings, mode

    # Step 1. Generate 'what' for each building by analyzing image
    # Note: images and buildings information are stored as global vars
    building_names = load_names('ass3-table.txt')
    analyze_what(building_names)
    print_info()

    # Step 2. Generate 'where' lookup table for building relations
    analyze_where(buildings)

    # Step 4. Generate path for user
    graph = generate_graph()
    generate_paths(graph)
    # print_all_instructions(parens_first=False)
    # for path in path_parens:
    #     print len(path)

    # Step 3. Source and Target Description and User Interface set up in click event
    cv2.namedWindow('Columbia Campus Map')
    cv2.setMouseCallback('Columbia Campus Map', click_event)
    print "\nShowing image...\n"

    # Step 4. Show first itinerary
    print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
    start = S_LIST[itinerary_num]
    cv2.circle(map_campus,start,6,(0,255,0),-1)
    print_instructions()

    while(1):
        cv2.imshow('Columbia Campus Map', map_campus)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
            if mode:
                modeval = 'Path Generation'
            else:
                modeval = 'Cloud Generation'
            print 'Changing mode to', modeval,'(you pressed m)...\n'
        if k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__": main()
