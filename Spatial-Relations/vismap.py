import cv2
import numpy as np
from matplotlib.path import Path
import sys

# ============================================================
# Globals
# ============================================================

# set such that full image array is printed out
np.set_printoptions(threshold=np.nan)
# reset python's low default recursion limit (1000)
sys.setrecursionlimit(150000)

# green, red, blue, teal, yellow, orange, magenta
colors = [(0,255,0),(0,0,255),(255,0,0), (255,255,0), (0,255,255),(0,128,255),(255,0,255)]
color = colors[0]

cloud = {}
called = {}
recursive_calls = 0
# Blocks of pixels to check in each direction
pix = 6

drawing = False # true if mouse is pressed
# mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
click_count = -1

map_labeled = cv2.imread('ass3-labeled.pgm', 0) # load map_labeled as grayscale
map_campus = cv2.imread('ass3-campus.pgm', 1) # load map_campus as color
map_binary = cv2.cvtColor(map_campus,cv2.COLOR_BGR2GRAY) # load map_campus as grayscale
MAP_H = len(map_binary)
MAP_W = len(map_binary[0])

buildings = []
num_buildings = 0
monument = {}

# ============================================================
# User Interface
# ============================================================

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,click_count,color

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        idx = create_building(ix,iy)
        print 'Mouse clicked: ({},{})'.format(ix,iy)
        # print buildings[idx-1]['name']

        # alternate colors based on clicks
        if click_count >= len(colors)-1: # reset
            click_count = 0
        else:
            click_count += 1
        color = colors[click_count]

        # get x,y coordinates of all similar pixels
        pixels = pixel_cloud(x,y)

    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if drawing == True:
    #         if mode == True:
    #             # cv2.rectangle(map_campus,(ix,iy),(x,y),(0,255,0),-1)
    #             cv2.circle(map_campus,(x,y),5,(0,255,0),-1)
    #         else:
    #             cv2.circle(map_campus,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(map_campus,(ix,iy),pix/2,color,-1)
        # white dot indicates original click location
        cv2.circle(map_campus,(ix,iy),1,(255,255,255),-1)

        # if mode == True:
        #     # cv2.rectangle(map_campus,(ix,iy),(x,y),(0,255,0),-1)
        # else:
        #     cv2.circle(map_campus,(x,y),pix/2,(255,255,255),-1)

# ============================================================
# Source and Target Description and User Interface
# ============================================================

def create_building(x,y):
    global num_buildings, buildings
    idx = int(map_labeled[y][x])
    # add new x,y as a new building
    building = {}
    building['number'] = num_buildings
    building['name'] = 'Building ' + str(num_buildings)
    building['centroid'] = (x,y)
    building['xywh'] = (x,y,1,1)
    buildings.append(building)
    # num_buildings = len(buildings)
    return idx

def pixel_cloud(x,y):
    global color, cloud
    # Reset cloud every time this function is called
    cloud = {}
    relationships = []
    # for num in xrange(0, num_buildings-1-click_count):
    for num in xrange(0, num_buildings-1):
        s = buildings[num]
        t = buildings[-1] # the newly added building
        # Note these methods require xywh, centroid, number
        idx = int(map_labeled[y][x])
        near = is_near(s,t) or is_near(t,s)
        relationships.append([is_north(s,t), is_south(s,t), is_east(s,t), is_west(s,t),near,num,idx])
        # relationships.append([is_north(s,t), is_east(s,t), is_near(s,t),idx])

    # print "Relationships", relationships

    flood_fill(x,y,relationships)

    cloud_size = len(cloud) * pix
    print "Size of cloud:", cloud_size, "(recursive calls: %d)" %recursive_calls

    # Color in the cloud
    for xy in cloud:
        col = xy[0]
        row = xy[1]
        # map_campus[row][col] = [0,255,0]
        # Draw filled circle with radius of 5
        cv2.circle(map_campus,(col,row),pix/2,color,-1)

   # transitive reduction for north and east
    # for i in range(0, num_buildings):
    #     ibuilding = building[i]
    #     if relationships[i][0]: #north of this building
    #         for k in range(0, num_buildings):
    #             kbuilding = building[k]
    #             if is_north(kbuilding,ibuilding):
    #                 relationships[k][0] = False
    #     if relationships[i][1]: #east of this building
    #         for k in range(0, num_buildings):
    #             kbuilding = building[k]
    #             if is_east(kbuilding,ibuilding):
    #                 relationships[k][1] = False
    #     if relationships[i][2]: #near building reduce
    #         for k in range(0, num_buildings):
    #             kbuilding = building[k]
    #             if k != i and is_near(kbuilding,ibuilding) and is_near(ibuilding,kbuilding) and relationships[k][2]:
    #                 if ibuilding['area'] > kbuilding['area']:
    #                     relationships[i][2] = False
    #                 else:
    #                     relationships[k][2] = False
    # for i in range(0, num_buildings):
    #     if relationships[i][2]:         # if it's not nearby it's not helpful
    #         for k in range(0, num_buildings):
    #             if is_near(i,k) == False:
    #                 relationships[k] = False

    # for i in xrange(0, num_buildings-1):
    #     print 'i', i
    #     ibuilding = buildings[i]
    #     if relationships[i][0]: #north of this building
    #         for k in range(0, num_buildings-1):
    #             if is_north(k,i):
    #                 relationships[k][0] = False
    #     if relationships[i][1]: #east of this building
    #         for k in range(0, num_buildings-1):
    #             if is_east(k,i):
    #                 relationships[k][1] = False
    #     if relationships[i][2]: #near building reduce
    #         for k in range(0, num_buildings-1):
    #             kbuilding = buildings[k]
    #             if k != i and is_near(k,i) and is_near(i,k) and relationships[k][2]:
    #                 if ibuilding['area'] > kbuilding['area']:
    #                     relationships[i][2] = False
    #                 else:
    #                     relationships[k][2] = False
    for i in range(0, num_buildings-1):
        # Only keep near relationships
        if relationships[i][4] == False:
            # for k in range(0, num_buildings):
            #     if is_near(i,k) == False and is_near(k,i) == False:
            relationships[i][:4] = [False,False,False,False,False]


    # print 'New relationships:', relationships

    # global names
    # # describe the point

    coordinates = '(%d,%d)' %(x,y)
    if click_count%2 == 1:
        print 'Target: ' #+ coordinates
        # description = 'Then go to the building that is '
    else:
        print 'Source: ' #+ coordinates
        # description = 'Go to the nearby building that is '

    if (relationships[0][-1] == 0):
        description = coordinates + ' is '
    else:
        description = coordinates + ' is inside a building, '

    for idx in range(0, num_buildings-1):
        count = 0
        if relationships[idx][0]:
            description += 'north of '
            count += 1
        if relationships[idx][1]:
            description += 'south of '
            count += 1
        if relationships[idx][2]:
            if count == 0:
                description += 'east of '
                count += 1
            else:
                description += 'and east of '
        if relationships[idx][3]:
            if count == 0:
                description += 'west of '
                count += 1
            else:
                description += 'and west of '
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
    description = description[:-2] + '.'
    print description

def what_description(idx):
    what = 'the '
    descr = buildings[idx]['description']
    for i in xrange(len(descr)):
        if i < len(descr)-1:
            what += descr[i] + ', '
        else:
            what += descr[i] + ' structure'
    return what


def flood_fill(x,y,rel_table):
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
    for num in xrange(0, num_buildings-1):
        s = buildings[num]
        t = buildings[-1]
        t['centroid'] = (x,y) # change centroid to new x,y
        idx = int(map_labeled[y][x])
        near = is_near(s,t) or is_near(t,s)
        # Note these methods require xywh, centroid, number
        # rel.append([is_north(s,t),is_east(s,t),is_near(s,t),idx])
        rel.append([is_north(s,t), is_south(s,t), is_east(s,t), is_west(s,t),near,num,idx])

    # print rel

    # Base case. If the current x,y is not the right rel do nothing
    if rel != rel_table:
        return

    # Add pixel to list of clouds to be recolored and used later
    cloud[(x,y)] = ''

    # Recursive calls. Make a recursive call as long as we are not
    # on boundary

    if x > (pix-1): # left # originally 0
        flood_fill(x-pix, y, rel_table)

    if y > (pix-1): # up # originally 0
        flood_fill(x, y-pix, rel_table)

    if x < MAP_W-(pix+1): # right # originally MAP_W-1
        flood_fill(x+pix, y, rel_table)

    if y < MAP_H-(pix+1): # down # originall MAP_H-`
        flood_fill(x, y+pix, rel_table)

def index_valid(x,y):
    x = xy[0]
    y = xy[1]
    if (x > 0) and (x < MAP_W) and (y > 0) and (y < MAP_H):
        return True
    else:
        return False



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
    cv2.circle(map_campus, centroid, 3, (255,255,0), -1)
    # To draw a circle, you need its center coordinates and radius
    cv2.circle(map_campus, centroid, 3, (255,255,0), -1)
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

def analyze_buildings(names):
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

    max_area, min_area = analyze_areas(buildings,print_results=True) # add True arg to print results

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

def find_ambiguity():
    global buildings
    for idx in xrange(num_buildings):
        bldg1 = buildings[idx]
        for jdx in xrange(num_buildings):
            bldg2 = buildings[jdx]
            if idx != jdx and bldg1['description'] == bldg2['description']:
                if is_north(bldg1,bldg2):
                    bldg2['description'].insert(0,'northern')
                    bldg1['description'].insert(0,'southern')
                elif is_south(bldg1,bldg2):
                    bldg1['description'].insert(0,'northern')
                    bldg2['description'].insert(0,'southern')
                buildings[idx] = bldg1
                buildings[jdx] = bldg2
                # print 'Ambiguity between', bldg1['name'], 'and', bldg2['name']

def find_extrema():
    """Find singularly defining characteristics and remove other details"""
    global buildings
    for idx in xrange(num_buildings):
        bldg1 = buildings[idx]
        description = bldg1['description']
        for characteristic in description:
            count = 0
            if characteristic == 'almost rectangular' or 'southernmost':
                break
            for jdx in xrange(num_buildings):
                bldg2 = buildings[jdx]
                if (idx != jdx) and (characteristic in tuple(bldg2['description'])):
                    count += 1
                    # print 'Found ', characteristic, 'from', bldg1['name'], 'in', bldg2['name']
                    break
            if count is 0:
                'Found extrema!', characteristic
                extrema = [characteristic]
                bldg1['description'] = extrema
                buildings[idx] = bldg1
                break



def find_monument():
    global monument, buildings
    for idx in xrange(num_buildings):
        if buildings[idx]['xywh'][2] > MAP_W - 10:
            monument = buildings[idx]
            buildings[idx]['description'] = ['longest']
            return

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
        return 'middling-sized'
    elif ratio > 0.1: # cutoff Earl Hall
        return 'small'
    else:
        return 'tiny'

def unpack(tup):
    if len(tup) is 4:
        return tup[0],tup[1],tup[2],tup[3]
    elif len(tup) is 5:
        return tup[0],tup[1],tup[2],tup[3],tup[4]

def describe_shape(building):
    """Describe shape based on corner and midpoint counts"""

    descriptions = []

    x,y,w,h = unpack(building['xywh'])

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
            descriptions.append('squarish cross-shaped')
        else:
            descriptions.append('cross-shaped')
    elif (corners_count == 4 and midpoints_count == 2):
        descriptions.append('I-shaped')
    elif (corners_count == 4 and midpoints_count == 3):
        descriptions.append('C-shaped')
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

# def find_extrema(buildings):
#     """Find extrema and return as list of tuple pairs of building index and extrema description"""
#     # num_buildings = len(buildings)
#     extrema = []

#     # Find largest and smallest by MBR area
#     # We use MBR area instea of area because it's more apparent to the human eye
#     # when a structure takes up more space in the bigger scheme of things
#     sorted_buildings = sorted(buildings, key=lambda k:-(k['area']/k['extent']))
#     areas = [(sorted_buildings[i]['number']-1) for i in range(num_buildings)]
#     extrema.append(('biggest', areas[0]))
#     extrema.append(('smallest', areas[-1]))
#     return extrema


def print_info(buildings):
    for building in buildings:
        print building['number'], ':', building['name']
        print '     Minimum Bounding Rectangle:', building['mbr'][0], ',', building['mbr'][1]
        print '     Center of Mass:', building['centroid']
        print '     Area:', building['area']
        print '     Description:', building['description']

# ============================================================
# The "Where"
# ============================================================

def analyze_relations(buildings):
    """Find all binary spatial relationships for every pair,
    and apply transitive reduction."""

    # num_buildings = len(buildings)

    # Lookup tables
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


    # Transitive reduction
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

def is_north(s,t):
    """Find out if 'North of S is T'"""
    # Form triangle to north border: (x,0)
    return triangulate_FOV(s,t,-1,0,1)

def is_south(s,t):
    """Find out if 'South of S is T'"""
    # Form triangle to south border: (x,MAP_H)
    return triangulate_FOV(s,t,-1,MAP_H,1)

def is_east(s,t):
    """Find out if 'East of S is T'"""
    # Form triangle to east border: (MAP_W,y)
    return triangulate_FOV(s,t,MAP_W,-1,1.2)

def is_west(s,t):
    """Find out if 'West of S is T'"""
    # Form triangle to west border: (0,y)
    return triangulate_FOV(s,t,0,-1,1.2)

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
    # TODO: change this after you do extrema
    if (t['number'] == 21):
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
    # TODO: change this after you do extrema
    if (t['number'] == 21):
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

def analyze_relations_single(source, direction, buildings):
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

def get_euclidean_distance(s,t):
    """ private double getEuclideanDistance(Vertex v1, Vertex v2) {
        double base = Math.abs(v1.x - v2.x); // x1 - x2
        double height = Math.abs(v1.y - v2.y); // y1 - y2
        double hypotenuse = Math
                .sqrt((Math.pow(base, 2) + (Math.pow(height, 2))));
        return hypotenuse;
    """
    x1 = s['centroid'][0]
    x2 = t['centroid'][0]
    y1 = s['centroid'][0]
    y2 = t['centroid'][0]

    base = abs(x1-x2)
    height = abs(y1-y2)
    hypotenuse = math.sqrt(math.pow(base,2)+(math.pow(height,2)))
    return hypotenuse

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

def get_near_points(building):
    if 'near_points' not in building:
        shift = 30 # Empirically chosen
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

    s_points = get_near_points(s)
    t_points = get_near_points(t)

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

# def transitive_reduce():
#     """Output should use building names rather than numbers"""

# ============================================================
# Main Invocation
# ============================================================

def main():

    # Analyze image
    names = load_names('ass3-table.txt')
    # buildings =
    analyze_buildings(names)
    print_info(buildings)

    # Generate lookup table for building relations
    relations = analyze_relations(buildings)
    # analyze_relations_single(20, 'west', buildings)

    cv2.namedWindow('Columbia Campus Map')
    cv2.setMouseCallback('Columbia Campus Map', draw_circle)
    print "Showing image..."

    # print buildings

    # cv2.waitKey(0)

    while(1):
        cv2.imshow('Columbia Campus Map', map_campus)
        k = cv2.waitKey(1) & 0xFF
        # if k == ord('m'):
        #     print 'pressed m'
        #     mode = not mode
        # el
        if k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__": main()
