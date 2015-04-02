import cv2
import numpy as np

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)

im = cv2.imread('ass3-labeled.pgm', cv2.IMREAD_UNCHANGED)
img = cv2.imread('ass3-campus.pgm', cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
im = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
# img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

buildings = {}
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
contours,hierarchy = cv2.findContours(imgray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
idx = 27
for cnt in contours:
    print idx
    # To get all the points which comprise an object
    # Numpy function gives coordinates in (row, col)
    # OpenCV gives coordinates in (x,y)
    # Note row = x and col = y
    mask = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    #pixelpoints = cv2.findNonZero(mask)
    color = cv2.mean(im,mask = mask)
    if (color[0] < 0.9):
        print 'ignored the cs courtyard'
        continue
    print 'building', color
    idx -= 1
    # Let (x,y) be top-left coordinate and (w,h) be width and height
    # Find min, max value of x, min, max value of y
    x,y,w,h = cv2.boundingRect(cnt)
    print "x,y,w,h", x,y,w,h
    roi=img[y:y+h,x:x+w]
    # cv2.imwrite(str(idx) + '.jpg', roi)
    # To draw a rectangle, you need T-L corner and B-R corner
    cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
    # Image moments help you to calculate center of mass, area of object, etc.
    # cv2.moments() gives dictionary of all moment values calculated
    # area = cv2.contourArea(cnt)
    # x,y,w,h = cv2.boundingRect(cnt)
    # rect_area = w*h
    # extent = float(area)/rect_area
    M = cv2.moments(cnt)
    # Centroid is given by the relations
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = (cx, cy)
    print 'center of mass', centroid
    # Contour area is given by the function cv2.contourArea(cnt) or
    area = M['m00']
    print 'area', area
    # To draw a circle, you need its center coordinates and radius
    cv2.circle(im, centroid, 3, (255,255,0), -1)

print "showing image"
cv2.imshow('Columbia Campus Map',im)
cv2.waitKey(0)


while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        print 'pressed m'
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()