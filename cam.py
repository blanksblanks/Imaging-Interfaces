import cv2, time, sys
import numpy as np

# Camera 0 is the integrated web cam
camera_port = 0

# Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30
timer = 15

# Initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)

# Captures a single image from the camera and returns it in PIL format
def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
     retval, im = camera.read()
     return im


def countdown(t):
    while t > 0:
    # Ramp the camera - these frames will be discarded and are only used to
    # adjust light levels, if necessary
        for i in xrange(ramp_frames/timer):
            temp = get_image()
        # Print countdown
        mins, secs = divmod(t, 60)
        timeformat ='\r                ... {:02d}:{:02d} seconds\n'.format(mins, secs)
        sys.stdout.write(timeformat)
        sys.stdout.flush()
        time.sleep(1)
        t -= 1

print('Taking image in ... 00:{:02d} seconds'.format(timer))
countdown(timer - 1)
print('Taking image!\n')

# Take the actual image we want to keep
camera_capture = get_image()
file = "./test_image.png"
# A nice feature of the imwrite method is that it will automatically choose the
# correct format based on the file extension you provide. Convenient!
cv2.imwrite(file, camera_capture)

# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
del(camera)




