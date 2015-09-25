#MY_DEBUG_STDOUT = True
MY_DEBUG_STDOUT = False

MY_DEBUG_STDERR = True


import sys
import time
import traceback

import cv2


"""
Note: to do lazy evaluation of the print arguments we can do:
    DebugPrint("%s", lambda: str(myNpArray))
  in the idea that we make
    MY_DEBUG_STDOUT = False
  and we don't want to incur time penalty of evaluating the formal arguments
    of DebugPrint(), which can be huge when doing something like str(np.array),
    etc.
 (inspired from http://swizec.com/blog/python-and-lazy-evaluation/swizec/5148).
"""

def DebugPrint(aText):
    if MY_DEBUG_STDOUT:
        print aText
        sys.stdout.flush()
    """
    try:
        if MY_DEBUG_STDOUT:
            print aText
            sys.stdout.flush()
    except:
        pass
    """


def DebugPrintErrorTrace():
    try:
        if MY_DEBUG_STDERR:
            traceback.print_exc()
            sys.stderr.flush()
    except:
        pass


def GetCurrentDateTimeStringWithMilliseconds():
    crtTime = time.localtime()

    # We use crtTime2 only to compute numMilliseconds.
    crtTime2 = time.time()

    # See http://discussion.forum.nokia.com/forum/showthread.php?116978-What-is-the-time-granularity-in-Pys60 .
    numMilliseconds = (crtTime2 - int(crtTime2)) * 1000

    #fileName = time.strftime("%Y_%m_%d_%H_%M_%S", crtTime) + \
    #    ("_%03d%s" % (numMilliseconds, fileExtension))

    return time.strftime("%Y_%m_%d_%H_%M_%S", crtTime) + \
            "_%03d" % numMilliseconds


# Inspired from Z:\Works\TopCoder\MM\MM_RobonautEye\edge.py
def ConvertImgToGrayscale(img):
    if False:
        print("Alex: img = %s" % str(img));
        print("Alex: len(img) = %d" % len(img));
        print("Alex: len(img[0]) = %d" % len(img[0]));
        print("Alex: img.flags = %s" % str(img.flags));
        print("Alex: img.astype = %s" % str(img.astype));
        print("Alex: img.dtype = %s" % str(img.dtype));
        #print("Alex: dir(img) = %s" % str(dir(img)));

    # convert image to grayscale only if it is NOT already gray
    assert img.ndim == 3;

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

    if False:
        print("Alex: gray = %s" % str(gray));
        print("Alex: len(gray) = %d" % len(gray));
        print("Alex: len(gray[0]) = %d" % len(gray[0]));

    return gray;
