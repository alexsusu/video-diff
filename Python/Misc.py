#import cv
#import cv2


def ConvertNPToCVMat(imgNP):
    """
    # This gives a: AttributeError: 'numpy.ndarray' object has no attribute 'from_array'
    ImageAlignment.template_image = ImageAlignment.template_image.from_array()
    """
    # Inspired from https://stackoverflow.com/questions/5575108/how-to-convert-a-numpy-array-view-to-opencv-matrix :
    #assert foo_numpy.dtype == 'uint8'
    #assert foo_numpy.ndim == 3
    hNP, wNP = imgNP.shape[:2]
    tmpCV = cv.CreateMat(hNP, wNP, cv.CV_8UC3)
    """
    cv.CreateMat(src_image.height, \
                    src_image.width, \
                    cv.CV_32F)
    """
    cv.SetData(tmpCV, imgNP.data, imgNP.strides[0])
    return tmpCV


def ConvertNPToIplImage(imgNP):
    # Inspired from https://stackoverflow.com/questions/11528009/opencv-converting-from-numpy-to-iplimage-in-python
    # imgNP is numpy array
    NUM_COLORS = 1 #3

    bitmap = cv.CreateImageHeader((imgNP.shape[1], imgNP.shape[0]), \
                                    cv.IPL_DEPTH_8U, NUM_COLORS)
    cv.SetData(bitmap, imgNP.tostring(), 
               imgNP.dtype.itemsize * NUM_COLORS * imgNP.shape[1])
    return bitmap
