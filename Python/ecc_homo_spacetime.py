"""
NOTE: running ecc_homo_spacetime() on the same inputs results in the same output
    images (so, if saving .PNG files with the same parameters,
    they should be the same).
The saved output frames:
    R, B (--> purple) channels are of the input (query) frame
    G channel is of the reference frame

(From TPAMI 2013 paper:
  "To highlight changes with green and pink colors, we use a
  modified RGB representation [1], by replacing the G
  channel of the input frame with the G component of the
  reference frame, but warped in space and time based on
  the ECC outcome"
)
So the purple part of the image is the input/query frame.

Note: from config.py we have:
    pixel_select = 0;
    time_flag = 0;
    weighted_flag = 1;
"""

import cv2

import Clustering
import common
import config
import Matlab
import ReadVideo

import os
import math
import numpy as np
from numpy import linalg as npla
import scipy.weave
import sys

import rcdtype


#INVERSE = True # inverse-compositional version of ECC image alignment algo
INVERSE = False # Forwards Additive Algorithm

if False:
    # See http://docs.continuum.io/mkl-service/index.html (also http://docs.continuum.io/mkl-optimizations/index.html, http://continuum.io/blog/mkl-optimizations)
    import mkl
    print("mkl.get_num_threads() = %s" % str(mkl.get_num_threads()));
    mkl.set_num_threads(2);

#MATRIX_FLOAT_TYPE = np.float64;
MATRIX_FLOAT_TYPE = np.float32;


"""
FOOBAR is where I tried to use WITHOUT SUCCESS np.c_ and np.r_ instead of
    np.hstack and np.vstack, respectively.
"""
FOOBAR = False
# Matlab traverses in find(A) the matrix A in column-major (Fortran) order
ORDER_MATTERS = True

#USE_DRIVER = True
USE_DRIVER = False



FitElem = rcdtype.recordtype("FitElem", "weights warp_p rms_error rho factor t")




#if config.TESTING_IDENTICAL_MATLAB:
captureR = None;
captureQ = None;

"""
We read frames from the video, be it input or reference (specified by capture).
"""
def MyImageRead(capture, index):
    global captureR, captureQ;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("Entered MyImageRead(capture=%s, index=%s)" % \
                                            (str(capture), str(index)));
    if config.USE_MULTITHREADING == True:
        """
        We must reopen the capture device in each different process, otherwise the
            program blocks at the first operation in the "global" capture device.
        """

        """
        !!!!TODO: study well: it appears that opening a VideoCapture in a
         process/thread and using it another thread blocks the program relying on
         OpenCV, in MyImageRead(), either at setting the position in the
         VideoCapture OR when reading the data.
         This might be a limitation of the ffmpeg library when working multithreaded.
        """
        if capture == captureQ:
            #capture = cv2.VideoCapture("/home/asusu/drone-diff_Videos/GoPro_clips/2/GOPR7269_clip_secs69-73.mp4");
            #capture = cv2.VideoCapture("Videos/input.avi");
            capture = cv2.VideoCapture(sys.argv[1]);

            #!!!!TODO: find a good way to prevent reopening every time the VideoCapture device
            """
            CLEARLY NOT A GOOD SOLUTION to prevent reopening every time the VideoCapture device:
                captureQ = capture;
            """
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("MyImageRead(): new capture=%s" % \
                                                    (str(capture)));
        elif capture == captureR:
            #capture = cv2.VideoCapture("/home/asusu/drone-diff_Videos/GoPro_clips/2/GOPR7313_clip_secs60-64.mp4");
            #capture = cv2.VideoCapture("Videos/reference.avi");
            capture = cv2.VideoCapture(sys.argv[2]);

            #!!!!TODO: find a good way to prevent reopening every time the VideoCapture device
            """
            CLEARLY NOT A GOOD SOLUTION to prevent reopening every time the
               VideoCapture device:
            captureR = capture;
            """
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("MyImageRead(): new capture=%s" % \
                                                        (str(capture)));

    if config.TESTING_IDENTICAL_MATLAB:
        if index < 0:
            return np.zeros( (ReadVideo.resVideoQ[1] * config.VIDEO_FRAME_RESIZE_SCALING_FACTOR, \
                              ReadVideo.resVideoQ[0] * config.VIDEO_FRAME_RESIZE_SCALING_FACTOR), \
                                dtype=np.uint8); #!!!!TODO: should we return * 3, for RGB?

        if capture == captureR:
            videoPathFileName = "Videos/reference/00%4d.jpeg" % (2001 + index); #2001.jpeg";
        elif capture == captureQ:
            videoPathFileName = "Videos/input/00%4d.jpeg" % (1001 + index); #2001.jpeg";
        else:
            assert False;

        if config.OCV_OLD_PY_BINDINGS:
            img = cv2.imread(videoPathFileName, cv2.CV_LOAD_IMAGE_GRAYSCALE);
        else:
            img = cv2.imread(videoPathFileName, cv2.IMREAD_GRAYSCALE);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint( \
                    "ecc_homo_spacetime.MyImageRead(): img.shape = %s" % \
                                                        (str(img.shape)));
            common.DebugPrint( \
                    "ecc_homo_spacetime.MyImageRead(): img.dtype = %s" % \
                                                        (str(img.dtype)));
            common.DebugPrint( \
                    "ecc_homo_spacetime.MyImageRead(): img[:10, :10] = %s" % \
                                                        (str(img[:10, :10])));

        if config.VIDEO_FRAME_RESIZE_SCALING_FACTOR != 1:
            # We resize the image
            imgGray = Matlab.imresize(imgGray, \
                            scale=config.VIDEO_FRAME_RESIZE_SCALING_FACTOR); #/2.0); #*2.0

        return img;

    if index < 0:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("MyImageRead(): index < 0 --> returning black frame");
        #return np.zeros( ReadVideo.resVideoQ );
        return np.zeros( (ReadVideo.resVideoQ[1] * config.VIDEO_FRAME_RESIZE_SCALING_FACTOR, \
                          ReadVideo.resVideoQ[0] * config.VIDEO_FRAME_RESIZE_SCALING_FACTOR), \
                            dtype=np.uint8);  #!!!!TODO: should we return *3 for RGB

    index *= config.counterRStep;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("MyImageRead(): index = %s" % (str(index)));

    if config.OCV_OLD_PY_BINDINGS:
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, index);
    else:
        """
        From http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get:
            <<CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be
              decoded/captured next.>>
        """
        capture.set(cv2.CAP_PROP_POS_FRAMES, index);

    # This is only for (paranoid) testing purposes:
    if config.OCV_OLD_PY_BINDINGS:
        indexCrt = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES);
    else:
        """
        From http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get:
            <<CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be
              decoded/captured next.>>
        """
        indexCrt = capture.get(cv2.CAP_PROP_POS_FRAMES);

    #assert int(indexCrt) == index;
    #!!!!TODO: think if OK
    if int(indexCrt) != index:
        #if common.MY_DEBUG_STDOUT:
        print("MyImageRead(): indexCrt != index --> returning black frame");
        ret = False;
    else:
        """
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("Alex: frameR = %d" % frameR);
        """

        #if myIndex > numFramesR:
        #    break;

        #ret, img = r_path.read();
        ret, img = capture.read();
        #if ret == False:
        #    break;

    #!!!!TODO: think if well
    #assert ret == True;
    if ret == False:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint( \
                "MyImageRead(): ret == False --> returning black frame");
        img = np.zeros( (ReadVideo.resVideoQ[1] * config.VIDEO_FRAME_RESIZE_SCALING_FACTOR, \
                         ReadVideo.resVideoQ[0] * config.VIDEO_FRAME_RESIZE_SCALING_FACTOR, \
                         3), \
                            dtype=np.uint8);
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("MyImageRead(): img.shape = %s" % str(img.shape));
        common.DebugPrint("MyImageRead(): img.dtype = %s" % str(img.dtype));

    # In the Matlab code he reads gray/8bpp JPEGs
    imgGray = common.ConvertImgToGrayscale(img);
    #assert ret;

    #if True:
    #if False:
    if config.VIDEO_FRAME_RESIZE_SCALING_FACTOR != 1:
        # We resize the image
        imgGray = Matlab.imresize(imgGray, \
                        scale=config.VIDEO_FRAME_RESIZE_SCALING_FACTOR); #/2.0); #*2.0

    # If np.set_options...() specify very verbose, it prints all bytes of img :)), so better not do it
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("  MyImageRead(%s, %d): img = %s" % \
                                    (str(capture), index, str(img)));

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("Exiting MyImageRead()");

    """
    We can apply the Canny operator on the input image(s) in attempt to
    reduce the time to execute the ECC algorithm.
    """
    #if True:
    if False:
        #for i in range(xxx.shape[2]):
            # Inspired from http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
            #xxx[:, :, i] = cv2.Canny(xxx[:, :, i], 100, 200);
        # Inspired from http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
        # See http://docs.opencv.org/modules/imgproc/doc/feature_detection.html#canny
        imgGray = cv2.Canny(imgGray, 170, 200);

    return imgGray;


def next_level(warp_in, transform, high_flag):
    """
    function warp=next_level(warp_in, transform, high_flag)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %WARP=NEXT_LEVEL(WARP_IN, TRANSFORM, HIGH_FLAG)
    This function modifies appropriately the WARP values in order to apply
    % the warp in the next level. If HIGH_FLAG is equal to 1, the function
    % makes the warp appropriate for the next level of higher resolution. If
    % HIGH_FLAG is equal to 0, the function makes the warp appropriate
    % for the previous level of lower resolution.
    %
    % Input variables:
    % WARP_IN:      the current warp transform,
    % TRANSFORM:    the type of adopted transform, accepted strings:
    %               'tranlation','affine' and 'homography'.
    % HIGH_FLAG:    The flag which defines the 'next' level. 1 means that the
    %               the next level is a higher resolution level,
    %               while 0 means that it is a lower resolution level.
    % Output:
    % WARP:         the next-level warp transform
    %--------------------------------------
    %
    % $ Ver: 1.3, 13/5/2012,  released by Georgios D. Evangelidis.
    % For any comment, please contact georgios.evangelidis@inria.fr
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    warp=warp_in;
    if high_flag==1
        if strcmp(transform,'homography')
            warp(7:8)=warp(7:8)*2;
            warp(3)=warp(3)/2;
            warp(6)=warp(6)/2;
        end

        if strcmp(transform,'affine')
            warp(7:8)=warp(7:8)*2;

        end

        if strcmp(transform,'translation')
            warp = warp*2;
        end

        if strcmp(transform,'euclidean')
            warp(1:2,3) = warp(1:2,3)*2;
        end

    end

    if high_flag==0
        if strcmp(transform,'homography')
            warp(7:8)=warp(7:8)/2;
            warp(3)=warp(3)*2;
            warp(6)=warp(6)*2;
        end

        if strcmp(transform,'affine')
            warp(7:8)=warp(7:8)/2;
        end

        if strcmp(transform,'euclidean')
            warp(1:2,3) = warp(1:2,3)/2;
        end

        if strcmp(transform,'translation')
            warp = warp/2;
        end

    end
    """

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("next_level(warp_in=%s, transform, high_flag)" % \
                                                            str(warp_in));

    #warp=warp_in;
    warp = warp_in;

    if high_flag == 1:
        #if strcmp(transform,'homography')
        if transform == "homography":
            #warp(7:8)=warp(7:8)*2;
            warp[0, 2] *= 2;
            warp[1, 2] *= 2;

            #warp(3)=warp(3)/2;
            warp[2, 0] /= 2;

            #warp(6)=warp(6)/2;
            warp[2, 1] /= 2;

        #if strcmp(transform,'affine')
        if transform == "affine":
            #warp(7:8)=warp(7:8)*2;
            warp[0, 2] *= 2;
            warp[1, 2] *= 2;

        #if strcmp(transform,'translation')
        if transform == "translation":
            #warp = warp*2;
            warp *= 2;

        #if strcmp(transform,'euclidean')
        if transform == "euclidean":
            #warp(1:2,3) = warp(1:2,3)*2;
            warp[0: 2, 2] *= 2;

    if high_flag == 0:
        #if strcmp(transform,'homography')
        if transform == "homography":
            #warp(7:8)=warp(7:8)/2;
            warp[0, 2] /= 2;
            warp[1, 2] /= 2;

            #warp(3)=warp(3)*2;
            warp[2, 0] *= 2;

            #warp(6)=warp(6)*2;
            warp[2, 1] *= 2;

        #if strcmp(transform,'affine')
        if transform == "affine":
            #warp(7:8)=warp(7:8)/2;
            warp[0, 2] /= 2;
            warp[1, 2] /= 2;

        #if strcmp(transform,'euclidean')
        if transform == "euclidean":
            #warp(1:2,3) = warp(1:2,3)/2;
            warp[0: 2, 2] /= 2;

        #if strcmp(transform,'translation')
        if transform == "translation":
            #warp = warp/2;
            warp /= 2;

    return warp;


#function H = hessian(VI_dW_dp, N_p, w)
def hessian(VI_dW_dp, N_p, w):
    #if nargin<3 error('Not enough input arguments'); end

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("Entered hessian(N_p=%s)." % str(N_p));
    print("hessian(VI_dW_dp.shape=%s, N_p=%d, w=%d)." % \
                        (str(VI_dW_dp.shape), N_p, w));

    #if size(VI_dW_dp,2)~=(N_p*w)
    if VI_dW_dp.shape[1] != N_p * w:
        #error('N_p times image-width in Hessian computation')
        print("VI_dW_dp.shape[1] != N_p * w does NOT hold");
        quit();

    H = np.zeros( (N_p, N_p) ); # N_p is usually small, so no need I guess to use np.float32 or smaller

    #for i=1:N_p
    """
    We substitute i - 1 with i, since in Python arrays indices start from 0,
         not 1 like in Matlab.
    """
    #for i in range(1, N_p + 1):
    for i in range(N_p):
        #h1 = VI_dW_dp(:,((i-1)*w)+1:((i-1)*w)+w);
        h1 = VI_dW_dp[:, (i * w) + 1 - 1: (i * w) + w];

        if i == 0:
            print("hessian(h1.shape=%s)." % \
                        (str(h1.shape)));

        #for j=1:N_p
        """
        We substitute j - 1 with j, since in Python arrays indices start from 0,
            not 1 like in Matlab.
        """
        #for j in range(1, N_p + 1):
        for j in range(N_p):
            #h2 = VI_dW_dp(:,((j-1)*w)+1:((j-1)*w)+w);
            h2 = VI_dW_dp[:, (j * w) + 1 - 1: (j * w) + w];

            if j == 0:
                print("hessian(h1.shape=%s)." % \
                        (str(h1.shape)));

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("hessian(): h1.shape = %s" % str(h1.shape));
                common.DebugPrint("hessian(): h2.shape = %s" % str(h2.shape));

            #H(j,i) = sum(sum((h1 .* h2)));
            H[j, i] = (h1 * h2).sum(); #.sum();

    print("hessian(): H (again) = %s" % str(H));

    return H;


def hessian_scipy(VI_dW_dp, N_p, w):
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("Entered hessian_scipy(N_p=%s)." % str(N_p));
    print("hessian_scipy(VI_dW_dp.shape=%s, N_p=%d, w=%d)." % \
                        (str(VI_dW_dp.shape), N_p, w));

    #if size(VI_dW_dp,2)~=(N_p*w)
    if VI_dW_dp.shape[1] != N_p * w:
        #error('N_p times image-width in Hessian computation')
        print("VI_dW_dp.shape[1] != N_p * w does NOT hold");
        quit();

    H = np.zeros( (N_p, N_p) ); # N_p is usually small, so no need I guess to use np.float32 or smaller

    if False:
        for i in range(N_p):
            for j in range(N_p):
                dpMat = 0;
                jiDiff = (j - i) * w;
                for r in range(VI_dW_dp.shape[0]):
                    for c in range((i * w), (i * w) + w):
                        dpMat += VI_dW_dp[r, c] * VI_dW_dp[r, c + jiDiff];

                H[j, i] = dpMat;

        #print("hessian(): H = %s" % str(H));
        #return H;

    """
    PyArray_Descr *PyArray_DESCR(PyArrayObject* arr)
        Returns a borrowed reference to the dtype property of the array.

    PyArray_Descr *PyArray_DTYPE(PyArrayObject* arr)
        New in version 1.7.
        A synonym for PyArray_DESCR, named to be consistent with the .dtype. usage within Python.
    """

    common.DebugPrint("hessian_scipy(): VI_dW_dp.strides = %s" % str(VI_dW_dp.strides));
    common.DebugPrint("hessian_scipy(): VI_dW_dp.shape = %s" % str(VI_dW_dp.shape));
    common.DebugPrint("hessian_scipy(): VI_dW_dp.dtype = %s" % str(VI_dW_dp.dtype));

    assert (VI_dW_dp.dtype == np.float32) or (VI_dW_dp.dtype == np.float64); #np.int64;

    assert VI_dW_dp.ndim == 2;

    if VI_dW_dp.dtype == np.float32:
        dtypeSize = 4; # np.float32 is 4 bytes
    elif VI_dW_dp.dtype == np.float64:
        dtypeSize = 8; # np.float64 is 8 bytes

    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html
    # We check if we have the matrices in row-major order-style
    assert VI_dW_dp.strides == (VI_dW_dp.shape[1] * dtypeSize, dtypeSize);

    """
    # We check if we have the matrices in column-major order (Fortran) style
    assert Xq.strides == (dtypeSize, Xq.shape[0] * dtypeSize);
    assert Yq.strides == (dtypeSize, Yq.shape[0] * dtypeSize);
    """

    CPP_code2_prefix_Row_Major_Order = """
    #define elemVI_dW_dp(row, col) (((double *)VI_dW_dp_array->data)[(row) * numC + (col)])
    """

    CPP_code2_prefix_Fortran_Major_Order = """
    #define elemXq(row, col) (((double *)Xq_array->data)[(col) * numR + (row)])
    #define elemYq(row, col) (((double *)Yq_array->data)[(col) * numR + (row)])
    """

    """
    if (Xq.strides == (dtypeSize, Xq.shape[0] * dtypeSize)):
        assert Yq.strides == (dtypeSize, Yq.shape[0] * dtypeSize);
        CPP_prefix = CPP_code2_prefix_Fortran_Major_Order;
    else:
        CPP_prefix = CPP_code2_prefix_Row_Major_Order;
    """

    # See http://wiki.scipy.org/Weave, about how to handle NP array in Weave

    CPP_code2 = """
    int i, j;
    int r, c;

    int numR, numC;
    numR = VI_dW_dp_array->dimensions[0];
    numC = VI_dW_dp_array->dimensions[1];

    #define elemVI_dW_dp(row, col) (((double *)VI_dW_dp_array->data)[(row) * numC + (col)])
    #define elemH(row, col) (((myDbl *)H_array->data)[(row) * N_p + (col)])

    /*
    for (i = 0; i < N_p; i++) {
        for (j = 0; j < N_p; j++) {
            myDbl dpMat = 0.0;
            int jiDiff = (j - i) * w;

            for (r = 0; r < numR; r++) {
                for (c = i * w; c < (i * w) + w; c++) {
                    dpMat += elemVI_dW_dp(r, c) * elemVI_dW_dp(r, c + jiDiff);
                }
            }

            elemH(j, i) = dpMat;
        }
    }
    */

    // As Evangelidis was observing (see also the OpenCV 3.0 ECC implementation), the Hessian is symmetric - so we can "half" the computational effort
    for (i = 0; i < N_p; i++) {
        for (j = i; j < N_p; j++) {
            /*
            For j == i, in the OpenCV 3.0 ECC implementation, they use pow(norm(mat), 2).
                I don't consider this to be faster than our element-wise multiplication.

              However, the implementation https://bitbucket.org/gevangelidis/eccblock/overview
                does better - they use:
                    cvDotProduct(&mat, &mat) (or cvDotProduct(&mat, &mat2))
                !!!!TODO: think more - do more tests, etc
            */
            myDbl dpMat = 0.0;
            int jiDiff = (j - i) * w;

            for (r = 0; r < numR; r++) {
                for (c = i * w; c < (i * w) + w; c++) {
                    dpMat += elemVI_dW_dp(r, c) * elemVI_dW_dp(r, c + jiDiff);
                }
            }

            elemH(j, i) = dpMat;
            if (j != i) {
                // The Hessian is symmetric
                elemH(i, j) = dpMat;
            }
        }
    }
    """

    #CPP_code = CPP_prefix + CPP_code2;
    CPP_code = CPP_code2;

    if VI_dW_dp.dtype == np.float32:
        CPP_code = CPP_code.replace("double", "float");

    CPP_code = """
    typedef double myDbl;
    """ + CPP_code.replace("double", "float");

    scipy.weave.inline(CPP_code, ["VI_dW_dp", "N_p", "w", "H"]);

    #common.DebugPrint("res[1, 0] = %d" % res[1, 0]);
    #common.DebugPrint("\n\nres = %s" % str(res));

    print("hessian2(): H = %s" % str(H));

    # res is H
    return H;

hessian = hessian_scipy;


#function out=my_morph(locs,radius,h,w)
def my_morph(locs, radius, h, w):
    # It seems this function is actually not executed
    #!!!!TODO: check that implementation is OK

    assert False;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("Entered my_morph(radius=%s, h=%s, w=%s)." % \
                            (str(radius), str(h), str(w)));

    zl = locs < 1;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("my_morph(): zl.shape = %s" % str(zl.shape));
        common.DebugPrint("my_morph(): zl.sum(1).shape = %s" % \
                                                    str(zl.sum(1).shape));
    """
    From Matlab help http://www.mathworks.com/help/matlab/ref/logical.html:
        "L = logical(A) converts numeric input A into an array of logical values.
        Any nonzero element of input A is converted to logical 1 (true) and
            zeros are converted to logical 0 (false).
        Complex values and NaNs cannot be converted to logical values and
            result in a conversion error."
    """
    #locs=locs(~logical(sum(zl,2)),:);
    locs = locs[np.logical_not(zl.sum(1)), :];

    inArray = np.zeros( (h, w) );

    #for i=1:length(locs)
    for i in range(1, len(locs) + 1):
        #if round(locs(i,2))<w && round(locs(i,2))>1 && round(locs(i,1))<h && round(locs(i,1))>1
        if (round(locs[i - 1, 1]) < w) and (round(locs[i - 1, 1]) > 1) and \
                (round(locs[i - 1, 0]) < h) and (round(locs[i - 1, 0]) > 1):
            #in(round(locs(i,1)),round(locs(i,2)))=1;
            inArray[round(locs[i - 1, 0]), round(locs[i - 1, 1])] = 1;

    #[x,y]=meshgrid(-radius:radius,-radius:radius);
    #y, x = np.mgrid[-radius: radius + 1, -radius: radius + 1];
    x, y = Matlab.meshgrid( range(-radius, radius + 1 + 1), \
                            range(-radius, radius + 1 + 1) );

    #temp=x.^2+y.^2<=radius^2;
    temp = (x**2 + y**2 <= pow(radius, 2));

    #out=in.*0;
    out = inArray * 0;

    #for i=radius+1:h-radius
    #    for j=radius+1:w-radius
    for i in range(radius + 1, h - radius + 1):
        for j in range(radius + 1, w - radius + 1):
            #if in(i,j)==1
            if inArray[i - 1, j - 1] == 1:
                #out(i-radius:i+radius,j-radius:j+radius)=temp|out(i-radius:i+radius,j-radius:j+radius);
                out[i - radius - 1: i + radius, j - radius - 1: j + radius] = \
                    np.logical_or(temp, \
                        out[i - radius - 1: i + radius, j - radius - 1: j + radius]);

    return out;




#function sd_delta_p = sd_update(VI_dW_dp, error_img, N_p, w)
def sd_update(VI_dW_dp, error_img, N_p, w):
    #if nargin<4 error('Not enough input arguments'); end

    #sd_delta_p = zeros(N_p, 1);
    sd_delta_p = np.zeros( (N_p, 1) );

    #for p=1:N_p
    """
    We substitute p - 1 with p, since in Python arrays indices start from 0,
         not 1 like in Matlab.
    """
    #for p in range(1, N_p + 1):
    for p in range(N_p):
        #h1 = VI_dW_dp(:,((p-1)*w)+1:((p-1)*w)+w);
        h1 = VI_dW_dp[:, (p * w) + 1 - 1: (p * w) + w];

        #sd_delta_p(p) = sum(sum(h1 .* error_img));
        sd_delta_p[p] = (h1 * error_img).sum(); #.sum();

    return sd_delta_p;


#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#function [warp_p,t] = update_step(warp_p,t,delta_p,time_flag)
def update_step(warp_p, t, delta_p, time_flag):
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("update_step(): warp_p.shape = %s" % \
                                            str(warp_p.shape));
        common.DebugPrint("update_step(): warp_p = %s" % str(warp_p));
        common.DebugPrint("update_step(): delta_p.shape = %s" % \
                                            str(delta_p.shape));
        common.DebugPrint("update_step(): delta_p = %s" % str(delta_p));
        common.DebugPrint("update_step(): t = %s" % str(t));

    assert delta_p.ndim < 3;

    #% Compute and apply the update

    if time_flag == 1:
        #t=t+delta_p[end];
        t += delta_p[-1]; # t is scalar

        #delta_p = [delta_p(1:end-1); 0];
        delta_p = np.r_[np.ravel(delta_p, order="F")[: -1], 0];
    else:
        #delta_p = [delta_p(1:end); 0];
        delta_p = np.r_[np.ravel(delta_p, order="F"), 0]; # np.zeros()... Try also np.r_[np.ravel(delta_p)]

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("update_step(): delta_p.shape = %s" % \
                                                str(delta_p.shape));

    #delta_p=reshape(delta_p,3,3);
    delta_p = delta_p.reshape( (3, 3), order="F" );

    #warp_p = warp_p + delta_p;
    warp_p = warp_p + delta_p;

    #warp_p(3,3) = 1;
    warp_p[2, 2] = 1.0;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("update_step(): warp_p.shape = %s" % \
                                                str(warp_p.shape));
        common.DebugPrint("update_step(): warp_p = %s" % str(warp_p));
        common.DebugPrint("update_step(): delta_p.shape = %s" % \
                                                str(delta_p.shape));
        common.DebugPrint("update_step(): delta_p = %s" % str(delta_p));

    return warp_p, t;


# Note: r_capture is actually the cv2.VideoCapture object.
#function [out]=short_time_seq(r_path, index, nof, imres, imformat)
def short_time_seq(r_capture, index, nof, imres, imformat):
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("short_time_seq(): imres = %s" % str(imres));
        common.DebugPrint("short_time_seq(): index = %s" % str(index));
        common.DebugPrint("short_time_seq(): nof = %s" % str(nof));

    """
    # We don't need to do resizing here because imres is already "resized" if
    #       it's the case
    if config.VIDEO_FRAME_RESIZE_SCALING_FACTOR != 1:
        out = np.empty( (imres[0] * config.VIDEO_FRAME_RESIZE_SCALING_FACTOR, \
                         imres[1] * config.VIDEO_FRAME_RESIZE_SCALING_FACTOR, \
                         nof), \
                       dtype=MATRIX_FLOAT_TYPE);
    else:
        #out = np.zeros( (imres[0], imres[1], nof), dtype=MATRIX_FLOAT_TYPE);
        out = np.empty( (imres[0], imres[1], nof), dtype=MATRIX_FLOAT_TYPE);
    """

    #% nof: number of frames
    #out=zeros([imres nof]);
    #out = np.zeros( (imres[0], imres[1], nof) );
    # This gives rather big differences in warp: out = np.zeros( (imres[0], imres[1], nof), dtype=np.uint8);
    out = np.empty( (imres[0], imres[1], nof), dtype=MATRIX_FLOAT_TYPE);

    #for i=-(nof-1)/2:(nof-1)/2
    for i in range(-(nof-1) / 2, (nof-1) / 2 + 1):
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("short_time_seq(): i = %d" % i);
            common.DebugPrint("short_time_seq(): i + (nof + 1) / 2 = %d" % \
                                                        (i + (nof + 1) / 2));

        #%Alex: sometimes it crashes here saying it doesn't find an image like 1999.jpeg, when the first frame is 2000.jpeg
        #fileName = [r_path num2str(index+i,'%.6d') imformat];

        #% Alex
        #%index
        #%i
        #%fileName

        #out(:,:,i+(nof+1)/2)=imread(fileName);
        """
        This error we get if we read an RGB frame, instead of gray (8bpp).
        ValueError: operands could not be broadcast together with
            shapes (240,320) (240,320,3).
        """
        out[:, :, i + (nof + 1) / 2 - 1] = MyImageRead(r_capture, index + i);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint( \
                "short_time_seq(): " \
                "out[:, :, i + (nof + 1) / 2 - 1].shape = %s" % \
                    str(out[:, :, i + (nof + 1) / 2 - 1].shape));
            common.DebugPrint( \
                "short_time_seq(): " \
                "out[:, :, i + (nof + 1) / 2 - 1].dtype = %s" % \
                    str(out[:, :, i + (nof + 1) / 2 - 1].dtype));

    return out;


# This function is available at http://www.mathworks.com/matlabcentral/fileexchange/25397-imgaussian
"""
function I=imgaussian(I,sigma,siz)
% IMGAUSSIAN filters an 1D, 2D color/greyscale or 3D image with an
% Gaussian filter. This function uses for filtering IMFILTER or if
% compiled the fast  mex code imgaussian.c . Instead of using a
% multidimensional gaussian kernel, it uses the fact that a Gaussian
% filter can be separated in 1D gaussian kernels.
%
% J=IMGAUSSIAN(I,SIGMA,SIZE)
%
% inputs,
%   I: The 1D, 2D greyscale/color, or 3D input image with
%           data type Single or Double
%   SIGMA: The sigma used for the Gaussian kernel
%   SIZE: Kernel size (single value) (default: sigma*6)
%
% outputs,
%   J: The gaussian filtered image
%
% note, compile the code with: mex imgaussian.c -v
%
% example,
%   I = im2double(imread('peppers.png'));
%   figure, imshow(imgaussian(I,10));
%
% Function is written by D.Kroon University of Twente (September 2009)
"""
#function I=imgaussian(I,sigma,siz)
def imgaussian(I, sigma, siz=-1):
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "Entered imgaussian(I.shape=%s, sigma=%s, siz=%s)" % \
                                (str(I.shape), str(sigma), str(siz)));

    # Used only when doing multi-level ECC
    # TODO: Not tested WELL

    #if(~exist('siz','var')), siz=sigma*6; end
    if siz == -1:
        siz=sigma*6;

    """
    From opencv2refman.pdf (also http://docs.opencv.org/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html#gaussian-filter):
      cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType ]]]) -> dst
        src - input image; the image can have any number of channels, which
            are processed inde-pendently, but the depth should be CV_8U,
            CV_16U, CV_16S, CV_32F or CV_64F.

        dst - output image of the same size and type as src.

        ksize - Gaussian kernel size. ksize.width and ksize.height can differ
            but they both must be positive and odd.
            Or, they can be zero's and then they are computed from sigma* .

        sigmaX - Gaussian kernel standard deviation in X direction.

        sigmaY - Gaussian kernel standard deviation in Y direction; if sigmaY is zero,
            it is set to be equal to sigmaX, if both sigmas are zeros, they
            are computed from ksize.width and ksize.height, respectively
            (see getGaussianKernel() for details); to fully control the
            result regardless of possible future modifications of all this
            semantics, it is recommended to specify all of ksize, sigmaX, and
            sigmaY.

        borderType - pixel extrapolation method (see borderInterpolate() for details).
            The function convolves the source image with the specified
            Gaussian kernel. In-place filtering is supported.
    """


    """
    From http://docs.opencv.org/modules/imgproc/doc/filtering.html#void%20GaussianBlur%28InputArray%20src,%20OutputArray%20dst,%20Size%20ksize,%20double%20sigmaX,%20double%20sigmaY,%20int%20borderType%29
     cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
      Note: was getting an exception:
        "SystemError: new style getargs format but argument is not a tuple"
      when giving:
        res = cv2.GaussianBlur(src=I, ksize=(siz, siz), sigmaX=sigma);
       (see solution below and at
        http://stackoverflow.com/questions/13225525/system-error-new-style-getargs-format-but-argument-is-not-a-tuple-when-using)
    """

    #TODO: test if GaussianBlur is equivalent
    #I = I.T;
    siz = int(siz);
    res = cv2.GaussianBlur(src=I, ksize=(siz, siz), sigmaX=sigma);

    #res = res.T;

    return res;


    #function I=imgaussian(I,sigma,siz)
    #% IMGAUSSIAN filters an 1D, 2D color/greyscale or 3D image with an
    #% Gaussian filter. This function uses for filtering IMFILTER or if
    #% compiled the fast  mex code imgaussian.c . Instead of using a
    #% multidimensional gaussian kernel, it uses the fact that a Gaussian
    #% filter can be separated in 1D gaussian kernels.
    #%
    #% J=IMGAUSSIAN(I,SIGMA,SIZE)
    #%
    #% inputs,
    #%   I: The 1D, 2D greyscale/color, or 3D input image with
    #%           data type Single or Double
    #%   SIGMA: The sigma used for the Gaussian kernel
    #%   SIZE: Kernel size (single value) (default: sigma*6)
    #%
    #% outputs,
    #%   J: The gaussian filtered image
    #%
    #% note, compile the code with: mex imgaussian.c -v
    #%
    #% example,
    #%   I = im2double(imread('peppers.png'));
    #%   figure, imshow(imgaussian(I,10));
    #%
    #% Function is written by D.Kroon University of Twente (September 2009)

    #if(~exist('siz','var')), siz=sigma*6; end

    if sigma > 0:
        #% Make 1D Gaussian kernel
        #x=-ceil(siz/2):ceil(siz/2);
        x = range(-math.ceil(siz / 2.0), math.ceil(siz / 2.0) + 1);

        #H = exp(-(x.^2/(2*sigma^2)));
        H = np.exp(-(x ** 2 / (2.0 * pow(sigma, 2))));

        #H = H/sum(H(:));
        H = H / H[:].sum();

        #% Filter each dimension with the 1D Gaussian kernels\
        #if(ndims(I)==1)
        #if len(I.shape) == 1:
        if I.ndim == 1:
            assert False;
            #I=imfilter(I,H, 'same' ,'replicate');
            I = imfilter(I, H, "same", "replicate");
        #elseif(ndims(I)==2)
        elif I.ndim == 2:
            #Hx=reshape(H,[length(H) 1]);
            Hx = H.reshape((H.size, 1), order="F");

            #Hy=reshape(H,[1 length(H)]);
            Hy = H.reshape((1, H.size), order="F");

            #I=imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
            I = imfilter( \
                    imfilter(I, Hx, "same", "replicate"), \
                    Hy, "same", "replicate");
        #elseif(ndims(I)==3)
        elif I.ndim == 3:
            assert False;
            """
            if(size(I,3)<4) % Detect if 3D or color image
                Hx=reshape(H,[length(H) 1]);
                Hy=reshape(H,[1 length(H)]);
                for k=1:size(I,3)
                    I(:,:,k)=imfilter(imfilter(I(:,:,k),Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
                end
            else
                Hx=reshape(H,[length(H) 1 1]);
                Hy=reshape(H,[1 length(H) 1]);
                Hz=reshape(H,[1 1 length(H)]);
                I=imfilter(imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate'),Hz, 'same' ,'replicate');
            """
        else:
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "imgaussian:input: unsupported input dimension");
            assert False;

    return I;


"""
TODO: See Paper TPAMI2013, Appendix II for details on computation J_{\Phi}.
jacobian_h is already implemented in the C++ OpenCV of ECC image (not video)
registration implementation as
    calculate_jacobian_homography(src1, src2, jacobian, map_matrix)??
"""
#function dW_dp = jacobian_h(nx, ny, warp_p,time_extention_flag,pixel_select)
def jacobian_h(nx, ny, warp_p, time_extension_flag, pixel_select):
    #if common.MY_DEBUG_STDOUT:
    print( \
            "Entered jacobian_h(nx.shape=%s, ny.shape=%s, warp_p.shape=%s, " \
                                "time_extension_flag=%d, pixel_select=%d)" % \
                            (str(nx.shape), str(ny.shape), str(warp_p.shape), \
                                        time_extension_flag, pixel_select));

    if pixel_select == 0:
        #snx=numel(nx);
        snx = nx.size;

        #sny=numel(ny);
        sny = ny.size;

        #jacob_x = kron(nx,ones(sny, 1));
        jacob_x = Matlab.kron(nx, np.ones( (sny, 1) ) );

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("jacobian_h(): jacob_x.shape = %s" % \
                                                        str(jacob_x.shape));

        #jacob_y = kron([ny]',ones(1,snx));
        jacob_y = Matlab.kron(np.array([ny]).T, np.ones( (1, snx) ) );

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("jacobian_h(): jacob_y.shape = %s" % \
                                                        str(jacob_y.shape));

        #% jacob_x=nx(:,ones(1,sny))';
        #% jacob_y=ny(:,ones(1,snx));

        #jacob_zero = zeros(sny, snx);
        jacob_zero = np.zeros( (sny, snx) );

        #jacob_one = ones(sny, snx);
        jacob_one = np.ones( (sny, snx) );



        #% % Easy bits
        #% jac_x = kron([0:nx - 1],ones(ny, 1));
        #% jac_y = kron([0:ny - 1]',ones(1, nx));
        #% jac_zero = zeros(ny, nx);
        #% jac_one = ones(ny, nx);

        #%     % Complicated bits are just homography of all image coordinates
        #%
        #xy = [jacob_x(:) jacob_y(:)];
        xy = np.c_[np.ravel(jacob_x, order="F"), \
                    np.ravel(jacob_y, order="F")];

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("jacobian_h(): xy.shape = %s" % str(xy.shape));

        #xy = [xy ones(length(xy),1)]';
        xy = np.c_[xy, np.ones( (len(xy), 1) )].T;

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("jacobian_h(): xy.shape (2nd) = %s" % \
                                                        str(xy.shape));

        M = warp_p;
        #%     M(1,1) = M(1,1) + 1;
        #%     M(2,2) = M(2,2) + 1;

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("jacobian_h(): M = %s" % str(M));
            common.DebugPrint("jacobian_h(): xy.shape = %s" % str(xy.shape));
            #common.DebugPrint("jacobian_h(): xy = %s" % str(xy));

        #uv = M * xy;
        uv = np.dot(M, xy); # Matrix multiplication

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint( \
                    "jacobian_h(): uv.shape = %s" % str(uv.shape));
            common.DebugPrint( \
                    "jacobian_h(): uv[:20, :20] = %s" % str(uv[:20, :20]));

        #uvc = uv ./ repmat(uv(3,:),3,1);
        uvc = uv / np.tile(uv[2, :], (3, 1) );


        #u_x = reshape(uvc(1,:),sny,snx);
        u_x = uvc[0, :].reshape((sny, snx), order="F");

        #u_y = reshape(uvc(2,:),sny,snx);
        u_y = uvc[1, :].reshape((sny, snx), order="F");

        #v = reshape(uv(3,:),sny,snx);
        v = uv[2, :].reshape((sny, snx), order="F");

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("jacobian_h(): v.shape = %s" % str(v.shape));
            common.DebugPrint( \
                        "jacobian_h(): v[:20, :20] = %s" % str(v[:20, :20]));

        #% Divide each jacobian image by v
        #iv = 1 ./ v;
        iv = 1.0 / v; #.astype(np.float64);

        #jacob_x = iv .* jacob_x;
        jacob_x = iv * jacob_x;

        #jacob_y = iv .* jacob_y;
        jacob_y = iv * jacob_y;

        #jacob_one = iv .* jacob_one;
        jacob_one = iv * jacob_one;

        if time_extension_flag == 0:
            """
            dW_dp = [jacob_x, jacob_zero, -jacob_x .* u_x, jacob_y, jacob_zero, -jacob_y .* u_x, jacob_one, jacob_zero;
                jacob_zero, jacob_x, -jacob_x .* u_y, jacob_zero, jacob_y, -jacob_y .* u_y, jacob_zero, jacob_one];
            """
            #TODO: check if OK
            dW_dp = np.r_[ \
                np.c_[jacob_x, jacob_zero, -jacob_x * u_x, jacob_y, jacob_zero, -jacob_y * u_x, jacob_one, jacob_zero], \
                np.c_[jacob_zero, jacob_x, -jacob_x * u_y, jacob_zero, jacob_y, -jacob_y * u_y, jacob_zero, jacob_one] ];
        else:
            #%%%extension in time
            """
            dW_dp = [jacob_x, jacob_zero, -jacob_x .* u_x, jacob_y, jacob_zero, -jacob_y .* u_x, jacob_one, jacob_zero,jacob_zero;
                jacob_zero, jacob_x, -jacob_x .* u_y, jacob_zero, jacob_y, -jacob_y .* u_y, jacob_zero, jacob_one,jacob_zero;
                jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_one.*v];
            """
            #TODO: check if OK
            dW_dp = np.r_[ \
                np.c_[jacob_x, jacob_zero, -jacob_x * u_x, jacob_y, jacob_zero, -jacob_y * u_x, jacob_one, jacob_zero, jacob_zero], \
                np.c_[jacob_zero, jacob_x, -jacob_x * u_y, jacob_zero, jacob_y, -jacob_y * u_y, jacob_zero, jacob_one, jacob_zero], \
                np.c_[jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_zero, jacob_one * v] ];
    else:
        #j_z=zeros(size(nx));
        assert nx.ndim == 2;
        j_z = np.zeros(nx.shape);

        #j_ones=ones(size(nx));
        assert nx.nimd == 2; #otherwise make np.ones((nx.shape[0], nx.shape[0]))
        j_ones = np.ones(nx.shape);

        #xy=[nx';ny';ones(1,length(nx))];
        xy = np.r_[nx.T, ny.T, np.ones( (1, len(nx)) )];

        M = warp_p;
        #%     M(1,1) = M(1,1) + 1;
        #%     M(2,2) = M(2,2) + 1;

        #uv = M*xy;
        uv = np.dot(M, xy); # Matrix multiplication

        #j_o=j_ones./uv(3,:)';
        j_o = j_ones.astype(np.float64) / uv[2, :].T;

        #nx=nx./uv(3,:)';
        nx = nx.astype(np.float64) / uv[2, :].T;

        #ny=ny./uv(3,:)';
        ny = ny.astype(np.float64) / uv[2, :].T;

        #uv = uv ./ repmat(uv(3,:),3,1);
        uv = uv.astype(np.float64) / np.tile(uv[2, :], (3, 1) );

        #uv = uv(1:2,:)';
        uv = uv[0:2, :].T;

        #x_prime = uv(:,1);
        x_prime = uv[:, 0];

        #y_prime = uv(:,2);
        y_prime = uv[:, 1];

        #%      if time_extension_flag==0
        """
        dW_dp=[nx,j_z,-nx.*x_prime,ny,j_z,-ny.*x_prime,j_o,j_z;
                j_z,nx,-nx.*y_prime,j_z,ny,-ny.*y_prime,j_z,j_o];
        """
        #TODO: check if OK
        dW_dp = np.r_[ \
                np.c_[nx, j_z, -nx * x_prime, ny, j_z, -ny * x_prime, j_o, j_z], \
                np.c_[j_z, nx, -nx * y_prime, j_z, ny, -ny * y_prime, j_z, j_o] ];

        #%      else
        #%
        #%          dW_dp=[nx,j_z,-nx.*x_prime,ny,j_z,-ny.*x_prime,j_o,j_z,j_z;
        #%                  j_z,nx,-nx.*y_prime,j_z,ny,-ny.*y_prime,j_z,j_o,j_z;
        #%                  j_z,j_z,j_z,j_z,j_z,j_z,j_z,j_z,j_ones];
        #%      end

    print("jacobian_h(): dW_dp.shape = %s" % str(dW_dp.shape));
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "jacobian_h(): dW_dp[:10, :10] = %s" % str(dW_dp[:10, :10]));
        common.DebugPrint( \
            "jacobian_h(): dW_dp[-10:, -10:] = %s" % str(dW_dp[-10:, -10:]));

    return dW_dp;


#function wimg_time = interp_time(volume,n)
def interp_time(volume, n):
    if n < 1:
        n = 1;

    #if n>size(volume,3)
    if n > volume.shape[2]:
        #n=size(volume,3);
        n = volume.shape[2];

    #bot=floor(n);
    bot = math.floor(n);

    #top=ceil(n);
    top = math.ceil(n);

    sub = n - bot;

    #wimg_time=sub*volume(:,:,top)+(1-sub)*volume(:,:,bot);
    wimg_time = sub * volume[:, :, top - 1] + (1 - sub) * volume[:, :, bot - 1];

    return wimg_time;


#function wimg = interp_space_time(volume, img_index, p, t, f_type, nx, ny, pixel_select)
#!!!!TODO: img_index is NOT used
def interp_space_time(volume, img_index, p, t, f_type, nx, ny, pixel_select):
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "Entered interp_space_time(p=%s, t=%s, nx=%s, ny=%s, pixel_select=%s)." % \
                        (str(p), str(t), str(nx), str(ny), str(pixel_select)));
        common.DebugPrint( \
                        "interp_space_time(): volume[: 10, : 10, 0] = %s" % \
                                                (str(volume[: 10, : 10, 0])));
        common.DebugPrint( \
                        "interp_space_time(): volume.shape = %s" % \
                                                (str(volume.shape)));
        common.DebugPrint("interp_space_time(): volume.dtype = %s" % \
                                                    (str(volume.dtype)));
        common.DebugPrint("interp_space_time(): nx.shape = %s" % \
                                                    (str(nx.shape)));
        common.DebugPrint("interp_space_time(): nx.dtype = %s" % \
                                                    (str(nx.dtype)));
        common.DebugPrint("interp_space_time(): ny.shape = %s" % \
                                                    (str(ny.shape)));
        common.DebugPrint("interp_space_time(): ny.dtype = %s" % \
                                                    (str(ny.dtype)));

    assert f_type == "linear";

    #%interpolation in time
    #[aa,bb,c]=size(volume);
    aa, bb, c = volume.shape;
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("interp_space_time(): aa = %s, bb = %s, c = %s" % \
                                                (str(aa), str(bb), str(c)));

    #wimg_time=interp_time(volume,(c+1)/2+t);
    wimg_time = interp_time(volume, (c + 1) / 2.0 + t);

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
                    "interp_space_time(): wimg_time.shape (before) = %s" % \
                                                        str(wimg_time.shape));
        common.DebugPrint( \
                    "interp_space_time(): wimg_time.dtype (before) = %s" % \
                                                        str(wimg_time.dtype));
        common.DebugPrint( \
                "interp_space_time(): wimg_time[:20, :20] (before) = %s" % \
                                                    str(wimg_time[:20, :20]));

    # Note: xg, yg, xy are used to matrix-multiply xy with M (==p)
    if pixel_select == 0:
        #% Get all points in destination to sample
        #[xg yg] = meshgrid(nx, ny);
        #!!!!TODO: can we avoid using xg and yg since they are very big - I guess I use them for vectorization - I could discard xg and yg and use loops maybe
        # The types of xg and yg are the ones of nx and ny - We assume the resolution is at most 32K*32K
        nx = nx.astype(np.uint16);
        ny = ny.astype(np.uint16);
        xg, yg = Matlab.meshgrid(nx, ny);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("interp_space_time(): xg.shape = %s" % \
                                                            str(xg.shape));
            common.DebugPrint("interp_space_time(): xg.dtype = %s" % \
                                                            str(xg.dtype));
            common.DebugPrint("interp_space_time(): xg[: 10, : 10] = %s" % \
                                                        str(xg[: 10, : 10]));

            common.DebugPrint("interp_space_time(): yg.shape = %s" % \
                                                            str(yg.shape));
            common.DebugPrint("interp_space_time(): yg.dtype = %s" % \
                                                            str(yg.dtype));
            common.DebugPrint("interp_space_time(): yg[: 10, : 10] = %s" % \
                                                        str(yg[: 10, : 10]));


        #xy = [reshape(xg, numel(xg), 1)'; reshape(yg, numel(yg), 1)'];
        # Note: reshape in Matlab goes column-major order
        xgr = np.ravel(xg, order="F");
        ygr = np.ravel(yg, order="F");

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("interp_space_time(): xgr.shape = %s" % \
                                                            str(xgr.shape));
            common.DebugPrint("interp_space_time(): xgr.dtype = %s" % \
                                                            str(xgr.dtype));
            common.DebugPrint("interp_space_time(): ygr.shape = %s" % \
                                                            str(ygr.shape));
            common.DebugPrint("interp_space_time(): ygr.dtype = %s" % \
                                                            str(ygr.dtype));

        if FOOBAR:
            # np.r_ puts the matrices one after the other on the horizontal...
            #xy = np.r_[np.ravel(xg, order="F").T, np.ravel(yg, order="F").T];

            # np.r_ puts the matrices one after the other on the horizontal...
            xy = np.r_[xgr, ygr];
            #xy = np.r_[xgr.T, ygr.T];
        else:
            xy = np.vstack( (xgr, ygr) );
        # END: xy = [reshape(xg, numel(xg), 1)'; reshape(yg, numel(yg), 1)'];


        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("interp_space_time(): xy.shape = %s" % \
                                                            str(xy.shape));
            common.DebugPrint("interp_space_time(): xy.dtype = %s" % \
                                                            str(xy.dtype));
            common.DebugPrint("interp_space_time(): xy[: 2, : 1000] = %s" % \
                                                        str(xy[: 2, : 1000]));

        #xy = [xy; ones(1,size(xy,2))];
        assert xy.ndim == 2;
        #xy = np.r_[xy, np.array([1])]; #np.ones( (1, 1) )];
        #xy = np.r_[xy, np.ones( (1, xy.shape[1]) )];
        xy = np.r_[xy, \
                   np.ones( (1, xy.shape[1]), dtype=MATRIX_FLOAT_TYPE)]. \
                        astype(MATRIX_FLOAT_TYPE);

        #% transformation
        M = p;

        #% M(1,1) = M(1,1) + 1;
        #% M(2,2) = M(2,2) + 1;
        #M(3,3) = 1;
        M[2, 2] = 1;

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("interp_space_time(): M.shape = %s" % \
                                                            str(M.shape));
            common.DebugPrint("interp_space_time(): M = %s" % str(M));
            common.DebugPrint("interp_space_time(): xy.shape (after) = %s" % \
                                                            str(xy.shape));
            common.DebugPrint("interp_space_time(): xy.dtype (after) = %s" % \
                                                            str(xy.dtype));

        #% Transform
        #uv = M * xy;
        #uv = np.dot(M, xy); # Matrix multiplication
        uv = np.dot(M.astype(MATRIX_FLOAT_TYPE), xy); # Matrix multiplication

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("interp_space_time(): uv.shape = %s" % \
                                                            str(uv.shape));
            common.DebugPrint("interp_space_time(): uv.dtype = %s" % \
                                                            str(uv.dtype));
            common.DebugPrint("interp_space_time(): uv[:20, :20] = %s" % \
                                                        str(uv[:20, :20]));

        #% Divide for homography
        #uv = uv ./ repmat(uv(3,:),3,1);
        #uv = uv.astype(np.float64) / np.tile(uv[2, :], (3, 1));
        try:
            """
            In very special cases we get the following exceptions:
            - <<RuntimeWarning: divide by zero encountered in divide>>
            - <<RuntimeWarning: invalid value encountered in divide>>
            """
            uv = uv / np.tile(uv[2, :], (3, 1));
        except:
            #!!!!TODO: think if we can we do something more intelligent?
            common.DebugPrintErrorTrace();
            #pass;

        #% Remove homogeneous
        #uv = uv(1:2,:)';
        uv = uv[0: 2, :].T;

        #% Sample
        #xi = reshape(uv(:,1),numel(ny),numel(nx));
        xi = uv[:, 0].reshape( (ny.size, nx.size), order="F");

        #yi = reshape(uv(:,2),numel(ny),numel(nx));
        yi = uv[:, 1].reshape( (ny.size, nx.size), order="F");

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("interp_space_time(): xi.shape = %s" % \
                                                        str(xi.shape));
            common.DebugPrint("interp_space_time(): yi.shape = %s" % \
                                                        str(yi.shape));
    else:
        #xy=[nx';ny';ones(1,length(nx))];
        xy = np.r_[nx.T, ny.T, np.ones( (1, len(nx) ))];
        #%xy(1,end)

        #% transformation
        M = p;
        #% M(1,1) = M(1,1) + 1;
        #% M(2,2) = M(2,2) + 1;
        #M(3,3) = 1;
        M[2, 2] = 1;

        #% Transform
        #uv = M * xy;
        uv = np.dot(M, xy); # Matrix multiplication

        #% Divide for homography
        #uv = uv ./ repmat(uv(3,:),3,1);
        uv = uv / np.tile(uv[2, :], (3,1));

        #% Remove homogeneous
        #uv = uv(1:2,:)';
        uv = uv[0: 2, :].T;

        #xi=uv(:,1);
        xi = uv[:, 0];

        #yi=uv(:,2);
        yi = uv[:, 1];

    #% wimg = interp2(wimg_time, xi, yi, f_type);
    #% xi=xi+180;
    #% yi=yi+135;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("interp_space_time(): wimg_time.shape = %s" % \
                                                        str(wimg_time.shape));
        common.DebugPrint("interp_space_time(): wimg_time.dtype = %s" % \
                                                        str(wimg_time.dtype));
        common.DebugPrint("interp_space_time(): wimg_time[:10, :10] = %s" % \
                                                    str(wimg_time[:10, :10]));

        common.DebugPrint("interp_space_time(): xi.shape = %s" % \
                                                    str(xi.shape));
        common.DebugPrint("interp_space_time(): xi.dtype = %s" % \
                                                    str(xi.dtype));
        common.DebugPrint("interp_space_time(): xi[:10, :10] = %s" % \
                                                    str(xi[:10, :10]));

        common.DebugPrint("interp_space_time(): yi.shape = %s" % \
                                                            str(yi.shape));
        common.DebugPrint("interp_space_time(): yi.dtype = %s" % \
                                                            str(yi.dtype));
        common.DebugPrint("interp_space_time(): yi[:10, :10] = %s" % \
                                                        str(yi[:10, :10]));

    #wimg = interp2(wimg_time, xi, yi, f_type);
    wimg = Matlab.interp2(wimg_time, xi, yi, f_type);

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("interp_space_time(): wimg.shape = %s" % \
                                                            str(wimg.shape));
        common.DebugPrint("interp_space_time(): wimg.dtype = %s" % \
                                                            str(wimg.dtype));
        common.DebugPrint("interp_space_time(): wimg[:10, :10] = %s" % \
                                                        str(wimg[:10, :10]));

    #% Check for NaN background pixels - replace them with a background of 0
    #idx = find(isnan(wimg));
    if ORDER_MATTERS:
        idxC, idxR = np.nonzero(np.isnan(wimg).T);
        idx = (idxR, idxC);
    else:
        # idx is a pair of arrays, (rowI, colI)
        idx = np.nonzero(np.isnan(wimg));

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("interp_space_time(): idx len =(%s, %s)" % \
                                    (str(idx[0].shape), str(idx[1].shape)));
        common.DebugPrint("interp_space_time(): idx = %s" % str(idx));

    #if ~isempty(idx)
    assert idx[0].size == idx[1].size;
    if idx[0].size != 0:
        #wimg(idx) = 0;
        if config.VISUAL_DIFF_FRAMES == True:
            wimg[idx] = 255.0;
        else:
            wimg[idx] = 0.0;

    return wimg;


#function w=weights_for_ecc(a,b,grid,nonze)
def weights_for_ecc(a, b, grid, nonze):
    #% a=imgaussian(a,3);
    #% b=imgaussian(b,3);
    #% a=imresize(a,.5);
    #% b=imresize(b,.5);

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("Entered weights_for_ecc(a.shape=%s, b.shape=%s, " \
                "grid=%s, nonze.shape=%s)" % \
                (str(a.shape), str(b.shape), str(grid), str(nonze.shape)));

    # Note: grid is for example [11, 11]

    #[m,n] = size(a);
    m, n = a.shape;

    #w=zeros([grid]);
    assert len(grid) == 2;
    w = np.zeros( grid );

    #ze = nonze~=0;
    if False:
        ze = nonze != 0;
    else:
        ze = nonze;
    #ze = np.abs(nonze) > 0.000001; # In some cases, it gives NO False while in Matlab it does
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("weights_for_ecc(): len(nonzero(ze)) = %s" % \
                                    str(len(np.nonzero(ze))));

    # Note: In Matlab nonze has only one zero :)

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "weights_for_ecc(): a[:10, :10] = %s" % str(a[:10, :10]));
        common.DebugPrint( \
            "weights_for_ecc(): b[:10, :10] = %s" % str(b[:10, :10]));

        common.DebugPrint( \
            "weights_for_ecc(): nonze.shape = %s" % str(nonze.shape));
        common.DebugPrint( \
            "weights_for_ecc(): ze.shape = %s" % str(ze.shape));

    #if True:
    if False:
        # TOO BIG:
        common.DebugPrint("weights_for_ecc(): nonze = %s" % str(nonze));
        common.DebugPrint("weights_for_ecc(): ze = %s" % str(ze));

    #zw = imresize(double(ze), grid);
    zw = Matlab.imresize(ze.astype(np.float64), newSize=grid); # NOT helpful: interpolationMethod=cv2.INTER_LANCZOS4);

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("weights_for_ecc(): zw.shape = %s" % str(zw.shape));
        common.DebugPrint("weights_for_ecc(): zw = %s" % str(zw));

    #w0 = n/grid(2);
    w0 = float(n) / grid[1];

    #h0 = m/grid(1);
    h0 = float(m) / grid[0];

    #ab=a.*b;
    ab = a * b;

    #for i=1:grid(1)
    for i in range(1, grid[0] + 1):
        for j in range(1, grid[1] + 1):
            #%if zw(i,j)==1
            #abn=ab(round(h0)*(i-1)+1:round(h0*i),round(w0)*(j-1)+1:round(w0*j));
            abn = ab[round(h0) * (i - 1) + 1 - 1: round(h0 * i), \
                        round(w0) * (j - 1) + 1 - 1: round(w0 * j)];

            #w(i,j)=sum(abn(:));
            #w[i - 1, j - 1] = abn.ravel(order="F").sum();
            w[i - 1, j - 1] = abn.sum();
            #%end

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("weights_for_ecc(): w.shape (first) = %s" % \
                                                            str(w.shape));
        common.DebugPrint("weights_for_ecc(): w (first) = %s" % str(w));

    #w=w/norm(a(:))/norm(b(:));
    denom1 = npla.norm(np.ravel(a, order="F"));
    denom2 = npla.norm(np.ravel(b, order="F"));

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("weights_for_ecc(): denom1 = %s" % str(denom1));
        common.DebugPrint("weights_for_ecc(): denom2 = %s" % str(denom2));

        if (denom1 < 1e-5) or (denom2 < 1e-5):
            common.DebugPrint("weights_for_ecc(): a.shape = %s" % \
                                                                str(a.shape));
            common.DebugPrint("weights_for_ecc(): b.shape = %s" % \
                                                                str(b.shape));
            common.DebugPrint("weights_for_ecc(): a[:30, :30] = %s" % \
                                                            str(a[:30, :30]));
            common.DebugPrint("weights_for_ecc(): b[:30, :30] = %s" % \
                                                            str(b[:30, :30]));

    w = w / denom1 / denom2;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("weights_for_ecc(): w.shape (2nd) = %s" % \
                                                            str(w.shape));
        common.DebugPrint("weights_for_ecc(): w (2nd) = %s" % str(w));

    #w(w<0)=0;
    w[w < 0] = 0;

    #w = w.*zw;
    w = w * zw;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("weights_for_ecc(): w.shape (3rd) = %s" % \
                                                            str(w.shape));
        common.DebugPrint("weights_for_ecc(): w (3rd) = %s" % str(w));

    #%w(w<(1/prod(grid)))=0;

    #% w=w-min(min(w));
    # Note: In Matlab, max(w) returns the max of columns of w.
    #w=w/max(max(w));
    w = w / w.max(); #.max();

    #% w=w/sum(sum(w));

    #clear ab abn #Think if TODO!!!!

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("weights_for_ecc(): w.shape (4th) = %s" % \
                                                                str(w.shape));
        common.DebugPrint("weights_for_ecc(): w (4th) = %s" % str(w));

    return w;

#function out=block_zero_mean(in,grid_height,grid_width)
def block_zero_mean(inArray, grid_height, grid_width):
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("Entered block_zero_mean(): inArray.shape = %s" % \
                                                        str(inArray.shape));
        common.DebugPrint("block_zero_mean(): inArray[:10, :10] = %s" % \
                                                    str(inArray[:10, :10]));
        common.DebugPrint( \
            "block_zero_mean(): grid_height = %s, grid_width = %s" % \
                                        (str(grid_height), str(grid_width)));

    #inArray=double(inArray);
    inArray = inArray.astype(np.float64);

    #[m,n,s] = size(in);
    assert inArray.ndim == 2;
    #m, n, s = inArray.shape; # Gives exception: "ValueError: need more than 2 values to unpack"
    m, n = inArray.shape;
    s = 1;

    w0 = float(n) / grid_width;

    #h0 = m/grid_height;
    h0 = float(m) / grid_height;

    if s == 1:
        d = 1;
    else:
        d = (s + 1) / 2.0;

    M = np.zeros( (grid_height, grid_width) );

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("block_zero_mean(): M.shape (init) = %s" % \
                                                            str(M.shape));

    """
    We substitute i-1 with i AND j-1 with j, since in Python arrays indices
      start from 0, not 1 like in Matlab.
    """
    #for i=1:grid_height
    #for i in range(1, grid_height + 1):
    for i in range(grid_height):
        #for j=1:grid_width
        #for j in range(1, grid_width + 1):
        for j in range(grid_width):
            #M(i,j) = mean2(in(round(h0)*(i-1)+1:round(h0*i),round(w0)*(j-1)+1:round(w0*j),d));
            """
            M[i -1, j -1] = Matlab.mean2(inArray[round(h0) * (i - 1) + 1 - 1: round(h0 * i), \
                                            round(w0) * (j - 1) + 1 - 1: round(w0 * j), \
                                            d - 1]); # Gives: "IndexError: invalid index"
            """
            M[i, j] = Matlab.mean2(inArray[round(h0) * i + 1 - 1: round(h0 * (i + 1)), \
                                            round(w0) * j + 1 - 1: round(w0 * (j + 1))]);

    #common.DebugPrint("block_zero_mean(): M (before resize) = %s" % str(M));

    #M=imresize(M,[m n]);
    M = Matlab.imresize(M, newSize=(m, n));

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "block_zero_mean(): M.shape (after resize) = %s" % str(M.shape));
        common.DebugPrint( \
            "block_zero_mean(): M[:20, :20] (after resize) = %s" % \
                                                    str(M[:20, :20]));

    #M=M(:,:,ones(1,s));
    #M = M[:, :, np.ones( (1, s) )]; # Gives: "IndexError: arrays used as indices must be of integer (or boolean) type"
    """
    Note: In Matlab, the operand M(:,:,ones(1, s)) is the same as
        M(:, :) when M is a bidimensional matrix.
    """
    assert M.ndim == 2; # In this case we don't do a thing

    #out=in-M;
    out = inArray - M;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "block_zero_mean(): out.shape = %s" % str(out.shape));
        common.DebugPrint( \
            "block_zero_mean(): out[:20, :20] = %s" % str(out[:20, :20]));

    return out;


#function out=zm(in);
def zm(inArray):
    #out=in-mean(in(:));
    out = inArray - inArray.mean();
    return out;


#function VI_dW_dp = sd_images(dW_dp, nabla_Ix, nabla_Iy, nabla_It, N_p, h, w,time_extension_flag,pixel_select)
def sd_images(dW_dp, nabla_Ix, nabla_Iy, nabla_It, N_p, h, w, \
                                        time_extension_flag, pixel_select):
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "Entered sd_images(N_p=%s, time_extension_flag=%s, " \
                            "pixel_select=%s)" % \
                (str(N_p), str(time_extension_flag), str(pixel_select)));

    #if nargin<9 error('Not enough input arguments'); end

    #VI_dW_dp = np.zeros( (240, 2560) );
    # (nabla_Ix.shape[0], nabla_Ix.shape[1] * N_p * ...!!!!)

    # This matrix is HUGE - for HD resolution it has shape (1080, 15360) (for 240x320 it has (240, 2560))
    VI_dW_dp = np.zeros( (nabla_Ix.shape[0], nabla_Ix.shape[1] * N_p), dtype=MATRIX_FLOAT_TYPE );

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "sd_images(): VI_dW_dp.shape = %s" % str(VI_dW_dp.shape));
        common.DebugPrint( \
            "sd_images(): VI_dW_dp.dtype = %s" % str(VI_dW_dp.dtype));

    if pixel_select == 0:
        if time_extension_flag == 1: # I guess this branch is never taken
            #for p=1:N_p-1
            for p in range(1, N_p-1 + 1):
                #Tx = nabla_Ix .* dW_dp(1:h,((p-1)*w)+1:((p-1)*w)+w);
                Tx = nabla_Ix * dW_dp[0: h, ((p-1) * w): ((p-1) * w) + w];

                #Ty = nabla_Iy .* dW_dp(h+1:2*h,((p-1)*w)+1:((p-1)*w)+w);
                Ty = nabla_Iy * dW_dp[h: 2*h, ((p - 1) * w) : ((p - 1) * w) + w];

                #VI_dW_dp(:,((p-1)*w)+1:((p-1)*w)+w) = Tx + Ty;
                VI_dW_dp[:, ((p - 1) * w): ((p - 1) * w) + w] = Tx + Ty;

            #VI_dW_dp=[VI_dW_dp nabla_It];
            VI_dW_dp = np.c_[VI_dW_dp, nabla_It];
        else: # time_extension_flag == 0 # It seems this branch is taken often.
            #for p=1:N_p
            """
            We substitute p - 1 with p, since in Python arrays indices start from 0,
             not 1 like in Matlab.
            """
            #for p in range(1, N_p + 1):
            for p in range(N_p):
                #Tx = nabla_Ix .* dW_dp(1:h,((p-1)*w)+1:((p-1)*w)+w);
                Tx = nabla_Ix * dW_dp[0: h, (p * w): (p * w) + w];

                #Ty = nabla_Iy .* dW_dp(h+1:end,((p-1)*w)+1:((p-1)*w)+w);
                Ty = nabla_Iy * dW_dp[h:, (p * w): (p * w) + w];

                #VI_dW_dp(:,((p-1)*w)+1:((p-1)*w)+w) = Tx + Ty;
                VI_dW_dp[:, (p * w): (p * w) + w] = Tx + Ty;
    else:
        #hh=size(dW_dp,1);
        hh = dW_dp.shape[0];

        #%
        #%     size(dW_dp)
        #%     size(nabla_Ix)
        #%     N_p-1
        #%     size(dW_dp(1:hh/2,:))
        #%
        if time_extension_flag == 1:
            #Gx=repmat(nabla_Ix,1,N_p-1);
            Gx = np.tile(nabla_Ix, (1, N_p - 1) );

            #Gy=repmat(nabla_Iy,1,N_p-1);
            Gy = np.tile(nabla_Iy, (1, N_p - 1));

            #G=Gx.*dW_dp(1:hh/2,:)+Gy.*dW_dp(hh/2+1:end,:);
            G = Gx * dW_dp[0: hh/2, :] + Gy * dW_dp[hh/2: , :];

            #VI_dW_dp=[G nabla_It];
            VI_dW_dp = np.c_[G, nabla_It];
        else:
            #Gx=repmat(nabla_Ix,1,N_p);
            Gx = np.tile(nabla_Ix, (1, N_p) );

            #Gy=repmat(nabla_Iy,1,N_p);
            Gy = np.tile(nabla_Iy, (1, N_p) );

            #G=Gx.*dW_dp(1:hh/2,:)+Gy.*dW_dp(hh/2+1:end,:);
            G = Gx * dW_dp[0: hh/2, :] + Gy * dW_dp[hh/2: , :];

            #VI_dW_dp=[G];
            VI_dW_dp = G;

    return VI_dW_dp;


############################# END HELPER FUNCTIONS ############################
############################# END HELPER FUNCTIONS ############################
############################# END HELPER FUNCTIONS ############################
############################# END HELPER FUNCTIONS ############################



# Note: r_capture is actually the cv2.VideoCapture object.
"""
function fit = ecc_homo_spacetime(img_index, tmplt_index, p_init, t0, n_iters, levels,...
    r_path, q_path, nof, time_flag, weighted_flag, pixel_select, mode, imformat, show)

NOTE:
img_index = reference frame,
tmplt_index = query frame.
"""
def ecc_homo_spacetime(img_index, tmplt_index, p_init, t0, n_iters, levels, \
                        r_capture, q_capture, nof, time_flag, weighted_flag, \
                        pixel_select, mode, imformat, save_image):
    if True: #config.TESTING_IDENTICAL_MATLAB: #!!!!TODO: check again if ok
        global captureR, captureQ
        captureR = r_capture;
        captureQ = q_capture;

    t1 = float(cv2.getTickCount());

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "Entered ecc_homo_spacetime(img_index=%s, tmplt_index=%s, p_init=%s, save_image=%s, " \
                            "levels=%d, weighted_flag=%s, pixel_select=%s)" % \
                            (str(img_index), str(tmplt_index), str(p_init), str(save_image), \
                             levels, str(weighted_flag), str(pixel_select)));

    #% this function is a modification of ECC alignment algorithm

    #if nargin<15; error('Not enough input arguments'); end

    o_path = "Videos/output/";
    if not os.path.exists(o_path):
        #!!!!TODO: can give expcetion since created in another thread - move to caller
        os.makedirs(o_path);

    #%wgrid=[15 15]; #% grid for weighted version
    wgrid = [11, 11];

    #% int_type='cubic';
    int_type = "linear";

    #if ~strcmp(imformat(1),'.')
    if imformat[0] != ".":
        #imformat = ['.' imformat];
        imformat = "." + imformat;


    ####################START DECLARATION VARIABLES############################
    ####################START DECLARATION VARIABLES############################
    ####################START DECLARATION VARIABLES############################
    ####################START DECLARATION VARIABLES############################
    """
    Note: we index volume in Python the same as in Matlab, since in Matlab
        maybe we reach volume0 also - we have volume0, wout, volume2.
    """
    #volume = [None] * (2 + 1);
    volume = [None] * (levels + 1 + 1);

    """
    Note: we index tmplt in Python the same as in Matlab, since in Matlab
        maybe we reach tmplt0 also - we have tmplt0, tmplt1, tmplt2.
    """
    #tmplt = [None] * (2 + 1);
    tmplt = [None] * (levels + 1 + 1);


    """
    THIS CODE IS JUST FOR DOCUMENTATION PURPOSES - TODO: take it out
    if False:
        levels = 1
        iterECC = 15; #%iterations of ECC - is passed as n_iters parameter

        for nol=2:levels:
            for f in range(1, n_iters % round(n_iters / float(pow(2, (nol - 1))))):

    nol = 0: 15 / 2 ^ 0 = 15, 15 % 15 = 0
    nol = 1: 15 / 2 ^ 1 = 8, 15 % 8 = 7

    # weights warp_p rms_error rho factor t
    fit[nol][fiter] = FitElem(None, None, None, None, None, None);
    fit[nol][fiter].weights = W;
    """

    #fit = [[None] * 6] * 2;
    #fit = [[None] * 6] * n_iters;
    #fit = [[None] * n_iters] * 2;
    fit = [[None] * n_iters] * (levels + 1);
    """
    for fitRow in fit:
        for fitElem in fitRow:
            # NOT CORRECT since we cannot update fit through the iterator fitElem
            fitElem = FitElem(None, None, None, None, None, None);
    """
    for r in range(len(fit)):
        #print "a fitrow"
        for c in range(len(fit[r])):
            #print "a fitelem"
            fit[r][c] = FitElem(None, None, None, None, None, None);

    if common.MY_DEBUG_STDOUT:
        #common.DebugPrint("ecc_homo_spacetime(): fit = %s" % str(fit));
        for indexR, fitRow in enumerate(fit):
            for indexC, fitElem in enumerate(fitRow):
                common.DebugPrint("ecc_homo_spacetime(): fit[%d][%d] = %s" % \
                                            (indexR, indexC, str(fitElem)));
    #####################END DECLARATION VARIABLES#############################
    #####################END DECLARATION VARIABLES#############################
    #####################END DECLARATION VARIABLES#############################

    # We read the query/input frame numbered tmplt_index:
    #tmplt1=double(imread([q_path num2str(tmplt_index,'%.6d') imformat]));
    tmplt[1] = MyImageRead(q_capture, tmplt_index);

    #common.DebugPrint("ecc_homo_spacetime(): tmplt[1].shape (returned by MyImageRead) = %s" % \
    #                                        str(tmplt[1].shape));
    #common.DebugPrint("ecc_homo_spacetime(): tmplt[1].dtype (returned by MyImageRead) = %s" % \
    #                                        str(tmplt[1].dtype));
    #common.DebugPrint("ecc_homo_spacetime(): tmplt[1][:10, :10] = %s" % \
    #                    str(tmplt[1][:10, :10]));

    # We read the reference frame numbered img_index:
    #image_temp = imread([r_path num2str(img_index,'%.6d') imformat]);
    #!!!!!!!!TODO: make image_temp = volume1[1][nos / 2]
    image_temp = MyImageRead(r_capture, img_index);

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint( \
            "ecc_homo_spacetime(): image_temp.shape (returned by MyImageRead) = %s" % \
                                                    str(image_temp.shape));
        common.DebugPrint( \
            "ecc_homo_spacetime(): image_temp.dtype (returned by MyImageRead) = %s" % \
                                                    str(image_temp.dtype));

    #volume1=short_time_seq(r_path, img_index, nof, [size(image_temp,1), size(image_temp,2)], imformat);
    volume[1] = short_time_seq(r_capture, img_index, nof, \
                                [image_temp.shape[0], image_temp.shape[1]], \
                                imformat);

    #clear image_temp;
    #!!!!TODO: do eventually del image_temp;

    #% tmplt1=imgaussian(tmplt1,.5);
    #% volume1=imgaussian(volume1,.5);

    #% tmplt1=imresize(tmplt1,.5);
    #% volume1=imresize(volume1,.5);
    #imres=size(tmplt1);
    imres = tmplt[1].shape;


    #!!!!TODO: think well
    xxx = None;
    # Alex: we preallocate xxx with the max dimension required for RGB image.
    """
    if levels == 1:
        xxx = np.zeros( (imres[0], imres[1], 3) );
    else:
        xxx = np.zeros( (imres[0], imres[1], 3) );
        #xxx = np.zeros( (imres[0] / pow(2, levels-1),
        #                    imres[1] / pow(2, levels-1), 3) );
    """

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("ecc_homo_spacetime(): volume = %s" % str(volume));
        common.DebugPrint("ecc_homo_spacetime(): volume[1].shape = %s" % \
                                str(volume[1].shape));
        common.DebugPrint("ecc_homo_spacetime(): volume[1].dtype = %s" % \
                                str(volume[1].dtype));
        common.DebugPrint("ecc_homo_spacetime(): tmplt[1].shape = %s" % \
                                str(tmplt[1].shape));
        common.DebugPrint("ecc_homo_spacetime(): tmplt[1].dtype = %s" % \
                                str(tmplt[1].dtype));

    #for nol=2:levels
    # IMPORTANT NOTE: We substitute nol - 1 --> nol (since array numbering
    #     starts with 0, not like in Matlab from 1)
    for nol in range(1, levels-1 + 1):
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("ecc_homo_spacetime(): nol = %s" % str(nol));

        #eval(['volume' num2str(nol-1) '=imgaussian(volume' num2str(nol-1) ',.5);'])
        #volume[nol-1] = imgaussian(volume[nol-1], 0.5);
        for i in range(volume[nol].shape[2]):
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("     i = %s" % str(i));

            volume[nol][:, :, i] = imgaussian(volume[nol][:, :, i], 0.5);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("     volume[nol].dtype = %s" % \
                                str(volume[nol].dtype));

        #eval(['volume' num2str(nol) '=imresize(volume' num2str(nol-1) ',.5);'])
        #volume[nol] = Matlab.imresize(volume[nol-1], scale=0.5);
        volume[nol + 1] = np.empty((volume[nol].shape[0] / 2,
                                volume[nol].shape[1] / 2,
                                volume[nol].shape[2]), dtype=MATRIX_FLOAT_TYPE);
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("     volume[nol + 1].dtype = %s" % \
                                str(volume[nol + 1].dtype));

        for i in range(volume[nol].shape[2]):
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("     i = %s" % str(i));
            volume[nol + 1][:, :, i] = Matlab.imresize(volume[nol][:, :, i], scale=0.5);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint( \
                "     volume[nol + 1].dtype (after imresize) = %s" % \
                                str(volume[nol + 1].dtype));

        #eval(['tmplt' num2str(nol-1) '=imgaussian(tmplt' num2str(nol-1) ',.5);'])
        """
        Note: we index tmplt in Python the same as in Matlab since in
            Matlab we reach tmplt0 also.
        """
        tmplt[nol] = imgaussian(tmplt[nol], 0.5);
        if False:
            for i in range(tmplt[nol].shape[2]):
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("     i = %s" % str(i));
                tmplt[nol][:, :, i] = imgaussian(tmplt[nol][:, :, i], 0.5);

        #eval(['tmplt' num2str(nol) '=imresize(tmplt' num2str(nol-1) ',.5);'])
        tmplt[nol + 1] = Matlab.imresize(tmplt[nol], scale=0.5);
        if False:
            tmplt[nol + 1] = np.empty((tmplt[nol].shape[0] / 2,
                                tmplt[nol].shape[1] / 2,
                                tmplt[nol].shape[2]));
            for i in range(tmplt[nol].shape[2]):
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("     i = %s" % str(i));
                tmplt[nol + 1][:, :, i] = Matlab.imresize(tmplt[nol][:, :, i], scale=0.5);


    """
    VERY IMPORTANT: this branch actually doesn't execute anything since
        pixel_select == 0.
    """
    if pixel_select == 1:
        #RD=dir([r_path 'multiharlocs*.mat']); # DOESN'T seem to be used

        #str1=[q_path 'multiharlocs_' num2str(tmplt_index,'%.6d') '.mat'];
        #load(str1)
        # Retrieve the query harloc (Harris features)
        har1pp = q_path[tmplt_index];

        #str2=[r_path 'multiharlocs_' num2str(img_index,'%.6d') '.mat'];
        #har2=load(str2);
        # Retrieve the reference harloc (Harris features)
        har2pp = r_path[img_index];

        #har2.pp(:,2)=har2.pp(:,2)-2*p_init(7);
        har2pp[:, 1] = har2pp[:, 1] - 2 * p_init[6];

        if levels == 1:
            #out1 = my_morph(pp(:,1:2),15,imres(1),imres(2));
            out1 = my_morph(har1pp[:, 0: 2], 15, imres[0], imres[1]);

            #out2 = my_morph([har2.pp(:,1) har2.pp(:,2)],15,imres(1),imres(2));
            out2 = my_morph(np.c_[har2pp[:, 0], har2pp[:, 1]], \
                            15, imres[0], imres[1]);

            #out=out1.*out2;
            out = out1 * out2;

            #%        out=imresize(out,.5)>0; %handle half-size images

            """
            numpy.nonzero(a)
                Return the indices of the elements that are non-zero.
                Returns a tuple of arrays, one for each dimension of 'a', containing
                the indices of the non-zero elements in that dimension.

            From http://www.mathworks.com/help/matlab/ref/find.html:
                [row,col] = find(X, ...) returns the row and column indices
                    of the nonzero entries in the matrix X.
                This syntax is especially useful when working with sparse
                    matrices.
                If X is an N-dimensional array with N > 2, col contains linear
                    indices for the columns.
                For example, for a 5-by-7-by-3 array X with a nonzero element
                    at X(4,2,3), find returns 4 in row and 16 in col.
                That is, (7 columns in page 1) + (7 columns in page 2) +
                                (2 columns in page 3) = 16.
            """
            #[ny,nx]=find(out==1);
            if ORDER_MATTERS:
                nx, ny = np.nonzero((out == 1).T);
            else:
                ny, nx = np.nonzero(out == 1);

    """
    This is important: if we don't copy the matrix, warp_p would referece the
        same numpy.array object as p_init and since warp_p is changed,
        p_init is changed as well, and at the next call of ecc_homo_spacetime()
        p_init will have this last value of warp_p before exiting ecc_homo_spacetime();
    """
    warp_p = p_init.copy();

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("ecc_homo_spacetime(): warp_p = %s" % str(warp_p));

    # This loop actually doesn't execute anything WHEN levels == 1.
    #for ii=1:levels-1:
    for ii in range(1, levels-1 + 1):
        warp_p = next_level(warp_p, "homography", 0);

    if weighted_flag == 1:
        #W=ones(size(tmplt1));
        assert tmplt[1].ndim == 2;
        #W = np.ones(tmplt[1].shape);
        W = np.ones(tmplt[1].shape, dtype=MATRIX_FLOAT_TYPE);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("ecc_homo_spacetime(): At init, W.shape = %s" % \
                                                                str(W.shape));

    if time_flag == 1:
        N_p = 9;
        t = t0;
    else:
        N_p = 8;
        t = 0;

    if USE_DRIVER == True:
        volumeA = volume[1];
        tmpltA = tmplt[1]; # input (query) frame

        """
        cv2.imwrite("template_query_frame" + imformat, tmpltA);
        cv2.imwrite("reference_frame" + imformat, volumeA);
        """
        cv2.imwrite(o_path + ("reference_frame") + imformat, \
                    volumeA[:, :, 0].astype(int));
        cv2.imwrite(o_path + ("query_frame") + imformat, \
                    tmpltA.astype(int));

        """
        templateImage=tmplt[1],
        inputImage=image_temp,
        warpMatrix=warp_p,
        """
        return fit;

    if config.USE_ECC_FROM_OPENCV:
        levels = 0; # We assign 0 to avoid executing the standard space-time ECC below
        weighted_flag = 0;

        #!!!!!!!TODO: volumeA = volume[1][:, :, 2];
        """
        volumeA is the sub-sequence of reference frames (a number of nof frames) used
            in interp_space_time(), for warping of the reference "frame".
        """
        volumeA = volume[1];

        tmpltA = tmplt[1]; # input (query) frame

        if save_image == 1:
            #clear xxx
            #TODO: think if we should do a del xxx, like he does in Matlab a clear

            # We allocate space for xxx
            xxx = np.zeros( (tmpltA.shape[0], tmpltA.shape[1], 3) ); #!!!!TODO: use MATRIX_FLOAT_TYPE

            #xxx(:,:,1)=tmplt;
            xxx[:, :, 0] = tmpltA;

            #xxx(:,:,3)=tmplt;
            xxx[:, :, 2] = tmpltA;

            if config.VISUAL_DIFF_FRAMES == True:
                xxx[:, :, 1] = tmpltA;

        #!!!!TODO: experiment if it helps to "reuse" warp_p between different pairs of frames, MAYBE reset it to eye(3) once in a while - for the sake of performance increase
        #warp_p = p_init;
        warp_p = p_init.copy();
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint( \
                "ecc_homo_spacetime(): warp_p.dtype = %s" % str(warp_p.dtype));
            common.DebugPrint( \
                "ecc_homo_spacetime(): warp_p (before transformECC) = %s" % \
                                                                str(warp_p));

        if False:
            cv2.imwrite(o_path + ("reference_frame") + imformat, \
                    volumeA[:, :, 0].astype(int));
            cv2.imwrite(o_path + ("query_frame") + imformat, \
                    tmpltA.astype(int));

        """
        From http://opencvpython.blogspot.ro/2013/01/k-means-clustering-3-working-with-opencv.html
        Define criteria = ( type, max_iter = ... , epsilon = ...)
           Note: OpenCV's ECC uses by default: 50, 0.001 .
        #aCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001);

        See examples of criteria definition also at:
            http://docs.opencv.org/trunk/doc/py_tutorials/py_video/py_meanshift/py_meanshift.html
            http://opencvpython.blogspot.ro/2013/01/k-means-clustering-3-working-with-opencv.html
            - book pg 258, http://books.google.ro/books?id=seAgiOfu2EIC&pg=PA258&lpg=PA258&dq=OpenCV+CvTermCriteria&source=bl&ots=hTD0bmeANg&sig=eS7FA1QeEy_K5vAFpG_tCOjak7w&hl=en&sa=X&ei=DJN8U5XnOvTrygP5mYH4Aw&ved=0CEMQ6AEwAg#v=onepage&q=OpenCV%20CvTermCriteria&f=false
            - http://stackoverflow.com/questions/18955760/how-does-cvtermcriteria-work-in-opencv
              cv::TermCriteria(cv::TermCriteria::MAX_ITER +
                            cv::TermCriteria::EPS,
                            50, // max number of iterations
                            0.0001)); // min accuracy
        """
        #aCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 15, 0.001);
        aCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, config.EPS_ECC);

        """
        See doc findTransformECC at
          http://docs.opencv.org/trunk/modules/video/doc/motion_analysis_and_object_tracking.html#findtransformecc
            http://docs.opencv.org/trunk/modules/video/doc/motion_analysis_and_object_tracking.html#double%20findTransformECC%28InputArray%20templateImage,%20InputArray%20inputImage,%20InputOutputArray%20warpMatrix,%20int%20motionType,%20TermCriteria%20criteria%29
        Src code at https://github.com/Itseez/opencv/blob/ef91d7e8830c36785f0b6fdbf2045da48413dd76/modules/video/src/ecc.cpp (see also modules/video/include/opencv2/video/tracking.hpp and https://github.com/Itseez/opencv/blob/master/samples/cpp/image_alignment.cpp)
        """
        tECC1 = float(cv2.getTickCount());

        USE_MY_ECC_PY = False;

        if USE_MY_ECC_PY:
            import ECC
            import cv
            # From http://stackoverflow.com/questions/9913392/convert-numpy-array-to-cvmat-cv2
            cvTmplt = cv.fromarray(tmplt[1]);
            cvImageTemp = cv.fromarray(image_temp);
            cvWarp = cv.fromarray(warp_p);

            print "cvTmplt = %s" % str(cvTmplt);
            #print "dir(cvTmplt) = %s" % str(dir(cvTmplt));
            print "cvImageTemp = %s" % str(cvImageTemp);

            warp_p, retval = ECC.cvFindTransform(cvTmplt,
                                                 cvImageTemp,
                                                 cvWarp,
                                                 ECC.WARP_MODE_HOMOGRAPHY,
                                                 (cv.CV_TERMCRIT_ITER + cv.CV_TERMCRIT_EPS,
                                                 15, 0.001));
            warp_p = Matlab.ConvertCvMatToNPArray(warp_p);
        else:
            retval, warp_p = cv2.findTransformECC( \
                                            templateImage=tmplt[1],
                                            inputImage=image_temp,
                                            warpMatrix=warp_p,
                                            motionType=cv2.MOTION_HOMOGRAPHY,
                                            criteria=aCriteria);
        tECC2 = float(cv2.getTickCount());
        myTime = (tECC2 - tECC1) / cv2.getTickFrequency();
        print("ecc_homo_spacetime(): cv2.findTransformECC() took %.6f [sec]" % myTime);

        """
        From http://docs.opencv.org/trunk/modules/video/doc/motion_analysis_and_object_tracking.html#findtransformecc:
         "It returns the final enhanced correlation coefficient, that is the
           correlation coefficient between the template image and the final
           warped input image."
        """
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint( \
                "ecc_homo_spacetime(): retval (final value of ECC) of findTransformECC = %s" % \
                                                                str(retval));
            common.DebugPrint( \
                "ecc_homo_spacetime(): warp_p (after findTransformECC) = %s" % \
                                                                str(warp_p));

        ##IMPORTANT NOTE: we don't execute the nol loop because now levels == 0
        #   (see above)
        ##IMPORTANT NOTE: we don't execute the nol loop because now levels == 0
        ##IMPORTANT NOTE: we don't execute the nol loop because now levels == 0
        ##IMPORTANT NOTE: we don't execute the nol loop because now levels == 0
        ##IMPORTANT NOTE: we don't execute the nol loop because now levels == 0
        ##IMPORTANT NOTE: we don't execute the nol loop because now levels == 0
        ##IMPORTANT NOTE: we don't execute the nol loop because now levels == 0

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("ecc_homo_spacetime(): levels = %s" % str(levels));

    """
    resize volume[*];
    resize tmplt[nol + 1]; # input (query) frame
        for i in range(volume[nol].shape[2]):
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("     i = %s" % str(i));
            volume[nol + 1][:, :, i] = Matlab.imresize(volume[nol][:, :, i], scale=0.5);
    """

    #%%Iterative procedure
    #for nol=levels:-1:1
    # IMPORTANT NOTE: We substitute nol - 1 --> nol (since array numbering
    #     starts with 0, not like in Matlab from 1)
    for nol in range(levels - 1, 0-1, -1): # If levels == 1, it gets executed only once.
        #eval(['volume=volume' num2str(nol) ';'])
        volumeA = volume[nol + 1]; # (volumeA = volume[1], for our setting, when levels=1).
        # IMPORTANT NOTE: volumeA is a 3D matrix - it has nof matrices from the video sequence (in gray)

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint( \
                "ecc_homo_spacetime(): Am here: volumeA.shape = %s" % \
                                                        str(volumeA.shape));
            common.DebugPrint( \
                "ecc_homo_spacetime(): Am here: volumeA.dtype = %s" % \
                                                        str(volumeA.dtype));

        if INVERSE == False:
            # Note: nof is the number of frames for sub-sequences
            if nof > 1:
                #!!!!TODO: we should be able to optimize heavily here if we compute only the required elements of vx, vy, vt, since these matrices are huge and very easy to compute
                #[vx,vy,vt]=gradient(volume);
                vx, vy, vt = Matlab.gradient(volumeA);
            else:
                #[vx,vy]=gradient(volume);
                #!!!!TODO: we should be able to optimize heavily here if we compute only the required elements of vx, vy since these matrices are huge and very easy to compute
                vx, vy = Matlab.gradient(volumeA);
                vt = 0 * vx;

        #eval(['tmplt=tmplt' num2str(nol) ';'])
        tmpltA = tmplt[nol + 1];

        if INVERSE == True:
            print("IMPORTANT: ecc_homo_spacetime(): volumeA.shape = %s" % str(volumeA.shape));
            print("IMPORTANT: ecc_homo_spacetime(): tmpltA.shape = %s" % str(tmpltA.shape));

            """
            From iat_eccIC.m:
            temp = TEMP{nol};

            [vx,vy]=gradient(temp);
            """
            #vx, vy, vt = Matlab.gradient(tmpltA);
            #!!!!TODO TODO TODO: see what I can do better
            vx, vy = Matlab.gradient(tmpltA);
            vt = 0 * vx;

        #[AA,BB,CC]=size(volume);
        AA, BB, CC = volumeA.shape;

        #%     if nol>1
        #%         margin=floor(mean(AA,BB)*.05/(2^(nol-1)));
        #%     else
        margin = 0;
        #%     end

        if save_image == 1:
            #clear xxx
            #TODO: think if we should do a del xxx, like he does in Matlab a clear

            # We allocate space for xxx
            xxx = np.zeros( (tmpltA.shape[0], tmpltA.shape[1], 3) );

            #xxx(:,:,1)=tmplt;
            xxx[:, :, 0] = tmpltA;

            #xxx(:,:,3)=tmplt;
            xxx[:, :, 2] = tmpltA;

            if config.VISUAL_DIFF_FRAMES == True:
                xxx[:, :, 1] = tmpltA;

        if pixel_select == 0:
            # Note nx and ny are index vectors to tmpltA .

            #nx=margin+1:size(tmplt,2)-margin;
            nx = np.array(range(margin, tmpltA.shape[1] - margin));

            #ny=margin+1:size(tmplt,1)-margin;
            ny = np.array(range(margin, tmpltA.shape[0] - margin));

            if common.MY_DEBUG_STDOUT:
                #common.DebugPrint("ecc_homo_spacetime(): tmpltA = %s" % \
                #                                               str(tmpltA));
                common.DebugPrint( \
                    "ecc_homo_spacetime(): tmpltA.shape = %s" % \
                                                str(tmpltA.shape));
                common.DebugPrint( \
                    "ecc_homo_spacetime(): tmpltA.dtype = %s" % \
                                                str(tmpltA.dtype));

                common.DebugPrint("ecc_homo_spacetime(): nx = %s" % str(nx));
                common.DebugPrint("ecc_homo_spacetime(): nx.shape = %s" % \
                                                                str(nx.shape));

                common.DebugPrint("ecc_homo_spacetime(): ny = %s" % str(ny));
                common.DebugPrint("ecc_homo_spacetime(): ny.shape = %s" % \
                                                                str(ny.shape));

            #tmplt=double(tmplt(ny,nx,:));
            #tmpltA = tmpltA[ny, nx, :].astype(np.float64);
            #   Gave exception: "ValueError: shape mismatch: objects cannot be broadcast to a single shape"
            assert tmpltA.ndim == 2;
            assert tmpltA.shape[0] == ny.size;
            assert tmpltA.shape[1] == nx.size;
            #tmpltA = tmpltA.astype(np.float64);
            tmpltA = tmpltA.astype(MATRIX_FLOAT_TYPE);

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): tmpltA.shape (after) = %s" % \
                                                        str(tmpltA.shape));
        else:
            if levels > 1:
                #[hh,ww]=size(tmplt);
                hh, ww = tmpltA.shape;

                #pp1=pp(:,1:2)/2;
                pp1 = har1pp[:, 0: 2] / 2.0;

                #pp2=[har2.pp(:,1:2)]/2;
                pp2 = har2pp[:, 0: 2] / 2.0;

                #out1=my_morph(pp1./2^(nol-1),8-2*nol+1,hh,ww);
                out1 = my_morph(pp1 / pow(2.0, nol), 8 - 2 * (nol+1) + 1, \
                                hh, ww);

                #out2=my_morph(pp2(:,1:2)./2^(nol-1),8-2*nol+1,hh,ww);
                out2 = my_morph(pp2[:, 0: 2] / pow(2.0, nol), \
                                8 - 2 * (nol+1) + 1, hh, ww);

                #out=out1.*out2;
                out = out1 * out2;

                #[ny,nx]=find(out==1);
                if ORDER_MATTERS:
                    nx, ny = np.nonzero((out == 1).T);
                else:
                    ny, nx = np.nonzero(out == 1);

            #tmplt=double(tmplt(logical(out)));
            tmpltA = tmpltA[np.logical(out)].astype(np.float64);

        #h=numel(ny);
        h = ny.size;
        #w=numel(nx);
        w = nx.size;

        #if weighted_flag & (nol==1)
        if (weighted_flag == 1) and (nol+1 == 1):
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): nol = %s" % str(nol));

            #tmplt=block_zero_mean(tmplt,wgrid(1),wgrid(2));
            tmpltA = block_zero_mean(tmpltA, wgrid[0], wgrid[1]);

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): W.shape = %s" % str(W.shape));

            #W=imresize(W,size(tmplt));
            W = Matlab.imresize(W, newSize=tmpltA.shape);

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): tmpltA.shape = %s" % \
                                                str(tmpltA.shape));
        else:
            #if ~strcmp(mode,'nophoto')
            if mode != "nophoto":
                tmpltA = zm(tmpltA); #%zero-mean image for brigthness compensation

        init_tmplt = tmpltA;

        #n_tmplt=norm(tmplt(:));
        n_tmplt = npla.norm(tmpltA.ravel(order="F"));

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint( \
                "ecc_homo_spacetime(): n_tmplt = %s" % str(n_tmplt));


        #clear vtt
        #del vtt; #!!!!TODO: check if OK

        factor = 1.0;

        #alexTmp = n_iters #% int(round(n_iters / float(pow(2, (nol - 1)))));
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("ecc_homo_spacetime(): nol = %d" % (nol));
            #common.DebugPrint( \
            #        "ecc_homo_spacetime(): alexTmp = %d" % alexTmp);

        if INVERSE == True:
            if True:
                nabla_Ix = interp_space_time(vx, img_index, warp_p, t, int_type, nx, ny, pixel_select);
                nabla_Iy = interp_space_time(vy, img_index, warp_p, t, int_type, nx, ny, pixel_select);
                nabla_It = interp_space_time(vt, img_index, warp_p, t, int_type, nx, ny, pixel_select);

                #!!!!TODO TODO: compute W like below
                if False:
                    #if weighted_flag (nol==1)
                    assert weighted_flag == 1;
                    if (weighted_flag == 1) and (nol+1 == 1): # Evangelidis wrote a weird predicate in Matlab here - he omitted the && ...
                        #nabla_Ix=W.*nabla_Ix;
                        nabla_Ix = W * nabla_Ix;

                        #nabla_Iy=W.*nabla_Iy;
                        nabla_Iy = W * nabla_Iy;

                        #nabla_It=W.*nabla_It;
                        nabla_It = W * nabla_It;

                        if common.MY_DEBUG_STDOUT:
                            common.DebugPrint( \
                                "ecc_homo_spacetime(): nabla_Ix[:20,:20](after) = %s" % \
                                                                str(nabla_Ix[:20, :20]));
                            common.DebugPrint( \
                                "ecc_homo_spacetime(): nabla_Iy[:20,:20](after) = %s" % \
                                                                str(nabla_Iy[:20, :20]));
                            common.DebugPrint( \
                                "ecc_homo_spacetime(): nabla_It[:20,:20](after) = %s" % \
                                                                str(nabla_It[:20, :20]));



            """
            From iat_eccIC.m:
            % Pre-computations

            % Compute the jacobian of identity transform
            J = iat_warp_jacobian(nx, ny, eye(3), transform);

            % Compute the jacobian of warped image wrt parameters (matrix G in the paper)
            G = iat_image_jacobian(vx, vy, J, nop);

            % Compute Hessian and its inverse
            C= G' * G;% C: Hessian matrix
            """
            #% Evaluate Jacobian
            dW_dp = jacobian_h(nx, ny, np.eye(3), time_flag, pixel_select);

            # Alex: VI_dW_dp is the equivalent of G
            #% Compute steepest descent images, VI_dW_dp
            VI_dW_dp = sd_images(dW_dp, nabla_Ix, nabla_Iy, nabla_It, N_p, \
                                    h, w, time_flag, pixel_select);

            #%%% Compute Hessian and inverse
            C = hessian(VI_dW_dp, N_p, w);

        #% Forwards Additive space-time ECC
        #for f=1:n_iters %round(n_iters/(2^(nol-1)))
        """
        We substitute fiter - 1 with fiter, to easily support 0-indexed arrays
          in Python, instead of 1-indexed in Matlab.
        """
        for fiter in range(n_iters):
            #if fit[nol][fiter] == None:

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): fiter = %d (n_iters = %d)" % \
                                                            (fiter, n_iters));
                common.DebugPrint( \
                        "ecc_homo_spacetime(): volumeA.shape " \
                        "(before interp_space_time) = %s" % \
                                                        (str(volumeA.shape)));
                common.DebugPrint( \
                        "ecc_homo_spacetime(): volumeA.dtype " \
                        "(before interp_space_time) = %s" % \
                                                        (str(volumeA.dtype)));

            #% Compute warped image with current parameters
            #wimg = interp_space_time(volume, img_index, warp_p, t, int_type, nx, ny, pixel_select);
            wimg = interp_space_time(volumeA, img_index, warp_p, t, int_type, \
                                        nx, ny, pixel_select);

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                        "ecc_homo_spacetime(): wimg.shape " \
                        "(after interp_space_time) = %s" % \
                                                            (str(wimg.shape)));
                common.DebugPrint( \
                        "ecc_homo_spacetime(): wimg.dtype " \
                        "(after interp_space_time) = %s" % \
                                                            (str(wimg.dtype)));

            #nonze = wimg~=0;
            nonze = wimg != 0;

            IWxp = wimg;
            #%nzind=wimg~=0;

            #if weighted_flag && (nol==1)
            if (weighted_flag == 1) and (nol+1 == 1):
                #IWxp = block_zero_mean(wimg,wgrid(1),wgrid(2));
                IWxp = block_zero_mean(wimg, wgrid[0], wgrid[1]);

                W = weights_for_ecc(init_tmplt, IWxp, wgrid, nonze);

                #common.DebugPrint("ecc_homo_spacetime(): nol = %s" % str(nol));
                ##common.DebugPrint("fit[nol][fiter] = %s" % str(fit[nol][fiter]));

                #fit(nol,f).weights = W;
                fit[nol][fiter].weights = W;

                #common.DebugPrint("ecc_homo_spacetime(): fit[nol][fiter].weights.shape = %s" % \
                #                    str(fit[nol][fiter].weights.shape));
                #common.DebugPrint("ecc_homo_spacetime(): fit[nol][fiter].weights = %s" % \
                #                    str(fit[nol][fiter].weights));

                #W = imresize(W,size(tmplt),'nearest');
                W = Matlab.imresize(W, newSize=tmpltA.shape, \
                                    interpolationMethod=cv2.INTER_NEAREST);

                #%             imagesc(W);
                #%             colorbar
                #%             pause

                #W(W<0)=0;
                W[W < 0] = 0;

                #W=W.*nonze;
                W = W * nonze;

                #IWxp = W.*IWxp;
                IWxp = W * IWxp;

                #tmplt = W.*init_tmplt;
                tmpltA = W * init_tmplt;
            else:
                #if ~strcmp(mode,'nophoto')
                if mode != "nophoto":
                    #IWxp=zm(IWxp); %zero-mean image for brigthness compensation
                    IWxp = zm(IWxp); #%zero-mean image for brigthness compensation

            #% error_img = tmplt(:) - IWxp(:);
            error_img = init_tmplt - IWxp;

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("ecc_homo_spacetime(): W.shape = %s" % \
                                                    str(W.shape));
                common.DebugPrint("ecc_homo_spacetime(): W[:20, :20] = %s" % \
                                                    str(W[:20, :20]));

                common.DebugPrint( \
                    "ecc_homo_spacetime(): error_img.shape = %s" % \
                                                    str(error_img.shape));
                common.DebugPrint( \
                    "ecc_homo_spacetime(): error_img[:20, :20] = %s" % \
                                                    str(error_img[:20, :20]));

            #% -- Save current fit parameters --
            #fit(nol,f).warp_p=warp_p;
            fit[nol][fiter].warp_p = warp_p;

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): fit[nol][fiter].warp_p.shape = %s" % \
                                        str(fit[nol][fiter].warp_p.shape));
                common.DebugPrint( \
                    "ecc_homo_spacetime(): fit[nol][fiter].warp_p = %s" % \
                                        str(fit[nol][fiter].warp_p));

            #fit(nol,f).rms_error = sqrt(mean(error_img(:).^2));
            fit[nol][fiter].rms_error = \
                    math.sqrt(np.mean(np.ravel(error_img, order="F") ** 2));

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): fit[nol][fiter].rms_error = %s" % \
                                            str(fit[nol][fiter].rms_error));

            #fit(nol,f).rho=dot(tmplt(:),IWxp(:))/n_tmplt/norm(IWxp(:));
            fit[nol][fiter].rho = \
                np.dot(np.ravel(tmpltA, order="F"), np.ravel(IWxp, order="F")) / \
                n_tmplt / npla.norm(np.ravel(IWxp, order="F"));

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): fit[nol][fiter].rho = %s" % \
                                                    str(fit[nol][fiter].rho));

            #fit(nol,f).factor=factor;
            fit[nol][fiter].factor = factor;

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): fit[nol][fiter].factor = %s" % \
                                                str(fit[nol][fiter].factor));

            #%          [t fit(nol,f).rho nol]
            #% -- iteration 1 is the zeroth, ignore final computation --
            #if (abs(t)>ceil((CC-1)/2)):
            if abs(t) > math.ceil((CC - 1) / 2.0):
                #%       if (abs(t)>1)
                #disp('frame correction too big (more than half of temporal window)');
                #common.DebugPrint("frame correction too big (more than half of temporal window)");

                #t=sign(t)*abs(fix(t));
                t = sgn(t) * abs(fix[t - 1]);

            #fit(nol,f).t=t;
            fit[nol][fiter].t = t;

            #common.DebugPrint("ecc_homo_spacetime(): fit[nol][fiter].t = %s" % \
            #                                str(fit[nol][fiter].t));

            #config.EPS_ECC = 0.001; # This is the default EPS used for convergence in OpenCV 3.0
            #config.EPS_ECC = 0.0005;
            #config.EPS_ECC = 0.0001;
            #config.EPS_ECC = 0.00000001;
            #% if (f == n_iters)
            #if fiter == n_iters - 1:
            if (fiter >= 1) and (abs(fit[nol][fiter].rho - fit[nol][fiter - 1].rho) < config.EPS_ECC):
                print("ecc_homo_spacetime(): ECC for level %d converged at iteration %d" % (nol, fiter));
                break;
            if fiter == n_iters - 1:
                print("ecc_homo_spacetime(): ECC for level %d finished all iterations" % (nol));
                break;

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("ecc_homo_spacetime(): vx.shape = %s" % \
                                                            str(vx.shape));
                common.DebugPrint("ecc_homo_spacetime(): vx.dtype = %s" % \
                                                            str(vx.dtype));
                common.DebugPrint("ecc_homo_spacetime(): vy.shape = %s" % \
                                                            str(vy.shape));
                common.DebugPrint("ecc_homo_spacetime(): vy.dtype = %s" % \
                                                            str(vy.dtype));
                common.DebugPrint("ecc_homo_spacetime(): vt.shape = %s" % \
                                                            str(vt.shape));
                common.DebugPrint("ecc_homo_spacetime(): vt.dtype = %s" % \
                                                            str(vt.dtype));

            if INVERSE == False:
                #% Evaluate gradient
                #!!!!TODO: we should be able to optimize heavily here if we compute only the required elements of vx, vy, vt since these matrices are huge and very easy to compute
                nabla_Ix = interp_space_time(vx, img_index, warp_p, t, int_type, nx, ny, pixel_select);
                nabla_Iy = interp_space_time(vy, img_index, warp_p, t, int_type, nx, ny, pixel_select);
                nabla_It = interp_space_time(vt, img_index, warp_p, t, int_type, nx, ny, pixel_select);

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("ecc_homo_spacetime(): nabla_Ix.shape = %s" % \
                                                        str(nabla_Ix.shape));
                    common.DebugPrint("ecc_homo_spacetime(): nabla_Ix[:20, :20] = %s" % \
                                                            str(nabla_Ix[:20, :20]));
                    common.DebugPrint("ecc_homo_spacetime(): nabla_Iy.shape = %s" % \
                                                        str(nabla_Iy.shape));
                    common.DebugPrint("ecc_homo_spacetime(): nabla_Iy[:20, :20] = %s" % \
                                                            str(nabla_Iy[:20, :20]));
                    common.DebugPrint("ecc_homo_spacetime(): nabla_It.shape = %s" % \
                                                        str(nabla_It.shape));
                    common.DebugPrint("ecc_homo_spacetime(): nabla_It[:20, :20] = %s" % \
                                                            str(nabla_It[:20, :20]));

                #if weighted_flag (nol==1)
                assert weighted_flag == 1;
                if (weighted_flag == 1) and (nol+1 == 1): # Evangelidis wrote a weird predicate in Matlab here - he omitted the && ...
                    #nabla_Ix=W.*nabla_Ix;
                    nabla_Ix = W * nabla_Ix;

                    #nabla_Iy=W.*nabla_Iy;
                    nabla_Iy = W * nabla_Iy;

                    #nabla_It=W.*nabla_It;
                    nabla_It = W * nabla_It;

                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): nabla_Ix[:20,:20](after) = %s" % \
                                                            str(nabla_Ix[:20, :20]));
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): nabla_Iy[:20,:20](after) = %s" % \
                                                            str(nabla_Iy[:20, :20]));
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): nabla_It[:20,:20](after) = %s" % \
                                                            str(nabla_It[:20, :20]));

                #% Evaluate Jacobian
                dW_dp = jacobian_h(nx, ny, warp_p, time_flag, pixel_select);

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): dW_dp.shape = %s" % \
                                                            str(dW_dp.shape));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): dW_dp[:20,:20] = %s" % \
                                                            str(dW_dp[:20,:20]));

                #% Compute steepest descent images, VI_dW_dp
                VI_dW_dp = sd_images(dW_dp, nabla_Ix, nabla_Iy, nabla_It, N_p, \
                                        h, w, time_flag, pixel_select);

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): VI_dW_dp.shape = %s" % \
                                                            str(VI_dW_dp.shape));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): VI_dW_dp.dtype = %s" % \
                                                            str(VI_dW_dp.dtype));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): VI_dW_dp[:20,:20] = %s" % \
                                                            str(VI_dW_dp[:20,:20]));

            if pixel_select == 0:
                if INVERSE == False:
                    #%%% Compute Hessian and inverse
                    C = hessian(VI_dW_dp, N_p, w);

                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): C.shape = %s" % str(C.shape));
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): C.dtype = %s" % str(C.dtype));
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): C = %s" % str(C));

                    if False:
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): npla.det(C) = %s" % \
                                                                str(npla.det(C)));

                try:
                    #i_C = inv(C);
                    i_C = npla.inv(C);
                except npla.LinAlgError: #lae:
                    print( \
                        "ecc_homo_spacetime(): got a LinAlgError exception --> bailing out from function");
                    common.DebugPrintErrorTrace();

                    fit[0][config.iterECC - 1].t = -1;
                    return fit;

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): i_C.shape = %s" % \
                        str(i_C.shape));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): i_C.dtype = %s" % \
                        str(i_C.dtype));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): i_C = %s" % str(i_C));

                if INVERSE == False:
                    #%  sd_delta_p = sd_update(VI_dW_dp, error_img, N_p, w);
                    V = sd_update(VI_dW_dp, tmpltA, N_p, w);
                    b = sd_update(VI_dW_dp, IWxp, N_p, w);
                else:
                    #%  sd_delta_p = sd_update(VI_dW_dp, error_img, N_p, w);
                    b = sd_update(VI_dW_dp, tmpltA, N_p, w);
                    V = sd_update(VI_dW_dp, IWxp, N_p, w);

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): V.shape = %s" % str(V.shape));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): V.dtype = %s" % str(V.dtype));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): V = %s" % str(V));

                    common.DebugPrint( \
                        "ecc_homo_spacetime(): b.shape = %s" % str(b.shape));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): b.dtype = %s" % str(b.dtype));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): b = %s" % str(b));

                #%%% ECC closed form solution

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): IWxp.shape = %s" % \
                                                            str(IWxp.shape));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): IWxp[:20, :20] = %s" % \
                                                        str(IWxp[:20, :20]));

                #factor=(norm(IWxp(:))^2-b'*i_C*b);
                if False:
                    factor1 = pow(npla.norm(np.ravel(IWxp, order="F")), 2);
                    factor2 = np.dot(np.dot(b.T, i_C), b); # Matrix multiplication

                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): factor1 = %s" % str(factor1));
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): factor2 = %s" % str(factor2));
                    factor = factor1 - factor2;
                factor = pow(npla.norm(np.ravel(IWxp, order="F")), 2) - \
                            np.dot(np.dot(b.T, i_C), b); # Matrix multiplication

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): factor = %s" % str(factor));

                assert V.shape[1] > 0;
                assert i_C.shape[1] > 0;
                assert b.shape[1] > 0;
                #den=(dot(tmplt(:),IWxp(:))-V'*i_C*b);
                den = np.dot(np.ravel(tmpltA, order="F"), np.ravel(IWxp, order="F")) - \
                        np.dot(np.dot(V.T, i_C), b); # Matrix multiplication here

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): den.dtype = %s" % str(den.dtype));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): den = %s" % str(den));

                #if strcmp(mode,'ecc')
                if mode == "ecc":
                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): den = %s" % str(den));

                    #factor=factor/den;
                    #assert str(den.dtype) != "int32";
                    assert (den.dtype == np.float32) or (den.dtype == np.float64);
                    factor = factor / den;

                    #common.DebugPrint("ecc_homo_spacetime(): factor = %s" % str(factor));

                #if strcmp(mode,'lk') % %  lucas-kanade
                if mode == "lk": #% %  lucas-kanade
                    #factor=den/(norm(tmplt(:))^2-V' * i_C * V);
                    #assert den.dtype != np.int32;
                    assert (den.dtype == np.float32) or (den.dtype == np.float64);
                    factor = den / pow((npla.norm(np.ravel(tmpltA, order="F")), 2) - \
                                np.dot(np.dot(V.T, i_C), V)); # Matrix multiplication here # !!!!TODO: test if OK

                #if strcmp(mode,'nophoto')
                if mode == "nophoto":
                    factor = 1.0;

                #%%% Step size matrix
                ssm = np.eye(N_p);

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): factor.shape = %s" % \
                                str(factor.shape));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): tmpltA.shape = %s" % \
                                str(tmpltA.shape));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): IWxp.shape = %s" % \
                                str(IWxp.shape));

                #error_img=factor*tmplt-IWxp;
                error_img = factor * tmpltA - IWxp;

                sd_delta_p = sd_update(VI_dW_dp, error_img, N_p, w);
            else:
                #C=VI_dW_dp'*VI_dW_dp;
                C = VI_dW_dp.T * VI_dW_dp;

                #i_C=inv(C);
                i_C = npla.inv(C);

                #V=VI_dW_dp'*tmplt;
                V = VI_dW_dp.T * tmpltA;

                #b=VI_dW_dp'* IWxp;
                b = VI_dW_dp.T * IWxp;

                #factor=(norm(IWxp(:))^2-b'*i_C*b);
                factor = pow(npla.norm(np.ravel(IWxp)), 2) - \
                            b.T * i_C * b;

                #den=(dot(tmplt(:),IWxp(:))-V'*i_C*b);
                den = np.dot(np.ravel(tmpltA, order="F"), np.ravel(IWxp, order="F")) - \
                        np.dot(np.dot(V.T, i_C), b); # Matrix multiplication here #!!!!TODO: check

                factor = float(factor) / den;

                #%%% Step size matrix
                #ssm=eye(N_p);
                ssm = np.eye(N_p);

                #ssm(N_p,N_p)=1/5;
                ssm[N_p - 1, N_p - 1] = 1.0 / 5;

                #error_img=factor*tmplt-IWxp;
                error_img = factor * tmpltA - IWxp;

                #sd_delta_p=VI_dW_dp'*error_img;
                assert error_img.ndim == 2;
                sd_delta_p = np.dot(VI_dW_dp.T, error_img); # Matrix multiplication

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): ssm.shape = %s" % str(ssm.shape));

                #common.DebugPrint( \
                #    "ecc_homo_spacetime(): i_C.shape = %s" % str(i_C.shape));

                common.DebugPrint( \
                    "ecc_homo_spacetime(): sd_delta_p.shape = %s" % \
                                                    str(sd_delta_p.shape));
                common.DebugPrint( \
                    "ecc_homo_spacetime(): sd_delta_p = %s" % str(sd_delta_p));

            #delta_p=ssm*i_C*sd_delta_p;
            delta_p = np.dot(np.dot(ssm, i_C), sd_delta_p); # Matrix multiplication

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): delta_p.shape = %s" % \
                                                        str(delta_p.shape));
                common.DebugPrint( \
                    "ecc_homo_spacetime(): delta_p = %s" % str(delta_p));

            #% Update warp parmaters
            #[warp_p,t] = update_step(warp_p,t,delta_p,time_flag);
            warp_p, t = update_step(warp_p, t, delta_p, time_flag);

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): After update_step()");
                common.DebugPrint( \
                    "ecc_homo_spacetime(): warp_p.shape = %s" % \
                                                        str(warp_p.shape));
                common.DebugPrint("ecc_homo_spacetime(): warp_p = %s" % \
                                                        str(warp_p));
                common.DebugPrint("ecc_homo_spacetime(): t = %s" % str(t));

        # END of f loop
        # END of f loop
        # END of f loop
        # END of f loop
        # END of f loop
        # END of f loop

        #if nol>1 %& time_flag==1;
        if nol+1 > 1:
            #warp_p(7:8)=warp_p(7:8)*2;
            warp_p[0, 2] *= 2;
            warp_p[1, 2] *= 2;

            #warp_p(3)=warp_p(3)/2;
            warp_p[2, 0] /= 2.0;

            #warp_p(6)=warp_p(6)/2;
            warp_p[2, 1] /= 2.0;
    # END of nol for loop
    # END of nol for loop
    # END of nol for loop
    # END of nol for loop
    # END of nol for loop
    # END of nol for loop



    # Now we warp the matched reference frame to perform the spatial alignment
    # Now we warp the matched reference frame to perform the spatial alignment
    # Now we warp the matched reference frame to perform the spatial alignment
    # Now we warp the matched reference frame to perform the spatial alignment
    # Now we warp the matched reference frame to perform the spatial alignment
    if save_image == 1:
        #w=ones([imres nof]);
        w = np.ones( (imres[0], imres[1], nof), dtype=MATRIX_FLOAT_TYPE);

        #nx=1:imres(2);
        nx = np.array(range(1, imres[1] + 1));

        #ny=1:imres(1);
        ny = np.array(range(1, imres[0] + 1));

        if weighted_flag == 1:
            #w=fit(nol,f).weights;
            w = fit[nol][fiter].weights; #!!!!TODO: use np.float32

            #w=imresize(w,imres);
            w = Matlab.imresize(w, newSize=imres);

            #%figure(1);imagesc(w); title('weight map');
            #w=w(:,:,ones(3,1));
            """
            # Gives: "IndexError: arrays used as indices must be of integer (or boolean) type"
            w = w[:, :, np.ones( (3,1) )];
            """
            assert w.ndim == 2;

        t11 = float(cv2.getTickCount());

        # Alex: we warp the reference frame (s.t. we align it to the input frame)
        #wout = interp_space_time(volume, img_index, warp_p, t, int_type, nx, ny, pixel_select);
        #if True:
        if False: #config.USE_ECC_FROM_OPENCV:
            refFrame = volumeA[:, :, 0]; #nof / 2]; #.T;

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): refFrame.shape = %s" % \
                                                    str(refFrame.shape));

            #!!!!TODO_PROFOUND: understand why we get poorer results with warpPerspective than with interp_space_time - see /home/asusu/drone-diff_test_against_EV_videos/Videos/output (*_good_new/new0.png) VS *_good.png
            wout = cv2.warpPerspective(src=refFrame, M=warp_p, \
                                dsize=(refFrame.shape[1], refFrame.shape[0]));
            """
            If you look at http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#void%20warpAffine%28InputArray%20src,%20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int%20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue%29
              you will see that M should be a 2*3 matrix.
                wout = cv2.warpAffine(src=refFrame, M=warp_p, \
                                dsize=(refFrame.shape[1], refFrame.shape[0]));
            """
            #wout = wout.T;
        else:
            wout = interp_space_time(volumeA, img_index, warp_p, t, int_type, \
                                                        nx, ny, pixel_select);

        t12 = float(cv2.getTickCount());
        myTime1 = (t12 - t11) / cv2.getTickFrequency();
        print("ecc_homo_spacetime(): warping took %.6f [sec]" % myTime1);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("ecc_homo_spacetime(): wout.shape = %s" % \
                                                            str(wout.shape));
            common.DebugPrint("ecc_homo_spacetime(): wout[:20] = %s" % \
                                                            str(wout[:20]));

            common.DebugPrint("ecc_homo_spacetime(): xxx[:, :, :20] = %s" % \
                                                        str(xxx[:, :, :20]));

            common.DebugPrint( \
                "ecc_homo_spacetime(): VISUAL_DIFF_FRAMES = %s" % \
                                            str(config.VISUAL_DIFF_FRAMES));
            common.DebugPrint( \
                "ecc_homo_spacetime(): (again0) VISUAL_DIFF_FRAMES = %s" % \
                                            str(config.VISUAL_DIFF_FRAMES));

        #xxx(:,:,2)=wout;
        if config.VISUAL_DIFF_FRAMES == False:
            xxx[:, :, 1] = wout; # This is the original behavior

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("ecc_homo_spacetime(): xxx.shape = %s" % \
                                                        str(xxx.shape));

        ############################# DIFF FRAMES ##############################
        ############################# DIFF FRAMES ##############################
        ############################# DIFF FRAMES ##############################
        ############################# DIFF FRAMES ##############################
        ############################# DIFF FRAMES ##############################
        ############################# DIFF FRAMES ##############################
        if config.VISUAL_DIFF_FRAMES == True:
            # But first we save the standard (as Evangelidis does) video alignment result with the 2 images overlapped
            # We have xxx[:, :, 1] = tmpltA;
            xxx1Orig = xxx[:, :, 1]; # Note: copy() does not help here

            # We should have:
            xxx[:, :, 1] = wout; # This is the original behavior

            #%figure(2); imshow(uint8(xxx)); title('spatio-temporal alignment');
            #imwrite(uint8(xxx),[o_path num2str(tmplt_index,'%.6d_good') imformat]);
            cv2.imwrite(o_path + ("%.6d_good" % tmplt_index) + imformat, \
                                                     xxx.astype(int));

            # Now reverting:
            xxx[:, :, 1] = xxx1Orig;


            tdiff1 = float(cv2.getTickCount());

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): (again) VISUAL_DIFF_FRAMES = %s" % \
                                            str(config.VISUAL_DIFF_FRAMES));
            """
            # Compute difference between frames inputFrame and refFrame:
            cv2.absdiff(src1=inputFrame, src2=refFrame, dst=imgDiff)
            """
            #!!!!TODO TODO TODO: use for xxx only 1 color channel (i.e., make xxx 2d matrix)
            xxx[:, :, 0] -= wout;
            xxx[:, :, 1] -= wout;
            xxx[:, :, 2] -= wout;

            #!!!!TODO_PROFOUND: what type/size of filter is OPTIMAL for the videos in discussion?
            # !!!!TODO_PROFOUND: also what cluster size threhsolds are optimal?
            for i in range(xxx.shape[2]):
                xxx[:, :, i] = cv2.GaussianBlur(src=xxx[:, :, i], \
                                                ksize=(7, 7), \
                                                sigmaX=10);
            #common.DebugPrint("ecc_homo_spacetime(): (again2) xxx[:, :, :20] = %s" % \
            #                                str(xxx[:, :, :20]));
            #common.DebugPrint("ecc_homo_spacetime(): (again2) VISUAL_DIFF_FRAMES = %s" % \
            #                                str(config.VISUAL_DIFF_FRAMES));

            DRAW_WARPED_BORDERS = False;
            if DRAW_WARPED_BORDERS == True:
                # See for Point and info on perspective transformation: we use (x, y) (x first) - http://docs.opencv.org/modules/core/doc/basic_structures.html#point
                tl = np.array([0, 0, 1]);
                tr = np.array([xxx.shape[1], 0, 1]);
                bl = np.array([0, xxx.shape[0], 1]);
                br = np.array([xxx.shape[1], xxx.shape[0], 1]);
                tlF = np.dot(warp_p, tl);
                trF = np.dot(warp_p, tr);
                blF = np.dot(warp_p, bl);
                brF = np.dot(warp_p, br);

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): warp_p = %s" % str(warp_p));

                    common.DebugPrint( \
                            "ecc_homo_spacetime(): tl = %s" % str(tl));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): tr = %s" % str(tr));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): bl = %s" % str(bl));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): br = %s" % str(br));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): tlF = %s" % str(tlF));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): trF = %s" % str(trF));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): blF = %s" % str(blF));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): brF = %s" % str(brF));

                def HomogenizeAndCarthesianize(p):
                    p /= p[2]; # We homogenize

                    # We transform in (integer) Carthesian coordinates
                    p = (int(p[0]), int(p[1]));

                    return p;

                myPts = [tlF, trF, brF, blF];
                myPts = map(HomogenizeAndCarthesianize, myPts);

                """
                !!!!TODO_PROFOUND: when we use interp_space_time(), the computed warped
                    border is not perfectly matching the warped reference image.
                Also I DO NOT understand why interp_space_time() does a
                    different warping than cv2.warpPerspective(), which seems to be
                    in accordance with my computed warped border.

                !!!!TODO_PROFOUND: Understand what interp_space_time() really does - it seems it aligns
                a bit better than warpPerspective(), even if we compute warp_p
                with OpenCV's ECC.
                """

                if False:
                    #myPts = np.array([[tlF], [trF], [brF], [blF]]); # Inspired from http://stackoverflow.com/questions/8369547/checking-contour-area-in-opencv-using-python
                    #myPts = np.array([tlF, trF, brF, blF]); # Gives: "error: /home/asusu/opencv/opencv-master/modules/core/src/drawing.cpp:2152: error: (-215) npoints > 0 in function drawContours"
                    myPts = [tlF, trF, brF, blF]; # Gives: "error: /home/asusu/opencv/opencv-master/modules/core/src/drawing.cpp:2152: error: (-215) npoints > 0 in function drawContours"

                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): After homogenization, myPts = %s" % \
                            str(myPts));

                    #contour = numpy.array([[[0,0]], [[10,0]], [[10,10]], [[5,4]]])
                    cv2.drawContours(image=xxx, contours=myPts, contourIdx=-1,
                                        color=(255,255,255), thickness=3);

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): After homogenization, myPts = %s" % \
                        str(myPts));

                # See http://docs.opencv.org/modules/core/doc/drawing_functions.html#line
                #   Inspired from http://stackoverflow.com/questions/18632276/how-to-draw-a-line-on-an-image-in-opencv
                """
                cv2.line(img=xxx, pt1=tlF, pt2=trF, color=(255,255,255), thickness=4);
                cv2.line(img=xxx, pt1=trF, pt2=brF, color=(255,255,255), thickness=4);
                cv2.line(img=xxx, pt1=brF, pt2=blF, color=(255,255,255), thickness=4);
                cv2.line(img=xxx, pt1=blF, pt2=tlF, color=(255,255,255), thickness=4);
                """
                #cv2.line(xxx, blF, tlF, (255,255,255));
                for myI in range(4):
                    cv2.line(img=xxx, pt1=myPts[myI], pt2=myPts[(myI+1) % 4], color=(255,255,255), thickness=4);

            """
            TODO: take only points within the warp transformation rectangle of the reference frame.
            IMPORTANT NOTE: Currently we instruct interp_space_time() to put
                black on all parts not covered by the warped (reference) frame.
            """
            """
            for r in range(xxx.shape[0]):
                for c in range(xxx.shape[1]):
                    if xxx[r][c].sum() > config.MEANINGFUL_DIFF_THRESHOLD: #10:
                        Z.append((r, c));
            """
            xxxSum = xxx.sum(axis=2);

            #!!!!TODO TODO: as Marius Leordean said we should normalize after luminance the ref and query frames and then keep only the points above the config.MEANINGFUL_DIFF_THRESHOLD

            #!!!!TODO TODO TODO: use instead cv2.threshold
            rDiff, cDiff = np.nonzero(xxxSum >= config.MEANINGFUL_DIFF_THRESHOLD * \
                                        xxx.shape[2]);
            meaningfulIndices = zip(rDiff, cDiff);
            if True:
                xxx = np.zeros((xxx.shape[0], xxx.shape[1]), dtype=np.uint8);
            xxx[(rDiff, cDiff)] = 255;
            #xxx[meaningfulIndices] = 255; # Gives: IndexError: index (271) out of range (0<=index<239) in dimension 0

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): rDiff.size = %d" % rDiff.size);

            assert rDiff.size == cDiff.size;
            assert rDiff.size == len(meaningfulIndices);

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint( \
                    "ecc_homo_spacetime(): meaningfulIndices = %s" % \
                                                    str(meaningfulIndices));

            if False:
                # Filter out the less important differences
                rDiffNot, cDiffNot = np.nonzero(xxxSum < config.MEANINGFUL_DIFF_THRESHOLD);
                xxx[(rDiffNot, cDiffNot)] = 0;

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): rDiffNot.size = %d" % \
                        rDiffNot.size);

                assert rDiffNot.size == cDiffNot.size;

            if True: #False:
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): xxx.dtype = %s" % \
                        str(xxx.dtype));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): xxx[:, :, 0].dtype = %s" % \
                        str(xxx[:, :, 0].dtype));

                    # Inspired from http://opencvpython.blogspot.ro/2012/06/hi-this-article-is-tutorial-which-try.html
                    for i in range(xxx.shape[2]):
                        #cv2.threshold(src, thresh, maxval, type[, dst]) -> retval, dst
                        #ret, thresh = cv2.threshold(xxx[:, :, i], 127, 255, cv2.THRESH_BINARY); # Gives error: error: /home/asusu/opencv/opencv-master/modules/imgproc/src/thresh.cpp:937: error: (-210)  in function threshold
                        ret, threshImg = cv2.threshold(src=xxx[:, :, i].astype(np.uint8), \
                                                    thresh=127, \
                                                    maxval=255, \
                                                    type=cv2.THRESH_BINARY); # Gives error: error: /home/asusu/opencv/opencv-master/modules/imgproc/src/thresh.cpp:937: error: (-210)  in function threshold
                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): ret = %s" % str(ret));
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): len(threshImg) = %s" % \
                            str(len(threshImg)));
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): threshImg = %s" % \
                            str(threshImg));

                    cv2.imwrite(o_path + \
                        ("%.6d_good_diff7_thresh_img" % tmplt_index) + \
                        imformat, threshImg);
                else:
                    threshImg = xxx;

                """
                See http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours
                cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
                """
                res = cv2.findContours( \
                                        #image=np.array(threshImg), \
                                        image=threshImg, \
                                        mode=cv2.RETR_TREE, \
                                        method=cv2.CHAIN_APPROX_SIMPLE); # Gives error: <<ValueError: too many values to unpack>>
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): res = %s" % str(res));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): len(res[0]) = %s" % \
                            str(len(res[0])));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): res[0] = %s" % str(res[0]));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): len(res[1]) = %s" % \
                            str(len(res[1])));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): res[1] = %s" % \
                            str(res[1]));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): len(res[2]) = %s" % \
                            str(len(res[2])));
                    common.DebugPrint( \
                            "ecc_homo_spacetime(): res[2][0] = %s" % \
                            str(res[2][0]));

                cv2.imwrite(o_path + ("%.6d_good_diff7_res_0" % tmplt_index) + imformat, \
                                np.array(res[0]));

                contours = res[1];
                hierarchy = res[2];

                colorC = 255;
                meaningfulContours = 0;
                xxx = np.zeros((xxx.shape[0], xxx.shape[1]), \
                                        dtype=np.uint8);
                for indexC, contour in enumerate(contours):
                    if len(contour) < 10:
                        continue

                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): len(contour) = %s" % \
                                                        str(len(contour)));
                        common.DebugPrint( \
                            "ecc_homo_spacetime(): contour = %s" % \
                                                            str(contour));
                    if False:
                        mask = np.zeros(xxx[:, :, 0].shape, np.uint8);
                        """
                        From http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#drawcontours
                        cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> None
                        """
                        cv2.drawContours( \
                                        #image=mask, \
                                        image=xxx, \
                                        contours=[contour], \
                                        contourIdx=0, \
                                        color=255 / (indexC + 1), \
                                        thickness=-1);
                        """
                        See
                          http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#mean
                         Calculates an average (mean) of array elements
                         with optional operation mask..
                        """
                        #mean = cv2.mean(xxx, mask=mask);
                    else:
                        """
                        From http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#drawcontours
                            <<thickness - Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.>>
                        """
                        cv2.drawContours( \
                                        #image=mask, \
                                        image=xxx, \
                                        contours=[contour], \
                                        contourIdx=0, \
                                        #color=255 / (indexC + 1), \
                                        #color=255, # The image is gray
                                        color=colorC, # The image is gray
                                        thickness=-1);

                    meaningfulContours += 1;
                    colorC -= 20;
                #common.DebugPrint("ecc_homo_spacetime(): meaningfulContours = %s" % \
                #                                    str(meaningfulContours));

            USE_CLUSTERING = False;

            """
            Clustering is slower than cv2.drawContours(), but could be useful
              if the objects/"components" are disconnected.
            """
            if USE_CLUSTERING == True:
                #!!!!TODO: implement in both caller and callee return a number (e.g., 4,5) biggest clusters
                """
                Complexity: how much it takes to do hierarchial-clustering,
                            plus Theta(2 * len(meaningfulIndices)).
                """
                clustersPixels = Clustering.HierarchicalClustering( \
                                                                   meaningfulIndices, \
                                                                   len(meaningfulIndices));

                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): xxx.shape = %s" % \
                        str(xxx.shape));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): len(Z=very meaningful points) = %d" % \
                        len(Z));
                    common.DebugPrint( \
                        "ecc_homo_spacetime(): Z=very meaningful points = %s" % \
                        str(Z));

                #xxx[Z] = np.array([255, 0, 0]); #127; # Usually gives exception like: <<IndexError: index (271) out of range (0<=index<239) in dimension 0>>
                if False:
                    for p in clustersPixels:
                        #xxx[p] = np.array([255, 0, 0]); # Blue colour
                        xxx[p] = np.array([0, 0, 255]); # Red colour
                else:
                    clusterColors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], \
                                    [255, 0, 255], [255, 255, 255], \
                                    [128, 128, 128], [128, 0, 128]];
                    """
                    See http://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists
                    (and https://docs.python.org/2/library/functions.html#zip)
                    """
                    if False:
                        unzippedPairList = zip(*clustersPixels);
                        xxx[unzippedPairList] = np.array([0, 0, 255]); # Red colour
                    else:
                        for cluster in clustersPixels:
                            unzippedPairList = zip(*clustersPixels[cluster]);
                            xxx[unzippedPairList] = \
                                    np.array(clusterColors[cluster]);
                            """
                            xxx[unzippedPairList] = \
                                    np.array([80 * cluster, \
                                            80 * cluster, \
                                            255 - cluster * 80]); # Red colour
                            """
                            if len(clustersPixels[cluster]) > 100: #!!!!TODO_PROFOUND: should we also look at previous (and future) frames to see if this cluster persists, ONLY in which case we should trigger the alarm?
                                #!!!!TODO TODO: report significant change in videos
                                pass;

            #!!!!TODO_PROFOUND: maybe use instead of findContours() find edges, etc

            tdiff2 = float(cv2.getTickCount());
            myTime = (tdiff2 - tdiff1) / cv2.getTickFrequency();
            print( \
                "ecc_homo_spacetime(): diff of the frames took %.6f [sec]" % \
                                                                        myTime);

        ############################# END DIFF FRAMES ##########################
        ############################# END DIFF FRAMES ##########################
        ############################# END DIFF FRAMES ##########################
        ############################# END DIFF FRAMES ##########################
        ############################# END DIFF FRAMES ##########################
        ############################# END DIFF FRAMES ##########################

        if False:
            # For fun: we apply Canny operator on the output image
            print("xxx.dtype = %s" % str(xxx.dtype));
            print("xxx.shape[2] = %d" % xxx.shape[2]);
            xxx = xxx.astype(np.uint8);
            print("xxx.dtype = %s" % str(xxx.dtype));
            for i in range(xxx.shape[2]):
                # Inspired from http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
                xxx[:, :, i] = cv2.Canny(xxx[:, :, i], 100, 200);
            #xxx = cv2.Canny(xxx, 100, 200);

        if config.VISUAL_DIFF_FRAMES == True:
            fileNameOImage = "%.6d_good_diff7" % tmplt_index;
        else:
            fileNameOImage = "%.6d_good" % tmplt_index;
        #%figure(2); imshow(uint8(xxx)); title('spatio-temporal alignment');
        #imwrite(uint8(xxx),[o_path num2str(tmplt_index,'%.6d_good') imformat]);
        cv2.imwrite(o_path + fileNameOImage + imformat, \
                    xxx.astype(int));

        if False: # Output the rather black (the "weighted") image
            if weighted_flag == 1:
                #%figure(3); imshow(uint8(xxx.*w)); title('alignment with weights');
                #imwrite(uint8(xxx.*w),[o_path num2str(tmplt_index,'%.6d_weights') imformat]);

                #xxx_weighted = xxx * w; # Gives: <<ValueError: operands could not be broadcast together with shapes (240,320,3) (240,320)>>
                for i in range(xxx.shape[2]):
                    xxx[:, :, i] *= w;

                cv2.imwrite(o_path + ("%.6d_weights" % tmplt_index) + imformat, \
                        xxx.astype(int));
        #%pause

    t2 = float(cv2.getTickCount());
    myTime = (t2 - t1) / cv2.getTickFrequency();
    print("ecc_homo_spacetime() took %.6f [sec]" % myTime);

    return fit;












import unittest

class TestSuite(unittest.TestCase):
    def testHessian(self):
        aZero = np.zeros( (10, 10) );
        self.assertTrue((aZero == 0).all());


    def testImgaussian(self):
        # Not tested, since not implemented

        """
        A = np.array([ \
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]);

        res = imgaussian(A, 6, 1);
        #common.DebugPrint("res from imgaussian() = %s" % str(res));

        #cv2.GaussianBlur returns:
        #  [[ 1.  4.  7.]
        #   [ 2.  5.  8.]
        #   [ 3.  6.  9.]]

        resGood = np.array([ \
                        [2.4357,   3.0767,   3.7178],
                        [4.3589,   5.0000,   5.6411],
                        [6.2822,   6.9233,   7.5643]]);
        """

        """
        res = imgaussian(A, 2, 10);
        Gives error:
          OpenCV Error: Assertion failed (ksize.width > 0 && ksize.width % 2 == 1 && ksize.height > 0 && ksize.height % 2 == 1) in unknown
          function, file ..\..\..\src\opencv\modules\imgproc\src\smooth.cpp, line 816
        """
        #common.DebugPrint("res from imgaussian() = %s" % str(res));

        """
        >> A=[1 2; 3 4];
        >> imgaussian(A, 1, 6)

        ans =
            1.9014    2.3005
            2.6995    3.0986

        >> A=[1 2 3; 4 5 6; 7 8 9];
        >> imgaussian(A, 1, 6)

        ans =
            2.4357    3.0767    3.7178
            4.3589    5.0000    5.6411
            6.2822    6.9233    7.5643

        >> A=[1 2 3 4; 5 6 7 8; 9 10 11 12];
        >> imgaussian(A, 1, 6)

        ans =
            2.7990    3.4941    4.3772    5.0723
            5.3633    6.0584    6.9416    7.6367
            7.9277    8.6228    9.5059   10.2010


        >> imgaussian(A, 1, 7)

        ans =
            2.8000    3.4949    4.3778    5.0727
            5.3637    6.0586    6.9414    7.6363
            7.9273    8.6222    9.5051   10.2000


        >> imgaussian(A, 2, 10)

        ans =
            4.2133    4.7125    5.2670    5.7662
            5.7235    6.2227    6.7773    7.2765
            7.2338    7.7330    8.2875    8.7867
        """


if __name__ == '__main__':
    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(threshold=1000000, linewidth=5000);
    unittest.main();

