import cv2

import config
import common
import Matlab

import numpy as np
import math

import sys
import unittest


# We store the Harris features in CSV format
def StoreMultiScaleHarrisFeatures(pathFileName, harrisFeatures):
    fOutput = open(pathFileName, "wt");

    for e in harrisFeatures:
        #fOutput.write("%f, %f, %d\n" % (e[0], e[1], e[2])); #.7f
        fOutput.write("%.9f, %.9f, %d\n" % (e[0], e[1], e[2])); #.7f
    fOutput.close();


def LoadMultiScaleHarrisFeatures(pathFileName):
    fInput = open(pathFileName, "rt");

    harrisFeatures = [];

    while True:
        fLine = fInput.readline();
        #common.DebugPrint("line = %s" % line);
        if not fLine: # fLine == None
            break;

        fLine = fLine.strip(" ");
        fLine = fLine.rstrip("\r\n");

        # We retrieve the Harris features from file in CSV format
        lineVals = fLine.split(", ");
        #"%f %f %d" % (e[0], e[1], e[2])

        #print lineVals;

        """
        harrisFeatures.append( (float(lineVals[0]), float(lineVals[1]), \
                                                        int(lineVals[2])) );
        """

        """
        harrisFeatures.append( [float(lineVals[0]), float(lineVals[1]), \
                                                        int(lineVals[2])] );
        """
        harrisFeatures.append( [int(float(lineVals[0])), int(float(lineVals[1])), \
                                                        int(lineVals[2])] );
    fInput.close();

    return np.array(harrisFeatures, dtype=np.int16); #np.int32



"""
!!!!TODO: Think more
Ev said:
    I had a feedback from other users recently and they told me that they tried pyrDown and they were not satisfied.
      So, I would work on each scale separately.
    Of course, you can use "filter2D" with a gaussian kernel, or directly the function "GaussianBlur",
       and then you can use the function cornerHarris. See the documentation of these functions.
       However, I should say that we used harris due to its increased repeatability
         compared to other detectors (as reported in literature).
       There may be detectors that work better in this problem, but this was not the scope of my research.
"""
#TODO: rename to ComputeMultiScaleHarrisFeaturesForImage)
# We compute the Harris features for a given image im, number of scales nos
def multi_scale_harris(im, nos, disp):
    """
    Followed advices from emails:
        - Vali Codreanu, Mar 18th, 2014:
        - Evangelidis, Apr th, 2014:
    """
    common.DebugPrint("Entered multi_scale_harris(nos=%d, disp=%s)" % \
                                                            (nos, str(disp)));

    t1 = float(cv2.getTickCount());

    # nos = number of scales

    im = common.ConvertImgToGrayscale(im);

    #print "feature_params (a dictionary) = %s" % str(feature_params)
    feature_params = dict( maxCorners = 1000,
                           qualityLevel = 0.01,
                           minDistance = 5, #9, #7, #~195 points, #6, #~210 points #3, #~300 points, #4, #~100 points, #5,#~85 Harris points found #8, ~45 found
                           blockSize = 19,
                           useHarrisDetector = True);
    """
    Alex: added useHarrisDetector for multi-scale harris detector inspired
      from Evangelidis, (otherwise it was using cornerMinEigenVal detector).
    """

    if False:
        pointsAux = cv2.goodFeaturesToTrack(im, **feature_params);

        points = [];
        if pointsAux == None:
            return points;

        for p in pointsAux:
            points.append((p[0][0], p[0][1], 1)); # we only have scale 1 now


    points = [];

    """
    multi-scale Harris detectors
        - http://opencv-users.1802565.n2.nabble.com/multi-scale-corners-detection-td4702476.html
            "Not in OpenCV, but you can always code it yourself using the available OpenCV functions.
            One approach is to generate a pyramid and apply cvGoodFeatureTracks to each level.
            You'll have to remove duplicate detected features though."
        - available in http://www.vlfeat.org/

    See for example of build image pyramids:
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html

    Note: cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType ]])
    """

    # We compute for pyramid of images the Harris features
    imScaled = im.copy();

    # Note: this is the only place where they have first the height (related to y) and width (related to x) - dimensions are y, x, like matrices
    im_h, im_w = im.shape[:2];



    """
    Using Evangelidis' scale params (local scale and integration scale) for
        multi-scale Harris
    """
    # scale values
    sigma_0 = 1.2;
    #n=[0:nos-1]; %scale levels
    n = np.array(range(nos)); #scale levels

    #sigma_D=sqrt(1.8).^n*sigma_0
    sigma_D = math.sqrt(1.8) ** n * sigma_0;

    #scn=sigma_D.^4;
    scn = sigma_D ** 4;

    common.DebugPrint("multi_scale_harris(): scn = %s" % (str(scn)));


    """
    Obtained from Evangelidis' code, we use these ranges to define ksizeGB below:
     mRange is basically the 1-D Gaussian kernel
        (Evangelidis uses meshgrid(mRange, mRange) to build a 2-D Gaussian kernel).
     mRange = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
     mRange = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
     mRange = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
     mRange = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
     mRange = [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    """
    ksizeGB = [11, 11, 15, 19, 27]; # kernel size GaussianBlur

    for i in range(nos):
        common.DebugPrint("multi_scale_harris(): i = %s" % (str(i)));

        #sd=sigma_D(i); %differentiation (local) scale
        sd = sigma_D[i]; #%differentiation (local) scale

        imScaled_h, imScaled_w = imScaled.shape[:2];

        #if False:
        if True:
            """
            From http://docs.opencv.org/modules/imgproc/doc/filtering.html
              Python: cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst

              where:
                ksize - Gaussian kernel size. ksize.width and ksize.height can
                        differ but they both must be positive and odd.
                        Or, they can be zero's and then they are computed
                            from sigma* .
                sigmaX - Gaussian kernel standard deviation in X direction.
                sigmaY - Gaussian kernel standard deviation in Y direction;
                            if sigmaY is zero, it is set to be equal to sigmaX,
                            if both sigmas are zeros, they are computed from
                            ksize.width and ksize.height , respectively
                            (see getGaussianKernel() for details); to fully
                            control the result regardless of possible future
                            modifications of all this semantics, it is
                            recommended to specify all of ksize, sigmaX,
                            and sigmaY.
                borderType - pixel extrapolation method
                                (see borderInterpolate() for details).


            NOTE:
              Since the 2D Gaussian kernel can be separable on both dimensions,
                GaussianBlur() calls getGaussianKernel() for dim X and Y.
              Python: cv2.getGaussianKernel(ksize, sigma[, ktype]) -> retval
                  Parameters:
                      ksize - Aperture size. It should be odd
                                    ( ksize \mod 2 = 1 ) and positive.
                      sigma - Gaussian standard deviation.
                            If it is non-positive, it is computed from ksize
                                as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 .
                      ktype - Type of filter coefficients.
                                It can be CV_32f or CV_64F.

              The function computes and returns the ksize x 1 matrix of
                        Gaussian filter coefficients:

              G_i= \alpha * e^{-(i-(ksize-1)/2)^2/(2*sigma)^2},

              where i=0..ksize-1 and \alpha is the scale factor chosen
                    so that \sum_i G_i=1.

              Two of such generated kernels can be passed to sepFilter2D()

              Alex: as we can see above, the mean of the Gaussian coefs
                are computed with sigma=sigma and mean=0.

            REFERENCE explaining the math behind GaussianBlur():
                http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
            """
            ks = ksizeGB[i];
            imScaled = cv2.GaussianBlur(src=imScaled, \
                                        ksize=(ks, ks), sigmaX=sd*sd); #0.0);

            common.DebugPrint("multi_scale_harris(): ks = %s" % (str(ks)));
            common.DebugPrint("multi_scale_harris(): sd^2 = %s" % (str(sd*sd)));

            #feature_params["minDistance"] = 5 - (i / 2); // i is the scale in use
            # The corners with the minimal eigenvalue less than \texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y) are rejected.
            # This is the threshold used in non-maximal supression.
            feature_params["qualityLevel"] = 0.001 * scn[i]; # * 30;
            feature_params["minDistance"] = 0.0; #100; (Returns very few points)

            """
            blockSize param given to cv::goodFeaturesToTrack() is used by:
                - cornerHarris, for Sobel filter (as aperture_size parameter);
                - boxFilter, which is applied exactly before computing the
                    Harris measure.

            NOTE: ksize of cv::cornerHaris() is set to 3 by
                cv::goodFeaturesToTrack() .
              From http://docs.opencv.org/modules/imgproc/doc/feature_detection.html?highlight=cornerharris#cornerharris
                "ksize - Aperture parameter for the Sobel() operator."
            """
            feature_params["blockSize"] = ks; #1!!!!

            feature_params["k"] = 0.06;

            common.DebugPrint("multi_scale_harris(): feature_params = %s" % \
                                                    (str(feature_params)));

            """
            cv::goodFeaturesToTrack() with Harris option does the following
                (http://docs.opencv.org/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack ,
                code at https://github.com/Itseez/opencv/blob/451be9ed538bbad6be61a783c2f4fa247fc83930/modules/imgproc/src/featureselect.cpp):
              <<The function finds the most prominent corners in the image or
                    in the specified image region, as described in [Shi94]:
                Function calculates the corner quality measure at every source
                    image pixel using the cornerMinEigenVal() or cornerHarris() .
                IMPORTANT: Function performs a non-maximum suppression (the
                            local maximums in 3 x 3 neighborhood are retained).
                The corners with the minimal eigenvalue less than
                    qualityLevel \cdot \max_{x,y} qualityMeasureMap(x,y) are rejected.
                The remaining corners are sorted by the quality measure in the
                    descending order.
                Function throws away each corner for which there is a stronger
                    corner at a distance less than maxDistance. >>
            """

            """
            Python: cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k ]]]]]) -> corners
            useHarrisDetector - Parameter indicating whether to use a Harris detector (see cornerHarris()) or cornerMinEigenVal()
            k - Free parameter of the Harris detector (see page 324 for details on the dst(x, y) formula using k)

            From opencv2refman.pdf, page 327, Section 3.7,
              "The function finds the most prominent corners in the image or in
                the specified image region, as described in [Shi94]"
            """
            #pointsAux = cv2.goodFeaturesToTrack(imScaled, **feature_params);
            pointsAux = cv2.goodFeaturesToTrack(imScaled, maxCorners=3600, #1000,
                                                qualityLevel=0.01,
                                                #qualityLevel=0.001 * scn[i], # * 30;
                                                minDistance=5, #9, #7, #~195 points, #6, #~210 points #3, #~300 points, #4, #~100 points, #5,#~85 Harris points found #8, ~45 found
                                                blockSize=ks, #19,
                                                useHarrisDetector=True,
                                                k=0.06);
                                                # The corners with the minimal eigenvalue less than \texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y) are rejected.
                                                # This is the threshold used in non-maximal supression.
                                                #feature_params["minDistance"]=0.0); #100; (Returns very few points)
        else:
            # Inspired from https://stackoverflow.com/questions/18255958/harris-corner-detection-and-localization-in-opencv-with-python

            """
            From OpenCV help:
              http://docs.opencv.org/modules/imgproc/doc/feature_detection.html#void%20cornerHarris%28InputArray%20src,%20OutputArray%20dst,%20int%20blockSize,%20int%20ksize,%20double%20k,%20int%20borderType%29
              Python: cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType ] ]) -> dst
            """
            cornerImg = cv2.cornerHarris(src=imScaled, blockSize=2, ksize=3, \
                                            k=0.04, borderType=cv2.BORDER_DEFAULT);
            if False:
                cornerImg = cv2.cornerHarris(src=imScaled, blockSize=19, ksize=3, \
                                            k=0.01, borderType=cv2.BORDER_DEFAULT);

            common.DebugPrint("multi_scale_harris(): cornerImg=%s" % \
                                                        (str(cornerImg)));

            """
            See http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
                Python: cv2.normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]]) -> dst
            """
            cornerImg = cv2.normalize(src=cornerImg, alpha=0, beta=255, \
                                        norm_type=cv2.NORM_MINMAX, \
                                        dtype=cv2.CV_32FC1);
            common.DebugPrint( \
                "multi_scale_harris(): cornerImg (after normalize)=%s" % \
                                                        (str(cornerImg)));

            """
            # From http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#convertscaleabs:
            Python: cv2.convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst
            """
            cornerImg = cv2.convertScaleAbs(cornerImg);
            common.DebugPrint( \
                "multi_scale_harris(): cornerImg (after convertScaleAbs)=%s" % \
                                                        (str(cornerImg)));

            pointsAux = [];

            """
            Inspired from
              http://docs.opencv.org/doc/tutorials/features2d/trackingmotion/harris_detector/harris_detector.html
            """
            thresh = 200; #200.0; #10e-06

            # iterate over pixels to get corner positions
            w, h = imScaled.shape
            common.DebugPrint("multi_scale_harris(): w, h = %s" % \
                                                        (str((w, h))));
            for y in range(0, h):
                for x in range (0, w):
                    #harris = cv2.cv.Get2D( cv2.cv.fromarray(cornerimg), y, x)
                    #if cornerimg[x,y] > 64:
                    # Following
                    if cornerImg[x, y] > thresh:
                        #common.DebugPrint("corner at ", x, y)
                        pointsAux.append((x, y));
                        """
                        cv2.circle( cornershow,  # dest
                                    (x,y),      # pos
                                    4,          # radius
                                    (115,0,25)  # color
                                    );
                        """

        if pointsAux == None:
            #return points
            continue;

        common.DebugPrint("multi_scale_harris(): len(pointsAux)=%d" % \
                                                            (len(pointsAux)));
        common.DebugPrint("multi_scale_harris(): len(pointsAux[0])=%d" % \
                                                        (len(pointsAux[0])));
        common.DebugPrint("multi_scale_harris(): pointsAux=%s" % (str(pointsAux)));

        # Note: pointsAux contain just the (x, y) coordinates of the points like this: [ [[x1, y1]] [[x2, y2]] ]
        #common.DebugPrint("multi_scale_harris(): pointsAux = %s" % str(pointsAux));
        for p in pointsAux:
            points.append((p[0][0] * float(im_w)/imScaled_w, p[0][1] * float(im_h)/imScaled_h, i + 1));
            #points.append((p[0] * float(im_w)/imScaled_w, p[1] * float(im_h)/imScaled_h, i + 1));
        sys.stdout.flush();

        # Sort the points after 2nd element first and then the 1st element:
        def CmpFunc(x, y):
            #return x - y
            if x[1] > y[1]:
                return 1
            elif x[1] < y[1]:
                return -1
            else: #(x[1] == y[1]):
                if (x[0] > y[0]):
                    return 1
                elif (x[0] < y[0]):
                    return -1
                else:
                    return 0

            return 0
        points.sort(cmp=CmpFunc);


        if False:
            # We work on the Gaussian pyramid
            imScaled = cv2.pyrDown(imScaled);

            common.DebugPrint("multi_scale_harris(): imScaled dimensions after cv2.pyrDown are %s" % str(imScaled.shape[:2]));

            #TODO: make course grained harris features be ~roughly same number like in the Matlab code - should I increase the minDistance, etc???
            #if False:
            if True:
                feature_params["minDistance"] = 5 - (i / 2);
                common.DebugPrint("multi_scale_harris(): feature_params[minDistance] = %d" % feature_params["minDistance"]);

    points = np.array(points);

    #if False:
    if True:
        common.DebugPrint("multi_scale_harris(): len(points)=%d, points = %s" % (len(points), str(points)));

    """
    if points is not None:
        # We display this information only at the first call of this function,
        #    BUT not in the next calls.
        for x, y in points[:,0]:
            cv2.circle(img1, (x, y), 5, green, -1)
        common_cv.draw_str(img1, (20, 20), \
                "feature count (from goodFeaturesToTrack): %d" % len(p))
    """

    t2 = float(cv2.getTickCount());
    myTime = (t2 - t1) / cv2.getTickFrequency();
    ##common.DebugPrint("multiscale_quad_tree() took %.6f [sec]" % myTime);
    common.DebugPrint("multi_scale_harris() took %.6f [sec]" % myTime);

    # We could even call this function when points is list, not an numpy array
    #StoreMultiScaleHarrisFeatures("Videos/harloc%", points);

    return points;


# Used by all methods - full-search, VD, (also BoW??)











"""
Note:
In OpenCV non-maximal suppression is implemented in the following functions at least:
    - canny() !!!!check if implementation is better
    - goodFeaturesToTrack() - but it is fixed on a 3x3 neighborhood
    - StarFeature, StarFeatureDetector
"""
#function [r,c, rsubp, csubp] = my_nms(cim, radius, thresh)
def my_nms(cim, radius, thresh):
    common.DebugPrint("Entered my_nms(cim.shape=%s, radius=%s, thresh=%s)" % \
                    (str(cim.shape), str(radius), str(thresh)));
    #%% modification of Peter-Kovesi non-maximum suppression software by
    #%% G.Evangelidis

    common.DebugPrint("my_nms(): cim = %s" % str(cim));

    #%subPixel = nargout == 4;            % We want sub-pixel locations
    #subPixel=1;
    subPixel = 1;

    #[rows,cols] = size(cim)
    rows, cols = cim.shape;

    common.DebugPrint("my_nms(): rows, cols (cim.shape) = %s" % str((rows, cols)));

    #% Extract local maxima by performing a grey scale morphological
    #% dilation and then finding points in the corner strength image that
    #% match the dilated image and are also greater than the threshold.

    sze = 2 * radius + 1                   #% Size of dilation mask.

    common.DebugPrint("my_nms(): cim.shape = %s" % str(cim.shape));
    #common.DebugPrint("my_nms(): cim = %s" % str(cim));
    common.DebugPrint("my_nms(): sze = %s" % str(sze));

    """
    Alex: we pass sze only being odd number and order = sze^2 makes ordfilt2()
        return the maximum from the domain.
    """
    #%% This modification runs 4x faster (11 secs less in a 6-scale approach)
    #mx = ordfilt2(cim,sze^2,ones(sze)); #% Grey-scale dilate.
    mx = Matlab.ordfilt2(cim, pow(sze, 2), np.ones((sze, sze))); #% Grey-scale dilate.
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("my_nms(): mx = %s" % str(mx));

    #% mx=my_max_filter(cim,[sze,sze]); %my mex-file for max-filter

    #%% This modification verify that the central point is the unique maximum in
    #%% neighborhood
    #% mx2= ordfilt2(cim,sze^2-1,ones(sze)); % This is used to vrify that the
    #% central point is the unique maximum in neighborhood
    #% imagesc(cim)
    #% pause
    #% imagesc(mx)
    #% pause
    #% close

    #% Make mask to exclude points within radius of the image boundary.
    #bordermask = zeros(size(cim));
    #bordermask = np.zeros(cim.shape);
    bordermask = np.zeros(cim.shape, dtype=np.uint8);

    #Alex: sets to 1 the matrix, except the first and last radius lines and first and last radius columns, which are left 0
    #bordermask(radius+1:end-radius, radius+1:end-radius) = 1;
    bordermask[radius: -radius, radius: -radius] = 1;

    #% Find maxima, threshold, and apply bordermask
    #cimmx = (cim==mx) & (cim>thresh) & bordermask; #%Peter-Kovesi
    cimmx = np.logical_and( \
                            np.logical_and((cim == mx), (cim > thresh)), \
                            bordermask); #%Peter-Kovesi
    #% cimmx = (cim==mx) & (cim~=mx2) & (cim>thresh) & bordermask; %my modification

    common.DebugPrint("my_nms(): cimmx.shape = %s" % str(cimmx.shape));
    #common.DebugPrint("my_nms(): cimmx = %s" % str(cimmx));

    #[r,c] = find(cimmx)                #% Find row,col coords.
    if True:
        """
        Retrieving indices in "Fortran" (column-major) order, like in Matlab.
        We do this in order to get the harris points sorted like in Matlab.
        """
        c, r = np.nonzero(cimmx.T);
    else:
        # Retrieving indices in row-major order.
        r, c = np.nonzero(cimmx);
    common.DebugPrint("my_nms(): r = %s" % str(r));
    common.DebugPrint("my_nms(): c = %s" % str(c));

    if subPixel:        #% Compute local maxima to sub pixel accuracy
        # From Matlab help: "Determine whether array is empty" "An empty array has at least one dimension of size zero, for example, 0-by-0 or 0-by-5."
        #if not isempty(r): #% ...if we have some ponts to work with
        if r.size > 0: #% ...if we have some ponts to work with
            #ind = sub2ind(size(cim),r,c);   #% 1D indices of feature points
            ind = Matlab.sub2ind(cim.shape, r, c); #% 1D indices of feature points

            w = 1;         #% Width that we look out on each side of the feature
            #% point to fit a local parabola

            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("my_nms(): ind.shape = %s" % str(ind.shape));
                common.DebugPrint("my_nms(): ind = %s" % str(ind));
                common.DebugPrint("my_nms(): ind - w = %s" % str(ind - w));

            # Don't forget that we are now in column major ('F'/Fortran) order

            #% Indices of points above, below, left and right of feature point
            #indrminus1 = max(ind-w,1);
            #indrminus1 = np.max(ind - w, 0);
            # In Matlab this is what it returns for 1D ind:
            assert ind.ndim == 1;
            indrminus1 = ind - w;

            #indrplus1  = min(ind+w,rows*cols);
            #indrplus1  = np.min(ind + w, rows * cols);
            # In Matlab this is what it returns for 1D ind:
            assert ind.ndim == 1;
            indrplus1 = ind + w;

            #indcminus1 = max(ind-w*rows,1);
            #indcminus1 = np.max(ind - w * rows, 1);
            # In Matlab this is what it returns for 1D ind:
            assert ind.ndim == 1;
            indcminus1 = ind - w * rows;

            #indcplus1  = min(ind+w*rows,rows*cols);
            #indcplus1  = np.min(ind + w * rows, rows * cols);
            # In Matlab this is what it returns for 1D ind:
            assert ind.ndim == 1;
            indcplus1 = ind + w * rows;


            # De-flattening the index arrays back into tuple, as accepted by numpy
            # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.unravel_index.html
            ind = np.unravel_index(ind, cim.shape, order="F");
            indrminus1 = np.unravel_index(indrminus1, cim.shape, order="F");
            indrplus1 = np.unravel_index(indrplus1, cim.shape, order="F");
            indcminus1 = np.unravel_index(indcminus1, cim.shape, order="F");
            indcplus1 = np.unravel_index(indcplus1, cim.shape, order="F");


            #% Solve for quadratic down rows
            #cy = cim(ind);
            cy = cim[ind];

            # In Matlab ay has float elements
            ay = (cim[indrminus1] + cim[indrplus1]) / 2.0 - cy;

            #by = ay + cy - cim(indrminus1);
            by = ay + cy - cim[indrminus1];

            #rowshift = -w*by./(2*ay);       #% Maxima of quadradic
            rowshift = -w * by / (2.0 * ay); #% Maxima of quadradic

            #% Solve for quadratic across columns
            #cx = cim(ind);
            cx = cim[ind];

            #ax = (cim(indcminus1) + cim(indcplus1))/2 - cx;
            ax = (cim[indcminus1] + cim[indcplus1]) / 2.0 - cx;

            #bx = ax + cx - cim(indcminus1);
            bx = ax + cx - cim[indcminus1];

            #colshift = -w*bx./(2*ax);       #% Maxima of quadratic
            colshift = -w * bx / (2.0 * ax); #% Maxima of quadratic

            rsubp = r + rowshift;  #% Add subpixel corrections to original row
            csubp = c + colshift;  #% and column coords.
        else:
            #rsubp = []; csubp = [];
            rsubp = np.array([]);
            csubp = np.array([]);

    """
    % if nargin==4 & ~isempty(r)     % Overlay corners on supplied image.
    %     figure, imshow(im,[]), hold on
    %     if subPixel
    %         plot(csubp,rsubp,'r+'), title('corners detected');
    %     else
    %         plot(c,r,'r+'), title('corners detected');
    %     end
    % end
    """
    return r, c, rsubp, csubp;


#function [points] = multi_scale_harris(im, nos, disp)
#%function [points] = multi_scale_harris(im, nos, disp)
def multi_scale_harris_Evangelidis(im, nos, disp):
    common.DebugPrint("Entered multi_scale_harris_Evangelidis(nos=%d, disp=%s)" % \
                                                            (nos, str(disp)));
    #tic

    #if size(im, 3) == 3:
    #    im = rgb2gray(im)

    tMSH1 = float(cv2.getTickCount());

    if im.ndim == 3:
        im = common.ConvertImgToGrayscale(im);

    #im = im2double(im)
    # From https://stackoverflow.com/questions/10873824/how-to-convert-2d-float-numpy-array-to-2d-int-numpy-array:
    #DO NOT USE: im = im.astype(np.uint8); # This messes up tremendously the computation of harlocs
    #im = im.astype(float);
    im = im.astype(np.float32);

    #!!!!COR is an UNused variable
    #COR = zeros(size(im,1), size(im,2), size(im,3))
    #COR = np.zeros((im.shape[0], im.shape[1], im.shape[1]));

    # scale values

    sigma_0 = 1.2;
    #n=[0:nos-1]; %scale levels
    n = np.array(range(nos)); #scale levels

    #sigma_D=sqrt(1.8).^n*sigma_0
    sigma_D = math.sqrt(1.8) ** n * sigma_0;

    #points=[];
    points = np.array([]);

    #scn=sigma_D.^4;
    scn = sigma_D ** 4;

    #for i=1:length(sigma_D)
    #for i in range(1, len(sigma_D) + 1):
    for i in range(1, nos + 1):
        common.DebugPrint("multi_scale_harris_Evangelidis(): i = %s" % (str(i)));

        #sd=sigma_D(i); %differentiation (local) scale
        sd = sigma_D[i - 1]; #%differentiation (local) scale

        #si=sd/.5; %integration scale
        si = sd / 0.5; #integration scale

        w = 3 * sd; # size for gaussian kernel to compute derivatives: 3 times local_scale

        r_w = int(round(w));
        common.DebugPrint("multi_scale_harris_Evangelidis(): r_w = %s" % \
                                                                (str(r_w)));

        #if mod(round(w),2):
        if (r_w % 2) == 1:
            #[xco,yco] = meshgrid(-round(w):round(w),-round(w):round(w))
            mRange = range(-r_w, r_w + 1);
            xco, yco = Matlab.meshgrid(mRange, mRange);
        else:
            #[xco,yco] = meshgrid(-(round(w)+1):round(w)+1,-(round(w)+1):round(w)+1)
            mRange = range(-(r_w + 1), r_w + 2);
            xco, yco = Matlab.meshgrid(mRange, mRange);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): xco = %s" % \
                                                                    str(xco));
            common.DebugPrint("multi_scale_harris_Evangelidis(): yco = %s" % \
                                                                    str(yco));

        # Note: even for HD frames, xco.shape = (11, 11) (yco the same)

        #arg = -(xco.*xco + yco.*yco) / (2*sd*sd)
        #arg = -(xco * xco + yco * yco) / (2.0 * sd * sd);
        arg = -(xco ** 2 + yco ** 2) / (2.0 * sd * sd);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): arg = %s" % \
                                                                    str(arg));

        #%2d gaussian kernel
        """
        From http://www.mathworks.com/help/matlab/ref/exp.html:
            "exp(X) returns the exponential for each element of array X."
        """
        #g=exp(arg)/(2*pi*sd^2); #2d gaussian kernel
        g = np.exp(arg) / (2.0 * math.pi * pow(sd, 2));

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): g = %s" % \
                                                                    str(g));

        # normalize to suppress any gain
        #if sum(g(:))~=0:
        g_sum = g.sum();
        #if abs(g.sum()) > 1.0e-6:
        if abs(g_sum) > 1.0e-6:
            #g = g / sum(g(:));
            #g = g / float(g.sum());
            g /= float(g_sum);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): sd = %s" % str(sd));
            common.DebugPrint("multi_scale_harris_Evangelidis(): w = %s" % str(w));

        """
        #%Instead of computing derivatives in the filtered image,
        we filter the image with the derivatives of the kernel.
        """

        #% kernels for gaussian derivatives
        #gx=-xco.*g/(sd*sd);
        gx = -xco * g / float(sd * sd);

        #gy=-yco.*g/(sd*sd);
        gy = -yco * g / float(sd * sd);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): gx = %s" % \
                                                                    str(gx));
            common.DebugPrint("multi_scale_harris_Evangelidis(): gy = %s" % \
                                                                    str(gy));

        """
        multi_scale_harris_Evangelidis(): arg.shape = (11, 11)
        multi_scale_harris_Evangelidis(): g.shape = (11, 11)
        multi_scale_harris_Evangelidis(): gx.shape = (11, 11)
        multi_scale_harris_Evangelidis(): gy.shape = (11, 11)
        multi_scale_harris_Evangelidis(): gi.shape = (15, 15)
        """

        common.DebugPrint("multi_scale_harris_Evangelidis(): xco.shape = %s" % str(xco.shape));
        common.DebugPrint("multi_scale_harris_Evangelidis(): yco.shape = %s" % str(yco.shape));
        common.DebugPrint("multi_scale_harris_Evangelidis(): arg.shape = %s" % str(arg.shape));
        common.DebugPrint("multi_scale_harris_Evangelidis(): g.shape = %s" % str(g.shape));
        common.DebugPrint("multi_scale_harris_Evangelidis(): gx.shape = %s" % str(gx.shape));
        common.DebugPrint("multi_scale_harris_Evangelidis(): gy.shape = %s" % str(gy.shape));

        #%compute the derivatives
        #Ix = imfilter(im, gx, 'replicate');
        Ix = Matlab.imfilter(im, gx, "replicate");

        #Iy = imfilter(im, gy, 'replicate');
        Iy = Matlab.imfilter(im, gy, "replicate");
        #% Alex: Ix and Iy have the same size as im

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): Ix = %s" % \
                                                                    str(Ix));
            common.DebugPrint("multi_scale_harris_Evangelidis(): Iy = %s" % \
                                                                    str(Iy));

        #% gaussian kernel to compute 2nd moment matrix
        #if mod(floor(6*si),2) %size: six times the integration scale
        if int(math.floor(6 * si)) % 2 == 1: #size: six times the integration scale
            #gi = fspecial('ga',max(1,fix(6*si)), si)
            gi = Matlab.fspecial("ga", max(1, Matlab.fix(6 * si)), si);
        else:
            #gi = fspecial('ga',max(1,fix(6*si)+1), si)
            gi = Matlab.fspecial("ga", max(1, Matlab.fix(6 * si) + 1), si);

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): gi = %s" % \
                                                                    str(gi));
        common.DebugPrint("multi_scale_harris_Evangelidis(): gi.shape = %s" % \
                                                                str(gi.shape));

        #Ix2 = imfilter(Ix.^2, gi,  'replicate');
        Ix2 = Matlab.imfilter(Ix ** 2, gi,  "replicate");

        #Iy2 = imfilter(Iy.^2, gi,  'replicate');
        Iy2 = Matlab.imfilter(Iy ** 2, gi,  "replicate");

        #Ixy = imfilter(Ix.*Iy, gi, 'replicate');
        Ixy = Matlab.imfilter(Ix * Iy, gi, "replicate");
        #% Alex: Ix2, Iy2 and Ixy have the same size as im

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): Ix2 = %s" % \
                                                                    str(Ix2));
            common.DebugPrint("multi_scale_harris_Evangelidis(): Iy2 = %s" % \
                                                                    str(Iy2));
            common.DebugPrint("multi_scale_harris_Evangelidis(): Ixy = %s" % \
                                                                    str(Ixy));
        common.DebugPrint("multi_scale_harris_Evangelidis(): Ix2.dtype = %s" % \
                          str(Ix2.dtype));
        common.DebugPrint("multi_scale_harris_Evangelidis(): Iy2.dtype = %s" % \
                          str(Iy2.dtype));
        common.DebugPrint("multi_scale_harris_Evangelidis(): Ixy.dtype = %s" % \
                          str(Ixy.dtype));

        #%% Cornerness measure
        #% Noble measure.
        #%
        #%     M = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2 + eps);

        #% Harris measure.
        #M = (Ix2.*Iy2 - Ixy.^2) - .06*(Ix2 + Iy2).^2;
        M = (Ix2 * Iy2 - Ixy ** 2) - 0.06 * (Ix2 + Iy2) ** 2;

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): M.dtype = %s" % \
                          str(M.dtype));
            common.DebugPrint("multi_scale_harris_Evangelidis(): M = %s" % \
                                                                str(M));

        #% Alex: scn is a vector - see definition above
        #% Alex: M has the same size as im
        #M = scn(i)*M;
        M = scn[i - 1] * M;

        #thresh = 0.001*max(M(:));
        thresh = 0.001 * M.max();

        #%       imagesc(M==abs(M));axis on;
        #%   colorbar
        #%   pause

        """
        %keep points that are the maximum in a neighborhood of
            radius=round(3*si/2) and are above thresh
        %non-maximum supression and subpixel refinement
        """
        #[r,c, rsubp, csubp] = my_nms(M, round(3*si/2), thresh);
        r, c, rsubp, csubp = my_nms(M, round(3 * si / 2.0), thresh);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multi_scale_harris_Evangelidis(): r.shape = %s" % str(r.shape));
            common.DebugPrint("multi_scale_harris_Evangelidis(): r = %s" % str(r));
            common.DebugPrint("multi_scale_harris_Evangelidis(): c.shape = %s" % str(c.shape));
            common.DebugPrint("multi_scale_harris_Evangelidis(): c = %s" % str(c));
            common.DebugPrint("multi_scale_harris_Evangelidis(): rsubp.shape = %s" % str(rsubp.shape));
            common.DebugPrint("multi_scale_harris_Evangelidis(): rsubp = %s" % str(rsubp));
            common.DebugPrint("multi_scale_harris_Evangelidis(): csubp.shape = %s" % str(csubp.shape));
            common.DebugPrint("multi_scale_harris_Evangelidis(): csubp = %s" % str(csubp));

        #% Alex: r,c, rsubp, csubp seem to always be the same size - and the
        #%          size I've seen is 56 * 1????
        #pp=[rsubp, csubp, i*ones(size(r,1),1)];
        pp = np.c_[rsubp, csubp, i * np.ones((r.shape[0], 1))];

        #% Alex: here we add more rows (pp) to points below the existing rows of
        #%                       points
        #points=[points; pp];
        common.DebugPrint("multi_scale_harris_Evangelidis(): points.shape = %s" % str(points.shape));
        common.DebugPrint("multi_scale_harris_Evangelidis(): pp.shape = %s" % str(pp.shape));
        if points.size == 0:
            # Avoiding exception: "ValueError: arrays must have same number of dimensions"
            points = pp;
        else:
            points = np.r_[points, pp];

    #toc
    if disp:
        assert False; # not implemented the display of Harris features
        """
        figure;
        imshow(im,[]) #hold on
        title('corners detected')
        for i = range(1, size(points,1) + 1):
            rectangle('Position',[points(i,2)-3*points(i,3),points(i,1)-3*points(i,3),...
                2*3*points(i,3),2*3*points(i,3)],'Curvature',[1,1],'EdgeColor','w','LineWidth',2)
            plot(points(i,2),points(i,1),'r+')
        """

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("multi_scale_harris_Evangelidis(): points = %s" % str(points));
    common.DebugPrint("multi_scale_harris_Evangelidis(): points.shape = %s" % str(points.shape));

    tMSH2 = float(cv2.getTickCount());
    myTime = (tMSH2 - tMSH1) / cv2.getTickFrequency();
    print("multi_scale_harris_Evangelidis(): multi_scale_harris_Evangelidis() took %.6f [sec]" % myTime);

    return points;



class TestSuite(unittest.TestCase):
    def testMulti_scale_harris(self):
        multi_scale_harris = multi_scale_harris_Evangelidis;

        videoPathFileName = "Videos/input/001001.jpeg";

        t1 = float(cv2.getTickCount());

        if config.OCV_OLD_PY_BINDINGS:
            #img = cv2.imread(videoPathFileName, cv2.CV_LOAD_IMAGE_GRAYSCALE);
            img = cv2.imread(videoPathFileName);
        else:
            #img = cv2.imread(videoPathFileName, cv2.IMREAD_GRAYSCALE);
            img = cv2.imread(videoPathFileName, cv2.IMREAD_GRAYSCALE);

        points = multi_scale_harris(img, 5, 0);
        #points = multi_scale_harris_Evangelidis(img, 5, 0);

        t2 = float(cv2.getTickCount());
        myTime = (t2 - t1) / cv2.getTickFrequency();
        common.DebugPrint("testMulti_scale_harrisy(): " \
                            "took %.6f [sec]" % myTime);

        """
        Matlab harlocs result of the 1st frame of the query video
            from Evangelidis.
        """
        pointsGood = np.array(\
            [[131.632095, 6.154750, 1.000000],
            [146.831779, 6.672792, 1.000000],
            [115.641756, 8.737362, 1.000000],
            [142.117458, 37.994987, 1.000000],
            [121.826803, 41.448319, 1.000000],
            [155.752696, 43.713423, 1.000000],
            [210.835970, 53.126403, 1.000000],
            [140.393291, 54.016071, 1.000000],
            [117.885810, 55.283617, 1.000000],
            [205.928233, 61.319267, 1.000000],
            [138.390780, 67.062827, 1.000000],
            [132.604070, 74.497933, 1.000000],
            [141.655730, 77.337403, 1.000000],
            [153.888428, 82.707709, 1.000000],
            [152.447974, 90.990718, 1.000000],
            [134.979613, 92.747693, 1.000000],
            [127.378205, 97.745154, 1.000000],
            [171.349592, 108.030164, 1.000000],
            [133.469991, 110.740279, 1.000000],
            [134.561093, 124.144843, 1.000000],
            [151.187393, 132.835981, 1.000000],
            [136.938641, 136.841886, 1.000000],
            [127.437358, 162.236749, 1.000000],
            [120.906237, 172.143386, 1.000000],
            [128.137890, 172.123635, 1.000000],
            [132.828951, 187.548345, 1.000000],
            [116.285979, 192.336551, 1.000000],
            [114.297469, 202.989782, 1.000000],
            [113.934076, 216.259835, 1.000000],
            [123.041898, 217.501817, 1.000000],
            [148.332124, 239.566719, 1.000000],
            [109.180759, 240.515507, 1.000000],
            [121.629418, 245.107743, 1.000000],
            [113.537399, 249.791571, 1.000000],
            [112.300557, 259.137270, 1.000000],
            [152.906100, 266.142001, 1.000000],
            [101.649102, 268.128725, 1.000000],
            [104.259044, 275.324923, 1.000000],
            [111.440298, 285.751911, 1.000000],
            [110.143546, 304.877568, 1.000000],
            [112.289315, 312.505130, 1.000000],
            [146.562562, 6.824207, 2.000000],
            [115.972203, 8.002075, 2.000000],
            [129.718912, 9.655225, 2.000000],
            [142.725688, 38.175548, 2.000000],
            [122.667561, 41.279501, 2.000000],
            [155.389401, 44.108806, 2.000000],
            [139.712007, 53.718072, 2.000000],
            [116.722052, 58.775573, 2.000000],
            [206.355399, 61.068029, 2.000000],
            [133.209928, 74.533496, 2.000000],
            [141.583546, 77.085668, 2.000000],
            [154.165950, 81.880436, 2.000000],
            [152.044304, 91.342924, 2.000000],
            [135.196273, 93.660427, 2.000000],
            [171.422921, 107.697442, 2.000000],
            [133.203511, 110.668542, 2.000000],
            [134.232848, 124.258084, 2.000000],
            [150.604853, 133.707842, 2.000000],
            [136.747305, 136.546157, 2.000000],
            [121.716396, 172.337590, 2.000000],
            [137.167345, 183.791824, 2.000000],
            [116.122641, 193.927587, 2.000000],
            [115.061848, 202.855561, 2.000000],
            [114.068695, 215.361333, 2.000000],
            [109.915388, 240.612238, 2.000000],
            [113.635058, 250.004103, 2.000000],
            [112.325007, 258.326368, 2.000000],
            [102.209163, 269.177027, 2.000000],
            [154.048196, 270.555867, 2.000000],
            [112.109784, 286.451163, 2.000000],
            [111.942606, 311.741336, 2.000000],
            [146.549699, 7.963894, 3.000000],
            [143.168806, 38.916493, 3.000000],
            [126.133408, 41.828468, 3.000000],
            [154.732134, 44.475668, 3.000000],
            [112.664304, 59.016811, 3.000000],
            [207.799639, 59.216013, 3.000000],
            [141.379564, 76.200705, 3.000000],
            [151.700405, 91.236859, 3.000000],
            [132.341399, 96.944035, 3.000000],
            [171.495430, 107.286690, 3.000000],
            [125.520125, 108.855125, 3.000000],
            [133.913674, 125.234340, 3.000000],
            [136.470097, 135.637420, 3.000000],
            [126.880607, 164.210606, 3.000000],
            [138.041934, 184.358483, 3.000000],
            [115.787910, 194.494653, 3.000000],
            [110.934231, 240.984736, 3.000000],
            [112.266318, 255.817086, 3.000000],
            [103.365679, 270.465536, 3.000000],
            [155.160827, 271.397790, 3.000000],
            [112.894407, 286.559136, 3.000000],
            [111.261756, 309.009162, 3.000000],
            [143.884715, 41.843378, 4.000000],
            [208.048686, 58.938188, 4.000000],
            [141.275003, 75.679015, 4.000000],
            [129.153940, 96.341677, 4.000000],
            [171.650433, 107.002048, 4.000000],
            [132.349345, 126.871236, 4.000000],
            [126.839728, 164.274674, 4.000000],
            [141.254835, 185.845021, 4.000000],
            [115.238079, 195.496360, 4.000000],
            [112.029856, 241.968132, 4.000000],
            [111.868842, 254.140232, 4.000000],
            [105.147115, 270.556549, 4.000000],
            [157.219033, 271.758651, 4.000000],
            [111.586819, 308.147176, 4.000000],
            [140.692451, 44.984592, 5.000000],
            [207.902882, 59.308253, 5.000000],
            [147.054723, 83.761322, 5.000000],
            [130.830337, 127.783892, 5.000000],
            [144.590946, 186.355798, 5.000000],
            [120.124663, 206.089013, 5.000000],
            [107.402948, 270.128482, 5.000000],
            [159.764117, 270.931039, 5.000000],
            [111.746355, 307.251195, 5.000000]]);

        """
        Decrementing the y and x from Matlab (there they start from 1,
            not like in Python).
        """
        for r in range(pointsGood.shape[0]):
            pointsGood[r, 0] -= 1;
            pointsGood[r, 1] -= 1;

        aZero = np.abs(points - pointsGood);
        common.DebugPrint("aZero = %s" % str(aZero));
        self.assertTrue((aZero < 1e-4).all());



if config.MULTI_SCALE_HARRIS == 0:
    multi_scale_harris = multi_scale_harris_Evangelidis;
elif config.MULTI_SCALE_HARRIS == 1:
    pass


if __name__ == '__main__':
    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    # We use 4 digits precision and suppress using scientific notation.
    np.set_printoptions(precision=4, suppress=True, \
                        threshold=1000000, linewidth=5000);

    unittest.main();

