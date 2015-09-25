PREPROCESS_REFERENCE_VIDEO_ONLY = True
#PREPROCESS_REFERENCE_VIDEO_ONLY = False
#PREPROCESS_REFERENCE_VIDEO_ONLY = "do_all" # I guess this should execute the entire algorithm in one run

OCV_OLD_PY_BINDINGS = True # This is to use the Python bindings to the OpenCV 2.x APIs (still has Python bindings to cv::Mat; does NOT use NumPy that much)
#OCV_OLD_PY_BINDINGS = False # This is to use the new slightly different APIs from latest OpenCV 3.0 (uses NumPy)

USE_GUI = False
#USE_GUI = True

DISPLAY_RIGHT_IMAGE_WITH_DIFF = False
#DISPLAY_RIGHT_IMAGE_WITH_DIFF = True


SAVE_FRAMES = True
#SAVE_FRAMES = False
#SHOW_FRAMES = True


IMAGES_FOLDER = "Img"
FRAME_PAIRS_FOLDER = IMAGES_FOLDER + "/Img_Frames_Pairs" # Not really used anymore since we save for efficiency ONLY matches
FRAME_PAIRS_MATCHES_FOLDER = FRAME_PAIRS_FOLDER + "/Img_Matches"


#if False:
if True:
    # First element is for query/input video; 2nd for the reference video
    initFrame = [0, 0] #928] #1100
    endFrame = [-1, -1]
else:
    # First element is for query/input video; 2nd for the reference video
    initFrame = [0, 2030] #1900]
    endFrame = [-1, 2400]
#initFrameSpatialAlignment #!!!!TODO
"""
In case we perform exhaustive search, we can "perforate" the
    loops/enumeration of frames of both videos and look at one every
    counterQStep and counterRStep, respectively.
"""
counterQStep = 1 #25 #200 #10
counterRStep = 1 #25 #200 #10



##################### EVANGELIDIS ALGORITHM RELATED ###########################
##################### EVANGELIDIS ALGORITHM RELATED ###########################
##################### EVANGELIDIS ALGORITHM RELATED ###########################
##################### EVANGELIDIS ALGORITHM RELATED ###########################
VIDEOS_FOLDER = "Videos"
HARRIS_FILENAME_EXTENSION = ".csv"
HARRIS_FILENAME_PREFIX = "harloc"
HARRIS_QUERY_FOLDER_NAME = "harlocs_query"
HARRIS_REFERENCE_FOLDER_NAME = "harlocs_ref"


USE_EVANGELIDIS_ALGO = True # Use port of Evangelidis' 2013 TPAMI algorithm from Matlab)
#USE_EVANGELIDIS_ALGO = False # Use our simple (~brute-force) video alignment algo

MULTI_SCALE_HARRIS = 0 # a la Evangelidis (slower, but good qualitatively)
#MULTI_SCALE_HARRIS = 1 # using Harris OpenCV primititve - much faster (~10 times) than 0, but poorer results

# When USE_EVANGELIDIS_ALGO = True, we use this to load the JPEGs we feed the
#   Matlab code from Evangelidis
#TESTING_IDENTICAL_MATLAB = True
TESTING_IDENTICAL_MATLAB = False

#!!!!TODO
# Types of temporal alignment:
temporalMethod = 1
#  full-search (working), visual dictionary (NOT implemented), BoW (NOT implemented)

# Decision temporal alignment for the "full-search" temporal alignment method:
temporalDecisionType = 0 #1
# 0 --> global solution (multi-scale dynamic programming - the dp3() function)
# 1 --> causal (local) syncronization

# other spatial alignment: seq2seq, affine !!!!TODO

if True:
    KDTREE_IMPLEMENTATION = 1 # Use OpenCV's Flann KD-tree implementation
    FLANN_KDTREE_INDEX = 1 # 1 = parallel kd_tree implementation
    FLANN_PARAMS = dict(algorithm=FLANN_KDTREE_INDEX, trees=2) # Using 2 parallel KD-trees
    FLANN_PARAMS_DETERMINISTIC = dict(algorithm=0) # Using brute-force? search - see also http://docs.opencv.org/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html#
else:
    KDTREE_IMPLEMENTATION = 0 # Use SciPy's KD-tree implementation


"""
Alex: Unfortunately, there is very little documentation on the Python bindings for the FLANN OpenCV library:
    Help on built-in function flann_Index in module cv2:
    flann_Index(...)
        flann_Index([features, params[, distType]]) -> <flann_Index object>
"""

#USE_MULTITHREADING = True
USE_MULTITHREADING = False
numProcesses = 3; #2;

VISUAL_DIFF_FRAMES = False
#VISUAL_DIFF_FRAMES = True

# We filter out the single-pixel differences between the 2 frames that are very small in colour
MEANINGFUL_DIFF_THRESHOLD = 30; #90; #30


# SPATIAL ALIGNMENT parameters (originally in SpatialAlignment.py / alignment.m)
# Usually nof=2 is good for all resolutions we used
nof = 2; #5; #%number of frames for sub-sequences
cropflag = 0; #% flag for cropped images (use 0)
"""
See email Evangelidis, May 7th, 2014:
  Assuming perfect temporal alignment you come to the problem of spatial alignment.
  Once the algorithm has converged, more iterations do not help.
    However, it depends on the deformation how many iterations ECC needs to converge.
    You can safely choose enough iterations if the complexity is not an issue.
    If the complexity is an issue, multi-level spatial alignment can be employed (see ECC in IATool http://iatool.net).
    Multi-level is better when the deformation is strong (e.g. strong rototranslation) and it
        requires lees iteration owing to the better initialization.
    When the deformation is weak, single-level alignment is fine though.
  Assumme a homography between the two corresponding frames.
    If you do simultaneous capturing it means that you capture exactly the same
      scene with the cameras and if they are static or jointly moving it means
      the the homography is fixed over time. Hence, it may be helpful to use
      more than a single frame (e.g. seq2seq).
    These cases are discussed in the paper.
      This can become extremely helpful when there is dynamic motion in the
        scene and the homography is just an approximation of the transformation.
      Striclty speaking, it is always an approximation when the cameras have
        different viewpoints and a real 3D scene is captured.
  affine_time may be useful when you have different frame-rates in cameras (affine_time assumes spatio-temporal alignment anyway).
"""
# For HD videos with fisheye and ~10% differences in FOV we used levelsECC = 5
# For 1/4^2 HD videos with fisheye and ~10% differences in FOV we used levelsECC = 2-3
levelsECC = 5; #1; #2; #4; #1; #%levels of multi-resolution ECC (use 1 normally, more for stronger roto-tranlation if spatial alignment is not good with 1)

iterECC = 15; #%iterations of ECC
#
# The EPS used for the convergence criteria of the ECC algorithm:
#EPS_ECC = 0.001;
#EPS_ECC = 0.00001;
#EPS_ECC = 0.0001;
#EPS_ECC = 0.00000001;
EPS_ECC = 0.001; # The default EPS used by cv::findTransformECC()

verboseECC = 1; #% save/see the spatial alignment (and if want, uncomment to get a pause per frame)

pixel_select = 0; #% when 1, it considers only pixels around salient points
time_flag = 0; #% when 0, it does only spatial alignment even in seq2seq
weighted_flag = 1; #% when 1, it considers a self-weighted version of ECC, not explained in PAMI paper

affine_time = 0;
seq2seq = 0;

imformat = ".png"; #".jpeg"; #% Output image format

"""
See email from Evangelidis, Apr 14, 2014:
  "This answer depends on your data.
  full-search and VD should perform similar in terms of the synchro part.
  I do not suggest BoW with quads owing to low dimensionality.
  I discuss in the paper the pros and cons of causal and non-causal solution.
  !!!!Re. spatial alignment, if synchro is quite good with retrieval, you can do only spatial alignment (no spatio-temporal).
  If you need spatio-temporal, again it depends on your data.
  E.g. if the homography remains fixed within a subsequence, then seq2seq might be better.
  Affine does not make sense if you have same cameras and if you know the ratio of fps."
"""
USE_ECC_FROM_OPENCV = False # We use the (originally) Matlab implementation from ecc_homo_spacetime.m provided by Evangelidis
#USE_ECC_FROM_OPENCV = True # This requires OpenCV 3.0
assert((USE_ECC_FROM_OPENCV == False) or (OCV_OLD_PY_BINDINGS == False))

VIDEO_FRAME_RESIZE_SCALING_FACTOR = 1
#VIDEO_FRAME_RESIZE_SCALING_FACTOR = 0.5
#VIDEO_FRAME_RESIZE_SCALING_FACTOR = 0.25
# END SPATIAL ALIGNMENT parameters


################### OUR SIMPLER IN-HOUSE ALGORITHM RELATED ####################
################### OUR SIMPLER IN-HOUSE ALGORITHM RELATED ####################
################### OUR SIMPLER IN-HOUSE ALGORITHM RELATED ####################
################### OUR SIMPLER IN-HOUSE ALGORITHM RELATED ####################

############################# TEMPORAL ALIGNMENT #############################
############################# TEMPORAL ALIGNMENT #############################
############################# TEMPORAL ALIGNMENT #############################
############################# TEMPORAL ALIGNMENT #############################

"""
For the temporal alignment algorithm we use:
    USE_EXHAUSTIVE_SEARCH = True
        --> perform exhaustive search: detect and match features for each pair of frames;

    USE_EXHAUSTIVE_SEARCH = False
        --> we use Simulated Annealing: probabilistic general heuristic
            technique to find near-optimal solution much faster than
            exhaustive technique.
"""
USE_EXHAUSTIVE_SEARCH = True


FEATURE_DETECTOR_AND_MATCHER = "orb-flann" # ORB detector with Flann descriptor matcher
#FEATURE_DETECTOR_AND_MATCHER = "sift-flann" # SIFT detector with Flann matcher
#FEATURE_DETECTOR_AND_MATCHER = "orb" # ORB detector with brute-force descriptor matcher
#FEATURE_DETECTOR_AND_MATCHER = "sift" # SIFT detector with brute-force descriptor matcher
#FEATURE_DETECTOR_AND_MATCHER = "surf" # SURF detector with brute-force descriptor matcher

numFeaturesToExtractPerFrame = 1000
#numFeaturesToExtractPerFrame = 400

########################### END TEMPORAL ALIGNMENT ############################
########################### END TEMPORAL ALIGNMENT ############################
########################### END TEMPORAL ALIGNMENT ############################
########################### END TEMPORAL ALIGNMENT ############################





############################## SPATIAL ALIGNMENT ###############################
############################## SPATIAL ALIGNMENT ###############################
############################## SPATIAL ALIGNMENT ###############################
############################## SPATIAL ALIGNMENT ###############################

"""
For the spatial alignment algorithm we use:
"""
#SPATIAL_ALIGNMENT_ALGO = "ECC"
SPATIAL_ALIGNMENT_ALGO = "LK"
#SPATIAL_ALIGNMENT_ALGO = "TEMPORAL_ALIGNMENT_HOMOGRAPHY"
"""
    This is simple homography from the feature extraction matching performed
        for temporal alignment.
"""


"""
#USE_ECC = False
USE_ECC = True # We use Evangelidis' ECC algorithm

USE_Lucas_Kanade = True
#USE_Lucas_Kanade = False
"""
########################## END SPATIAL ALIGNMENT ###############################
########################## END SPATIAL ALIGNMENT ###############################
########################## END SPATIAL ALIGNMENT ###############################
########################## END SPATIAL ALIGNMENT ###############################






################################# CLUSTERING ##################################
################################# CLUSTERING ##################################
################################# CLUSTERING ##################################
################################# CLUSTERING ##################################

"""
This is used to TRIGGER ALARM that an object is not in place w.r.t. the
    reference video.
"""
THRESHOLD_NUM_NONMATCHED_ELEMENTS_IN_CLUSTER = 40

#DISPLAY_PYTHON_CLUSTERING = False
DISPLAY_PYTHON_CLUSTERING = True

