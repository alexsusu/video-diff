import cv2
import math
import numpy as np
import os
import sys

import common
import config

import Matlab
import multi_scale_harris
import multiscale_quad_tree
import multiscale_quad_retrieval
import multiscale_synchro_decision


harlocsQ = []
harlocsR = []


######## GLOBALS DEF
# No longer that much used
#r_path = "Videos/reference/" # reference sequence directory (database)
#q_path = "Videos/input/" # query sequence

# sychronization parameters
nos = 5; #number of scales
noc = 3000;  #%number of centers (visual words)
# md_threshold=100; % maximum distance between poionts in a quad (defined again below)
st_threshold = 50; # %100 ; % space-time contraint threshold

if True:
    method = 1; # 1: full-search (either DP3 or causal), 2: VD, 3: BoW

    #compute_harris = 1; # set this flag to 1 if harris points need recomputing (e.g. the first time you run the demo)
    #compute_harris = 0; # set this flag to 1 if harris points need recomputing (e.g. the first time you run the demo)

    compute_dictionary = 0; # (USED with BoW and VD method)set this flag to 1 if visual dictionary needs recomputing


# parameters for BoW method
vmf = 4; #% 4: tf-idf weighting, 3: tf weighting. 2: no weighiting, 1: binary bows
const_type = 1; #% 1: spatio-temporal constraint applies before the voting, 2: after the voting (not used in the paper)

cropflag = 0; # % 1: cropped images, 0: non-cropped images (do not change this value)

sequence = "test"; # % this is used by the function multiscale_quad_retrieval (give any name)

#imformat='tiff';
#imformat="jpeg";
imformat = "png";
######## END GLOBALS DEF
######## END GLOBALS DEF
######## END GLOBALS DEF
######## END GLOBALS DEF
######## END GLOBALS DEF


if config.USE_MULTITHREADING == True:
    import threading
    from threading import Thread

    import multiprocessing

    class Globals:
        captureQ = None;
        captureR = None;
        harlocsFolder = None;
        fileNamePrefix = None;
        fileNameExtension = None;
        indexVideo = -1;
    g = Globals();


#def MyImageReadMSH(capture, index):
def MyImageReadMSH(index):
    global g;

    """
    common.DebugPrint("MyImageReadMSH(): initFrame = %d)" % \
                            (config.initFrame[g.indexVideo]));
    index += config.initFrame[g.indexVideo];
    """

    """
    common.DebugPrint("Entered MyImageReadMSH(capture=%s, index=%s)" % \
                            (str(capture), str(index)));
    """
    common.DebugPrint("Entered MyImageReadMSH(index=%s)" % \
                            (str(index)));

    """
    We must reopen the capture device in each different process, otherwise the
        program blocks at the first operation in the "global" capture device.
    """
    if g.harlocsFolder.endswith(config.HARRIS_QUERY_FOLDER_NAME):
        if g.captureQ == None:
            capture = cv2.VideoCapture(sys.argv[1]);
            g.captureQ = capture;
            common.DebugPrint("MyImageReadMSH(): new capture=%s" % \
                                (str(capture)));
        else:
            capture = g.captureQ;
    elif g.harlocsFolder.endswith(config.HARRIS_REFERENCE_FOLDER_NAME):
        if g.captureR == None:
            capture = cv2.VideoCapture(sys.argv[2]);
            g.captureR = capture;
            common.DebugPrint("MyImageReadMSH(): new capture=%s" % \
                                (str(capture)));
        else:
            capture = g.captureR;
    else:
        assert False;

    assert (g.indexVideo == 0) or (g.indexVideo == 1);
    if config.OCV_OLD_PY_BINDINGS:
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, \
                            index);
    else:
        """
        From http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get:
            <<CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be
              decoded/captured next.>>
        """
        capture.set(cv2.CAP_PROP_POS_FRAMES, \
                    index);

    common.DebugPrint("MyImageReadMSH(): after capture.set()");

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
        common.DebugPrint( \
            "MyImageReadMSH(): indexCrt != index --> returning black frame");
        ret = False;
    else:
        #common.DebugPrint("Alex: frameR = %d" % frameR);

        #if myIndex > numFramesR:
        #    break;

        #ret, img = r_path.read();
        ret, img = capture.read();
        #if ret == False:
        #    break;

    #!!!!TODO: think if well
    #assert ret == True;
    if ret == False:
        common.DebugPrint(
                "MyImageReadMSH(index=%d): ret == False --> returning None" % index);
        img = None;
    else:
        common.DebugPrint("MyImageReadMSH(): img.shape = %s" % str(img.shape));
        common.DebugPrint("MyImageReadMSH(): img.dtype = %s" % str(img.dtype));

    #!!!!TODO: I suggest to do the gray conversion at reading, not in multi_scale_harris.py
    if False:
        # In the Matlab code he reads gray/8bpp JPEGs
        imgGray = common.ConvertImgToGrayscale(img);

    if config.VIDEO_FRAME_RESIZE_SCALING_FACTOR != 1:
        # We resize the image
        img = Matlab.imresize(img, \
                        scale=config.VIDEO_FRAME_RESIZE_SCALING_FACTOR);

    common.DebugPrint("Exiting MyImageReadMSH()");
    if False:
        return imgGray;

    return img;


def IterationStandaloneMSH(index):
    #capture = g.capture;
    harlocsFolder = g.harlocsFolder;
    fileNamePrefix = g.fileNamePrefix;
    fileNameExtension = g.fileNameExtension;

    common.DebugPrint("Entered IterationStandaloneMSH(index=%d)" % index);
    #img = MyImageReadMSH(capture, index);
    img = MyImageReadMSH(index);

    im = img;
    pp = multi_scale_harris.multi_scale_harris(im, nos, disp=0); # n=0:nos-1

    if False:
        #harlocs = pp
        harlocs.append(pp);

    multi_scale_harris.StoreMultiScaleHarrisFeatures( \
            harlocsFolder + "/" + fileNamePrefix + "%05d%s" % \
                                (index, fileNameExtension),
            pp);
    if False:
        counter += counterStep;
        # If we try to seek to a frame out-of-bounds frame it gets to the last one
        if config.OCV_OLD_PY_BINDINGS:
            capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, counter);
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, counter);

    return 1;


"""
Computes (or loads if results are saved on the disc)
  the multiscale Harris features.
"""
def ComputeHarlocs(capture, counterStep, folderName, fileNamePrefix,
                    fileNameExtension=".csv", indexVideo=-1):
    print( \
            "Entered ComputeHarlocs(capture=%s, counterStep=%d, folderName=%s, " \
                                     "indexVideo=%d)" % \
                            (str(capture), counterStep, folderName, indexVideo));

    harlocsFolder = config.VIDEOS_FOLDER + "/" + folderName;

    t1 = float(cv2.getTickCount());

    harlocs = [];

    if config.OCV_OLD_PY_BINDINGS:
        numFrames = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT));
    else:
        numFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT));
    common.DebugPrint("ComputeHarlocs(): numFrames = %d" % numFrames);

    if not os.path.exists(harlocsFolder):
        os.makedirs(harlocsFolder);
    else:
        #!!!!TODO: check that the loaded Harlocs are complete - same frame numbers as in the videos
        # Folder with precomputed Harris features exists
        folderContent = os.listdir(harlocsFolder);
        sortedFolderContent = sorted(folderContent);

        for fileName in sortedFolderContent:
            pathFileName = harlocsFolder + "/" + fileName;
            """
            common.DebugPrint("ComputeHarlocs(): pathFileName = %s" % pathFileName);
            common.DebugPrint("ComputeHarlocs(): fileName = %s" % fileName);
            """
            if os.path.isfile(pathFileName) and \
                            fileName.startswith(fileNamePrefix) and \
                            pathFileName.endswith(fileNameExtension):
                common.DebugPrint("ComputeHarlocs(): Loading %s" % pathFileName);
                harrisFeatures = multi_scale_harris.LoadMultiScaleHarrisFeatures(pathFileName);
                harlocs.append(harrisFeatures);

        if config.endFrame[indexVideo] == -1:
            assert (len(harlocs) + config.initFrame[indexVideo]) == numFrames; #!!!!TODO: if condition is NOT met, give a nicer error, or redo computations of Harlocs
        else:
            assert (len(harlocs) + config.initFrame[indexVideo]) == config.endFrame[indexVideo] + 1; #!!!!TODO: if condition is NOT met, give a nicer error, or redo computations of Harlocs

        return harlocs;


    if config.USE_MULTITHREADING == True:
        global g;
        g.captureQ = None; # We need to reopen the capture device in each process, separately
        g.captureR = None; # We need to reopen the capture device in each process, separately
        #g.capture = capture;

        g.harlocsFolder = harlocsFolder;
        g.fileNamePrefix = fileNamePrefix;
        g.fileNameExtension = fileNameExtension;
        g.indexVideo = indexVideo;

        if config.OCV_OLD_PY_BINDINGS:
            frameCount = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT));
        else:
            frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT));
        #listParams = range(frameCount);
        listParams = range(config.initFrame[indexVideo], frameCount, counterStep);

        common.DebugPrint("ComputeHarlocs(): frameCount = %d" % frameCount);

        print("ComputeHarlocs(): Spawning a pool of %d workers" % \
                                config.numProcesses);

        """
        # DEBUG purposes ONLY - since when we use Pool() and call function, if
        #   we have an error in the function the exception reported is very
        #   vague...
        for i in listParams:
            IterationStandaloneMSH(i);
        #import time
        #time.sleep(1000);
        """

        """
        Start worker processes to use on multi-core processor (circumvent
          also the GIL issue).
        """
        pool = multiprocessing.Pool(processes=config.numProcesses);
        print("ComputeHarlocs(): Spawned a pool of %d workers" % \
                                config.numProcesses);

        #res = pool.map(IterationStandaloneMSH, listParams);
        # See https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.pool
        res = pool.map(func=IterationStandaloneMSH, iterable=listParams, \
                        chunksize=1);
        print("Pool.map returns %s" % str(res));

        """
        From https://medium.com/building-things-on-the-internet/40e9b2b36148
         close the pool and wait for the work to finish
        """
        pool.close();
        pool.join();

        #!!!!TODO: do more efficient - don't load the results from the CSV files
        return ComputeHarlocs(capture, counterStep, folderName, \
                              fileNamePrefix, fileNameExtension, indexVideo);
        #return [];

    #indexHarloc = 0;
    while True:
        if config.OCV_OLD_PY_BINDINGS:
            framePos = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES);
        else:
            """
            From http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get:
                <<CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.>>
            """
            framePos = capture.get(cv2.CAP_PROP_POS_FRAMES);
        common.DebugPrint("ComputeHarlocs(): framePos = %d" % framePos);

        counter = int(framePos); #0
        common.DebugPrint("ComputeHarlocs(): counter = %d" % counter);

        ret, img = capture.read();

        common.DebugPrint("ComputeHarlocs(): img = %s" % str(img));

        if False and config.SAVE_FRAMES:
            fileName = config.IMAGES_FOLDER + "/img_%05d.png" % counter;
            if not os.path.exists(fileName):
                #print "dir(img) = %s"% str(dir(img))
                """
                imgCV = cv.fromarray(img)
                cv2.imwrite(fileName, imgCV)
                """
                cv2.imwrite(fileName, img);

        #if ret == False: #MatchFrames.counterQ == 3:
        if (ret == False) or ((counter > numFrames) or \
                             (config.endFrame[indexVideo] != -1 and \
                              counter > config.endFrame[indexVideo])):
            break;

        if config.VIDEO_FRAME_RESIZE_SCALING_FACTOR != 1:
            img = Matlab.imresize(img, \
                            scale=config.VIDEO_FRAME_RESIZE_SCALING_FACTOR);

        im = img;
        pp = multi_scale_harris.multi_scale_harris(im, nos, disp=0); # n=0:nos-1

        #harlocs = pp
        harlocs.append(pp);

        multi_scale_harris.StoreMultiScaleHarrisFeatures( \
                harlocsFolder + "/" + fileNamePrefix + "%05d%s" % \
                                    (counter, fileNameExtension),
                pp);

        counter += counterStep;
        # If we try to seek to a frame out-of-bounds frame it gets to the last one
        if config.OCV_OLD_PY_BINDINGS:
            capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, counter);
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, counter);

        #indexHarloc += 1;

    t2 = float(cv2.getTickCount());
    myTime = (t2 - t1) / cv2.getTickFrequency();
    common.DebugPrint("ComputeHarlocs(): computing the multiscale harlocs " \
                        "took %.6f [sec]" % myTime);

    #common.DebugPrint("ComputeHarlocs(): len(harlocs) = %s" % str(len(harlocs)));

    if False:
        for i, h in enumerate(harlocs):
            #common.DebugPrint("ComputeHarlocs(): len(harlocs[%d]) = %d" % \
            #                                                        (i, len(h)));
            multi_scale_harris.StoreMultiScaleHarrisFeatures(
                    harlocsFolder + "/" + fileNamePrefix + "%05d.txt" % i, h);

    if False:
        common.DebugPrint("ComputeHarlocs(): harlocs = %s" % str(harlocs));

    return harlocs;


#def QuadTreeDecision(captureQ, captureR):
def QuadTreeDecision():
    """
    global r_path, q_path
    common.DebugPrint("QuadTreeDecision(): r_path = %s" % r_path);
    common.DebugPrint("QuadTreeDecision(): q_path = %s" % q_path);
    """
    #global harlocsQ, harlocsR

    common.DebugPrint("Entered QuadTreeDecision().");

    totalT1 = float(cv2.getTickCount());


    r_quadsTree = None;

    """
    Matlab code:
    www=whos('tree');
    if size(www,1)>0
        kdtree_delete(tree);
        clear tree;
    end
    """

    #clear Votes_space H;
    Votes_space = None;
    H = None;

    #['arr_0']

    if r_quadsTree != None:
        # TODO: clear tree
        pass;


    if method == 1:
        #%% search among all reference quads
        common.DebugPrint("\nQuadTreeDecision(): Search among all reference quads...(Tree method)");

        try:
            crossref = np.load("crossref.npz")['arr_0'];
            common.DebugPrint("\nQuadTreeDecision(): Found already precomputed crossref.npz - returning it)");
            return crossref;
        except:
            common.DebugPrintErrorTrace();

        #BOV_flag=0;
        BOV_flag = 0;

        foundFiles = False;
        try:
            Votes_space = np.load("Votes_space.npz")['arr_0'];
            H = np.load("H.npz")['arr_0'];
            foundFiles = True;
        except:
            common.DebugPrintErrorTrace();

        if foundFiles == False:
            """
            Alex: scale s is 1 for original frame resolution and the higher
              we go we have lower image resolutions (we go higher in the
              Guassian pyramid I think).
            """
            #for s=1:nos
            for s in range(1, nos + 1):
                common.DebugPrint("QuadTreeDecision(): Scale %d" % s);

                #md_threshold=round(s*100+100^(log(s)));
                md_threshold = round( s * 100 + pow(100, math.log(s)) );

                #[tree,all_id,all_cen,all_max,all_ori,n_d,all_quads]=multiscale_quad_tree(r_path, md_threshold,s);
                #tree, all_id, all_cen, all_max, all_ori, n_d, all_quads = multiscale_quad_tree.multiscale_quad_tree(r_path, md_threshold, s)
                r_quadsTree, all_id, all_cen, all_max, all_ori, n_d, all_quads = \
                        multiscale_quad_tree.multiscale_quad_tree(harlocsR, \
                                                                md_threshold, s);

                if config.PREPROCESS_REFERENCE_VIDEO_ONLY == True:
                    continue;

                if r_quadsTree == None:
                    continue;

                common.DebugPrint("QuadTreeDecision(): md_threshold = %s" % str(md_threshold));

                #[Votes_space(:,:,s),H(:,:,s)]=multiscale_quad_retrieval(tree, r_path, q_path, md_threshold, st_threshold, all_ori, all_id, all_max, all_cen,nos, s, cropflag, sequence);
                # Votes_space(:,:,s),H(:,:s)  =multiscale_quad_retrieval(tree, r_path, q_path, md_threshold, st_threshold, all_ori, all_id, all_max, all_cen, nos, s, cropflag, sequence)
                #Votes_space[:, :, s - 1], H[:, :,s - 1] = multiscale_quad_retrieval.multiscale_quad_retrieval(r_quadsTree, harlocsR, harlocsQ, md_threshold, st_threshold, all_ori, all_id, all_max, all_cen, nos, s, cropflag, sequence)
                Votes_space_res, H_res = multiscale_quad_retrieval.multiscale_quad_retrieval(r_quadsTree, \
                                            harlocsR, harlocsQ, md_threshold, \
                                            st_threshold, all_ori, all_id, \
                                            all_max, all_cen, nos, s, cropflag, \
                                            sequence);

                if Votes_space == None:
                    Votes_space = np.zeros( (Votes_space_res.shape[0], Votes_space_res.shape[1], nos) );
                    """
                    Inspired from https://stackoverflow.com/questions/17559140/matlab-twice-as-fast-as-numpy
                        BUT doesn't help in this case:
                    Votes_space = np.asfortranarray(np.zeros( (Votes_space_res.shape[0], Votes_space_res.shape[1], nos) ));
                    """
                if H == None:
                    H = np.zeros( (H_res.shape[0], H_res.shape[1], nos), dtype=np.int8 );
                    """
                    Inspired from https://stackoverflow.com/questions/17559140/matlab-twice-as-fast-as-numpy
                        BUT doesn't help in this case:
                    H = np.asfortranarray(np.zeros( (H_res.shape[0], H_res.shape[1], nos) ));
                    """
                Votes_space[:, :, s - 1] = Votes_space_res;
                H[:, :, s - 1] = H_res;

                common.DebugPrint("QuadTreeDecision(): For scale %d: " \
                    "Votes_space_res = %s,\n      H_res = %s" % \
                    (s, str(Votes_space_res), str(H_res)));

                common.DebugPrint("QuadTreeDecision(): For scale %d: " \
                    "Votes_space_res.shape = %s,\n      H_res.shape = %s" % \
                    (s, str(Votes_space_res.shape), str(H_res.shape)));
                #quit();
                #kdtree_delete(tree); # TODO: think if want to delete kdtree
                if config.KDTREE_IMPLEMENTATION == 1:
                    r_quadsTree.release();

        if config.PREPROCESS_REFERENCE_VIDEO_ONLY == True:
            common.DebugPrint("QuadTreeDecision(): Exiting program " \
                              "since we finished preprocessing the reference video");
            common.DebugPrint("QuadTreeDecision(): time before exit = %s" % \
                    common.GetCurrentDateTimeStringWithMilliseconds());
            quit();

        common.DebugPrint("QuadTreeDecision(): Before multiscale_synchro_decision(): " \
                "Votes_space = %s,\n      H = %s" % (str(Votes_space), str(H)));

        try:
            # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
            np.savez_compressed("Votes_space", Votes_space);
            np.savez_compressed("H", H);
        except:
            common.DebugPrintErrorTrace();

        #q_path = [None] * len(harlocsQ);
        numFramesQ = len(harlocsQ);
        #r_path = [None] * len(harlocsR);
        numFramesR = len(harlocsR);

        if config.temporalDecisionType == 1:
            # causal solution - "local"

            #cross=multiscale_synchro_decision(Votes_space, H, q_path, r_path, BOV_flag, cropflag, const_type);
            crossref = multiscale_synchro_decision.causal( \
                        Votes_space, H, numFramesQ, numFramesR, BOV_flag, cropflag, \
                        const_type);

            # str=['save ' q_path 'cross_baseline cross'];
            # eval(str)
        elif config.temporalDecisionType == 0:
            # decision (non-causal solution)

            #[y,x,D,Tback,cross] = dp3(Votes_space, r_path, q_path, BOV_flag);
            y, x, D, Tback, crossref = multiscale_synchro_decision.dp3( \
                                        Votes_space, numFramesR, numFramesQ, BOV_flag);
            #     str=['save ' q_path 'cross_baseline_dp cross'];
            #     eval(str)
    else:
        """
        !!!!TODO: implement if useful VD (or BoW)
          NOTE: see config.py for Evangelidis' comments from email of Apr 14, 2014:
            Basically he argues that:
            - the VD method is similar in quality with the full-search VS
            - BoW is not great.
        """
        assert False; # not implemented

    crossref[:, 1] += config.initFrame[1];

    #myText = "crossref = \n%s" % crossref;
    myText = "";
    for r in range(crossref.shape[0]):
        myText += "  %d  %d\n" % (crossref[r][0], crossref[r][1])
    fOutput = open("crossref.txt", "wt");
    fOutput.write(myText);
    fOutput.close();

    try:
        # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
        np.savez_compressed("crossref", crossref);
    except:
        common.DebugPrintErrorTrace();

    totalT2 = float(cv2.getTickCount());
    myTime = (totalT2 - totalT1) / cv2.getTickFrequency();
    print("QuadTreeDecision() took %.6f [sec]" % (myTime));

    return crossref;


def TemporalAlignment(captureQ, captureR):
    global harlocsQ, harlocsR;

    # if compute_harris == 1:...

    totalT1 = float(cv2.getTickCount());

    #if config.PREPROCESS_REFERENCE_VIDEO_ONLY == True: # This will give error in multiscale_quad_retrieval
    if True:
        # We compute and Store in files the multi-scale Harris features of the reference video
        """
        harlocsR = ComputeHarlocs(captureR, config.counterRStep, \
                            folderName="/harlocs_ref", fileNamePrefix="harloc");
        """
        harlocsR = ComputeHarlocs(captureR, config.counterRStep, \
                            folderName=config.HARRIS_REFERENCE_FOLDER_NAME, \
                            fileNamePrefix=config.HARRIS_FILENAME_PREFIX, \
                            fileNameExtension=config.HARRIS_FILENAME_EXTENSION,
                            indexVideo=1);
        common.DebugPrint("TemporalAlignment(): len(harlocsR) = %s" % str(len(harlocsR)));
        sumNbytes = 0;
        for hr in harlocsR:
            sumNbytes += hr.nbytes;
        common.DebugPrint("TemporalAlignment(): harlocsR.nbytes = %s" % str(sumNbytes));
        #common.DebugPrint("TemporalAlignment(): harlocsR.nbytes = %s" % str(harlocsR.nbytes));


    if config.PREPROCESS_REFERENCE_VIDEO_ONLY == False:
        # We compute and Store in files the multi-scale Harris features of the query video
        """
        Note: if the "harloc" files exist, load directly the features from
                the files without computing them again.
        """
        harlocsQ = ComputeHarlocs(captureQ, config.counterQStep, \
                            folderName=config.HARRIS_QUERY_FOLDER_NAME, \
                            fileNamePrefix=config.HARRIS_FILENAME_PREFIX, \
                            fileNameExtension=config.HARRIS_FILENAME_EXTENSION,
                            indexVideo=0);
        common.DebugPrint("TemporalAlignment(): len(harlocsQ) = %s" % \
                            str(len(harlocsQ)));
        sumNbytes = 0;
        for hq in harlocsQ:
            sumNbytes += hq.nbytes;
        common.DebugPrint("TemporalAlignment(): harlocsQ.nbytes = %s" % str(sumNbytes));

    #res = QuadTreeDecision(captureQ, captureR);
    res = QuadTreeDecision();

    totalT2 = float(cv2.getTickCount());
    myTime = (totalT2 - totalT1) / cv2.getTickFrequency();
    print("TemporalAlignment() took %.6f [sec]" % (myTime));

    return res;




import unittest

class TestSuite(unittest.TestCase):
    def testSynchro(self):
        common.MY_DEBUG_STDOUT = True;
        multiscale_quad_retrieval.DBGPRINT = True;

        #assert config.temporalDecisionType == 1; #1
        """
        # 0 --> global solution (multi-scale dynamic programming - the dp3() function)
        # 1 --> causal (local) syncronization
        """

        if True: # We use dp_Alex (instead of causal, normally)
            config.temporalDecisionType = 0;
            #dp3Orig = dp3;
            multiscale_synchro_decision.dp3 = multiscale_synchro_decision.dp_Alex;

            config.KDTREE_IMPLEMENTATION = 1; # Use OpenCV's KDtree
            config.FLANN_PARAMS = config.FLANN_PARAMS_DETERMINISTIC;

        """
        The cross result from Matlab, for the videos from
                Evangelidis, with step=25 (1fps), when
               applying the ORIGINAL causal() temporal alignment.
        """
        resCrossref = [ \
                    [1001,        2049],
                    [1002,        2049],
                    [1003,        2049],
                    [1004,        2049],
                    [1005,        2049],
                    [1006,        2049],
                    [1007,        2001],
                    [1008,        2001],
                    [1009,        2061],
                    [1010,        2061],
                    [1011,        2061],
                    [1012,        2061],
                    [1013,        2068],
                    [1014,        2068],
                    [1015,        2068],
                    [1016,        2068],
                    [1017,        2068],
                    [1018,        2068],
                    [1019,        2073],
                    [1020,        2073],
                    [1021,        2073],
                    [1022,        2079],
                    [1023,        2079],
                    [1024,        2079],
                    [1025,        2079],
                    [1026,        2079],
                    [1027,        2083],
                    [1028,        2083],
                    [1029,        2083],
                    [1030,        2055],
                    [1031,        2055]];

        resCrossref = np.array(resCrossref);
        resCrossref[:, 0] -= 1001;
        resCrossref[:, 1] -= 2001;

        #common.DebugPrint("resCrossref = %s" % str(resCrossref));
        #crossref = TemporalAlignment(None, None);
        #crossref = QuadTreeDecision(None, None);
        crossref = QuadTreeDecision();
        common.DebugPrint("crossref = %s" % str(crossref));

        aZero = resCrossref - crossref;
        common.DebugPrint("aZero = %s" % str(aZero));

        """
        Note: if we use the OpenCV KD-tree implementation in findquads
            (instead of SciPy implementation) we will have
                different results on rows 5 and 6.
        """
        if config.KDTREE_IMPLEMENTATION == 1:
            common.DebugPrint("The following 2 values we disregard:");
            common.DebugPrint("  aZero[5, 1] = %s" % str(aZero[5, 1]));
            common.DebugPrint("  aZero[6, 1] = %s" % str(aZero[6, 1]));

            aZero[5, 1] = 0;
            aZero[6, 1] = 0;

        self.assertTrue( (aZero == 0).all() );


if __name__ == '__main__':
    # We load the harlocs test data
    import synchro_script_testdata
    harlocsQ = synchro_script_testdata.harlocsQ;
    harlocsR = synchro_script_testdata.harlocsR;

    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(threshold=1000000, linewidth=3000);

    config.PREPROCESS_REFERENCE_VIDEO_ONLY = False;
    #
    config.temporalDecisionType = 0; # Using dp3-Alex (or original dp3)
    #multiscale_synchro_decision.dp3 = multiscale_synchro_decision.dp3Orig;
    #
    #config.temporalDecisionType = 1; # Using causal
    #multiscale_synchro_decision.CAUSAL_DO_NOT_SMOOTH = False; # original causal

    # Using SciPy's KD-tree implementation, NOT the OpenCV one
    config.KDTREE_IMPLEMENTATION = 0;
    unittest.main()

