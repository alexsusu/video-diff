import cv2
import os
import traceback
import shutil
import sys

# Find max number of matched features with Simulated annealing
import common
import config

if config.OCV_OLD_PY_BINDINGS == True:
    import cv


if config.USE_EVANGELIDIS_ALGO == False:
    import MatchFrames
    import SimAnneal



captureQ = None
frameCountQ = None
captureR = None
frameCountR = None

"""
widthQ = None
widthR = None
heightQ = None
heightR = None
"""
resVideoQ = (-1, -1)
resVideoR = (-1, -1)

# Do performance benchmarking
def Benchmark():
    global captureQ, frameCountQ
    global captureR, frameCountR

    # Doing benchmarking
    while True:
        if config.OCV_OLD_PY_BINDINGS:
            frame1 = captureQ.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        else:
            frame1 = captureQ.get(cv2.CAP_PROP_POS_FRAMES)
        common.DebugPrint("Alex: frame1 = %d" % frame1)

        MatchFrames.counterQ = int(frame1) #0

        common.DebugPrint("Alex: MatchFrames.counterQ = %d" % MatchFrames.counterQ)

        retQ, imgQ = captureQ.read()
        if retQ == False:
            break

        if False:
            grayQ = common.ConvertImgToGrayscale(imgQ)

        MatchFrames.Main_img1(imgQ, MatchFrames.counterQ)

    # 36.2 secs (38.5 secs with Convert to RGB)
    common.DebugPrint("Alex: time after Feature Extraction of all frames of " \
                "video 1 = %s" % GetCurrentDateTimeStringWithMilliseconds())

    while True:
        if config.OCV_OLD_PY_BINDINGS:
            frameR = captureR.get(cv2.cv.CV_CAP_PROP_POS_FRAMES);
        else:
            frameR = captureR.get(cv2.CAP_PROP_POS_FRAMES);

        common.DebugPrint("Alex: frameR = %d" % frameR);

        MatchFrames.counterR = int(frameR); #0

        common.DebugPrint("Alex: counterR = %d" % (MatchFrames.counterR));

        retR, imgR = captureR.read();
        if retR == False:
            break;

        if False:
            grayR = common.ConvertImgToGrayscale(imgR);

        MatchFrames.Main_img2(imgR, MatchFrames.counterR);

    # Note: 47.2 secs (56.7 secs with Convert to RGB)
    common.DebugPrint("Alex: time after Feature Extraction of all frames of " \
                    "video 2 and (FLANN?) matching once for each frame = %s" % \
                    GetCurrentDateTimeStringWithMilliseconds());

    quit();



def OpenVideoCapture(videoPathFileName, videoType): # videoType = 0 --> query (input), 1 --> reference
    # OpenCV can read AVIs (if no ffmpeg support installed it can't read MP4, nor 3GP, nor FLVs with MPEG compression)
    # From http://answers.opencv.org/question/6/how-to-readwrite-video-with-opencv-in-python/

    if False:
        """
        We get the following error when trying to open .3gp or .flv.

        OpenCV Error: Bad flag (parameter or structure field)
            (Unrecognized or unsupported array type)
            in unknown function,
            file ..\..\..\src\opencv\modules\core\src\array.cpp, line 2482
        """
        capture = cv2.VideoCapture("2010_06_22_16_05_29_1.3gp");
        #capture = cv2.VideoCapture("1WF2gHYmuFg.flv")

    """
    Unfortunately, normally cv2.VideoCapture() continues even if it does not
        find videoPathFileName
    """
    assert os.path.isfile(videoPathFileName);

    # From http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-videocapture

    capture = cv2.VideoCapture(videoPathFileName);
    # Inspired from https://stackoverflow.com/questions/16703345/how-can-i-use-opencv-python-to-read-a-video-file-without-looping-mac-os

    if config.OCV_OLD_PY_BINDINGS:
        frameCount = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT));
    else:
        frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT));

    if config.OCV_OLD_PY_BINDINGS:
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, config.initFrame[videoType]);
    else:
        capture.set(cv2.CAP_PROP_POS_FRAMES, config.initFrame[videoType]);
    #captureQ.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frameCountQ / 2)

    if config.OCV_OLD_PY_BINDINGS:
        width = capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH);
        height = capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT);
        fps = capture.get(cv2.cv.CV_CAP_PROP_FPS);
        codec = capture.get(cv2.cv.CV_CAP_PROP_FOURCC);
    else:
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH);
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT);
        fps = capture.get(cv2.CAP_PROP_FPS);
        codec = capture.get(cv2.CAP_PROP_FPS);

    assert width < 32767; # we use np.int16
    assert height < 32767; # we use np.int16
    """
    common.DebugPrint("Video '%s' has resolution %dx%d, %d fps and " \
            "%d frames" % \
            (videoPathFileName,
            capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), \
            capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT), \
            capture.get(cv2.cv.CV_CAP_PROP_FPS), \
            frameCount));
    """
    duration = frameCount / fps;
    print("Video '%s' has resolution %dx%d, %.2f fps and " \
            "%d frames, duration %.2f secs, codec=%s" % \
            (videoPathFileName, width, height, fps, \
             frameCount, duration, codec));

    steps = [config.counterQStep, config.counterRStep];
    usedFrameCount = frameCount / steps[videoType]; #!!!!TODO: take into account also initFrame
    assert not((videoType == 0) and (usedFrameCount <= 10));
    common.DebugPrint("We use video '%s', with %d frames, from which we use ONLY %d\n" % \
                     (videoPathFileName, frameCount, usedFrameCount));
    """
    CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    CV_CAP_PROP_FPS Frame rate.
    """
    resolution = (width, height);

    return capture, frameCount, resolution;

    """
    # cv2.VideoCapture() continues even if it does not find videoPathFileNameB
    assert os.path.isfile(videoPathFileNameB)
    captureR = cv2.VideoCapture(videoPathFileNameB)

    #print "Alex: dir(captureQ.read) = %s" % (str(dir(captureQ.read)))
    #print "Alex: help(captureQ.read) = %s" % (str(help(captureQ.read)))

    if config.OCV_OLD_PY_BINDINGS:
        frameCountR = int(captureR.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        frameCountR = int(captureR.get(cv2.CAP_PROP_FRAME_COUNT))
    common.DebugPrint("Alex: frameCountR = %d" % frameCountR)
    """


"""
Note: the extension of the videoPathFileName needs to be the same as the fourcc,
    otherwise, gstreamer, etc can return rather criptic error messages.
"""
def WriteVideoCapture(videoPathFileName, folderName):
    # OpenCV can read only AVIs - not 3GP, nor FLVs with MPEG compression
    # From http://answers.opencv.org/question/6/how-to-readwrite-video-with-opencv-in-python/

    #assert os.path.isfile(videoPathFileName);

    #assert False; # UNFINISHED

    # From http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videowriter-videowriter
    #capture = cv2.VideoWriter(videoPathFileName); # Gives error: <<TypeError: Required argument 'fourcc' (pos 2) not found>>

    # Inspired from http://stackoverflow.com/questions/14440400/creating-a-video-using-opencv-2-4-0-in-python
    #writer = cv.CreateVideoWriter("out.avi", CV_FOURCC("M", "J", "P", "G"), fps, frame_size, True)
    if False:
        writer = cv.CreateVideoWriter("out.avi",
                                  cv.CV_FOURCC("M", "J", "P", "G"),
                                  fps, frameSize, True);
    else:
        videoWriter = None;

    folderContent = os.listdir(folderName);
    sortedFolderContent = sorted(folderContent);

    for fileName in sortedFolderContent:
        pathFileName = folderName + "/" + fileName;
        if os.path.isfile(pathFileName) and \
                            fileName.lower().endswith("_good.png"):
            common.DebugPrint("ComputeHarlocs(): Loading %s" % pathFileName);
            img = cv2.imread(pathFileName);
            assert img != None;

            if videoWriter == None:
                common.DebugPrint("img.shape = %s" % str(img.shape));
                # From http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#saving-a-video
                # See also http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
                # WRITES 0 BYTES IN THE VIDEO: vidFourcc = cv2.VideoWriter_fourcc('M','J','P','G');

                # See also http://www.fourcc.org/codecs.php
                vidFourcc = cv2.VideoWriter_fourcc(*'XVID');

                videoWriter = cv2.VideoWriter(filename=videoPathFileName, \
                                            fourcc=vidFourcc, fps=10, \
                                            frameSize=(img.shape[1], img.shape[0]));
                if not videoWriter:
                    common.DebugPrint("Error in creating video writer");
                    sys.exit(1);

            #cv.WriteFrame(writer, img);
            videoWriter.write(img);

    videoWriter.release();
    common.DebugPrint("Finished writing the video");
    return;

    height, width, layers =  img1.shape;

    video = cv2.VideoWriter("video.avi", -1, 1, (width, height));

    video.write(img1);

    video.release();

    resolution = (width, height);

    # From http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videowriter-write
    capture.write(im);

    return;

    if config.OCV_OLD_PY_BINDINGS:
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0);
    else:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0);

    if config.OCV_OLD_PY_BINDINGS:
        frameCount = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT));
    else:
        frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT));






"""
def ReadFrame(capture, ):
    SynchroEvangelidis(captureQ, captureR);
    return;

    # Allocate numFeaturesMatched
    numFeaturesMatched = [None] * numFramesQ
    for i in range(numFramesQ):
        numFeaturesMatched[i] = [-2000000000] * numFramesR

    while True:
        if config.OCV_OLD_PY_BINDINGS:
            frameQ = captureQ.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        else:
            frameQ = captureQ.get(cv2.CAP_PROP_POS_FRAMES)
        common.DebugPrint("Alex: frameQ = %d" % frameQ)

        counterQ = int(frameQ) #0
        common.DebugPrint("Alex: counterQ = %d" % counterQ)

        ret1, imgQ = captureQ.read()

        if False and config.SAVE_FRAMES:
            fileName = config.IMAGES_FOLDER + "/imgQ_%05d.png" % counterQ
            if not os.path.exists(fileName):
                #print "dir(imgQ) = %s"% str(dir(imgQ));

                #imgQCV = cv.fromarray(imgQ);
                #cv2.imwrite(fileName, imgQCV);

                cv2.imwrite(fileName, imgQ);

        #if ret1 == False: #MatchFrames.counterQ == 3:
        if (ret1 == False) or (counterQ > numFramesQ):
            break;

        #I don't need to change to gray image if I do NOT do
        #    explore_match() , which requires gray to
        #    concatenate the 2 frames together.
        if False:
        #if True:
            #common.ConvertImgToGrayscale(imgQ)
            #gray1 = common.ConvertImgToGrayscale(imgQ)
            imgQ = common.ConvertImgToGrayscale(imgQ)

        ComputeFeatures1(imgQ, counterQ) #TODO: counterQ already visible in module MatchFrames

        # We set the video stream captureR at the beginning
        if config.OCV_OLD_PY_BINDINGS:
            captureR.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0) #900)
        else:
            captureR.set(cv2.CAP_PROP_POS_FRAMES, 0) #900)

        # Start time profiling for the inner loop
        t1 = float(cv2.getTickCount())

        #TODO: counterQ already visible in module MatchFrames
        TemporalAlignment(counterQ, frameQ, captureR, \
                                numFramesR, numFeaturesMatched, fOutput)

        # Measuring how much it takes the inner loop
        t2 = float(cv2.getTickCount())
        myTime = (t2 - t1) / cv2.getTickFrequency()
        common.DebugPrint(
            "Avg time it takes to complete a match (and to perform " \
            "INITIAL Feat-Extract) = %.6f [sec]" % \
            (myTime / (numFramesR / config.counterRStep)) )

        counterQ += config.counterQStep
        # If we try to seek to a frame out-of-bounds frame it gets to the last one
        if config.OCV_OLD_PY_BINDINGS:
            captureQ.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, counterQ)
        else:
            captureQ.set(cv2.CAP_PROP_POS_FRAMES, counterQ)

    common.DebugPrint("numFeaturesMatched = %s" % str(numFeaturesMatched))
"""


def Main(videoPathFileNameQ, videoPathFileNameR):
    global captureQ, frameCountQ
    global captureR, frameCountR
    global resVideoQ, resVideoR

    if config.USE_EVANGELIDIS_ALGO == False:
        if not os.path.exists(config.IMAGES_FOLDER):
            os.makedirs(config.IMAGES_FOLDER);
        if not os.path.exists(config.FRAME_PAIRS_FOLDER):
            os.makedirs(config.FRAME_PAIRS_FOLDER);
        if not os.path.exists(config.FRAME_PAIRS_MATCHES_FOLDER):
            os.makedirs(config.FRAME_PAIRS_MATCHES_FOLDER);

    totalT1 = float(cv2.getTickCount());

    #if False:
    if True:
        #print "dir(cv) = %s" % str(dir(cv))
        if config.OCV_OLD_PY_BINDINGS == True:
            common.DebugPrint("dir(cv) = %s" % str(dir(cv)));
        common.DebugPrint("dir(cv2) = %s" % str(dir(cv2)));
    #print "cv2._INPUT_ARRAY_GPU_MAT = %s" % str(cv2._INPUT_ARRAY_GPU_MAT)
    #sys.stdout.flush()

    if config.USE_EVANGELIDIS_ALGO == False:
        fOutput = open("output.txt", "w")
        print >>fOutput, "Best match for frames from (input/current) video A w.r.t. reference video B:"


    captureQ, frameCountQ, resVideoQ = OpenVideoCapture(videoPathFileNameQ, 0);
    captureR, frameCountR, resVideoR = OpenVideoCapture(videoPathFileNameR, 1);

    common.DebugPrint("Main(): frameCountQ = %d" % frameCountQ);
    common.DebugPrint("Main(): frameCountR = %d" % frameCountR);

    """
    In case the videos have different resolutions an error will actually take
        place much longer when using Evangelidis' algorithm, in spatial
        alignment, more exactly in Matlab.interp2():

      File "/home/asusu/drone-diff/Backup/2014_03_25/Matlab.py", line 313, in interp2
        V4 = np.c_[V[1:, 1:] * xM[:-1, :-1] * yM[:-1, :-1], nanCol1];
        ValueError: operands could not be broadcast together with shapes (719,1279) (239,319)
    """
    assert resVideoQ == resVideoR;

    #numInliersMatched = None

    if config.USE_EVANGELIDIS_ALGO == False:
        SimAnneal.LIMIT = frameCountR;
        SimAnneal.captureR = captureR;
        #SimAnneal.lenA = frameCountR;

    print("ReadVideo.Main(): time before PreMain() = %s" % \
                    common.GetCurrentDateTimeStringWithMilliseconds());

    if config.USE_EVANGELIDIS_ALGO == False:
        #MatchFrames.PreMain(nFramesQ=1000, nFramesR=1000)
        MatchFrames.PreMain(nFramesQ=frameCountQ, nFramesR=frameCountR);

    common.DebugPrint("Main(): time after PreMain() = %s" % \
                    common.GetCurrentDateTimeStringWithMilliseconds());

    if False:
        Benchmark();

    ################ Here we start the (main part of the) algorithm

    # Distinguish between Alex's alignment algo and Evangelidis's algo
    if config.USE_EVANGELIDIS_ALGO:
        #import synchro_script
        #synchro_script.SynchroEvangelidis(captureQ, captureR);
        import VideoAlignmentEvangelidis
        VideoAlignmentEvangelidis.AlignVideos(captureQ, captureR);
    else:
        MatchFrames.ProcessInputFrames(captureQ, captureR, fOutput);

    if config.USE_GUI:
        cv2.destroyAllWindows();

    if not config.USE_EVANGELIDIS_ALGO:
        fOutput.close();

    captureQ.release();
    captureR.release();

    totalT2 = float(cv2.getTickCount());
    myTime = (totalT2 - totalT1) / cv2.getTickFrequency();
    #common.DebugPrint("ReadVideo.Main() took %.6f [sec]" % (myTime));
    print("ReadVideo.Main() took %.6f [sec]" % (myTime));



if __name__ == '__main__':
    """
    # Inspired from \OpenCV2-Python-Tutorials-master\source\py_tutorials\py_core\py_optimization

    # normally returns True - relates to using the SIMD extensions of x86: SSX, AVX
    common.DebugPrint("cv2.useOptimized() is %s" % str(cv2.useOptimized()));

    if False:
        cv2.setUseOptimized(True);
        cv2.useOptimized();
    """
    #Main(None, None);

    #WriteVideoCapture(videoPathFileName="MIT_drive.xvid", folderName=sys.argv[1]);
    WriteVideoCapture(videoPathFileName="MIT_drive.avi", folderName=sys.argv[1]);

