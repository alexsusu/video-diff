import common
import config
import ReadVideo

import cv2

import numpy as np
import scipy
import sys
import time




def AskFirst():
    print("To speed up the video alignment on future runs, we save intermediate results for later reuse:\n" \
          "   - Harris features and\n" \
          "   - matrices computed in the decision step of temporal alignment.\n" \
          "In case you do NOT want to use them we invite you to " \
          "delete this data from the local folder(s) yourself, otherwise we can obtain WRONG results, " \
          "if the saved data is not corresponding to the videos analyze.\n" \
          "Are you OK to continue and use any of these intermediate results, if any?\n");

    #time.sleep(10);
    return;

    # From http://rosettacode.org/wiki/Keyboard_input/Obtain_a_Y_or_N_response
    try:
        from msvcrt import getch
    except ImportError:
        def getch():
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

    print("Press Y to continue or N to quit: ")
    return

    while True:
        char = getch()
        if char.lower() in ("y", "n"):
            print char
            break

    if char.lower() == "n":
        quit()

    # See also https://stackoverflow.com/questions/1450393/how-do-you-read-from-stdin-in-python:
    #data = sys.stdin.readlines();


if __name__ == '__main__':
    assert len(sys.argv) >= 3;
    if sys.argv[3] == "--preprocess-ref":
        config.PREPROCESS_REFERENCE_VIDEO_ONLY = True;
    elif sys.argv[3] == "--process-query-and-align-videos":
        config.PREPROCESS_REFERENCE_VIDEO_ONLY = False;
    else:
        config.PREPROCESS_REFERENCE_VIDEO_ONLY = "do_all";

    print("config.PREPROCESS_REFERENCE_VIDEO_ONLY = %s" % str(config.PREPROCESS_REFERENCE_VIDEO_ONLY));

    AskFirst();

    # Inspired from https://stackoverflow.com/questions/1520234/how-to-check-which-version-of-numpy-im-using
    print("numpy.version.version = %s" % str(np.version.version));
    print("scipy.version.version = %s" % str(scipy.version.version));
    #scipy.version.version
    np.show_config();
    scipy.show_config();


    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    # We use 7 digits precision and suppress using scientific notation.
    np.set_printoptions(precision=7, suppress=True, \
                        threshold=70000, linewidth=4000);
                        #threshold=7000000, linewidth=4000);
                        #threshold=7000, linewidth=300);
                        #threshold=1000000, linewidth=3000);


    # Inspired from \OpenCV2-Python-Tutorials-master\source\py_tutorials\py_core\py_optimization

    # normally returns True - relates to using the SIMD extensions of x86: SSX, AVX
    common.DebugPrint("cv2.useOptimized() is %s" % str(cv2.useOptimized()));

    if False:
        cv2.setUseOptimized(True);
        cv2.useOptimized();

    """
    From http://docs.opencv.org/modules/core/doc/utility_and_system_functions_and_macros.html#checkhardwaresupport
        CV_CPU_MMX - MMX
        CV_CPU_SSE - SSE
        CV_CPU_SSE2 - SSE 2
        CV_CPU_SSE3 - SSE 3
        CV_CPU_SSSE3 - SSSE 3
        CV_CPU_SSE4_1 - SSE 4.1
        CV_CPU_SSE4_2 - SSE 4.2
        CV_CPU_POPCNT - POPCOUNT
        CV_CPU_AVX - AVX
    """
    if config.OCV_OLD_PY_BINDINGS == False:
        featDict = {cv2.CPU_AVX: "AVX",
                cv2.CPU_MMX: "MMX",
                cv2.CPU_NEON: "NEON",
                cv2.CPU_POPCNT: "POPCNT",
                cv2.CPU_SSE: "SSE",
                cv2.CPU_SSE2: "SSE2",
                cv2.CPU_SSE3: "SSE3",
                cv2.CPU_SSE4_1: "SSE4.1",
                cv2.CPU_SSE4_2: "SSE4.2",
                cv2.CPU_SSSE3: "SSSE3"};

        for feat in featDict:
            res = cv2.checkHardwareSupport(feat);
            print("%s = %d" % (featDict[feat], res));
        #cv2.setUseOptimized(onoff)!!!!

    # "Returns the number of logical CPUs available for the process."
    common.DebugPrint("cv2.getNumberOfCPUs() (#logical CPUs) is %s" % str(cv2.getNumberOfCPUs()));
    common.DebugPrint("cv2.getTickFrequency() is %s" % str(cv2.getTickFrequency()));

    """
    Available only in C++:
    # "getNumThreads - Returns the number of threads used by OpenCV for parallel regions."
    common.DebugPrint("cv2.getNumThreads() (#logical CPUs) is %s" % str(cv2.getNumThreads()));
    """


    videoPathFileNameQ = sys.argv[1]; # input/current video
    videoPathFileNameR = sys.argv[2]; # reference video


    #!!!!TODO: use getopt() to run Evangelidis' or "Alex's" algorithm, etc

    #if True:
    if False:
        import hotshot

        prof = hotshot.Profile("hotshot_edi_stats_Main");
        #prof.runcall(findquads, Points, threshold, reflect_flag);
        prof.runcall(ReadVideo.Main, videoPathFileNameQ, videoPathFileNameR);
        print;
        prof.close();

        """
        from hotshot import stats

        s = stats.load("hotshot_edi_stats_findquads");
        s.sort_stats("time").print_stats();
        #s.print_stats()
        """
    else:
        ReadVideo.Main(videoPathFileNameQ, videoPathFileNameR);

