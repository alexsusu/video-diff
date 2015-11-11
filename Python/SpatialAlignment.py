import cv2
import numpy as np
import os

import common
import config
import ecc_homo_spacetime


DONT_USE_MATLAB_FIT_ECC_RECORD_SYNTACTIC_SUGAR = True



if config.USE_MULTITHREADING == True:
    import threading
    from threading import Thread

    import multiprocessing
    class Globals:
        crossref = None;
        captureQ = None;
        captureR = None;
        x0 = None;
        y0 = None;
        start = None;
        refined_crossref = None;
        warp_p = None;
        t = None;
    g = Globals();

#!!!!TODO: just for testing purposes: remove, later, if necessary
"""
Note: /home/asusu/drone-diff_multilevel/stderr_levels_0_9_old
        PicklingError: Can't pickle <type 'function'>: attribute lookup __builtin__.function failed
    http://stackoverflow.com/questions/8804830/python-multiprocessing-pickling-error
      "Here is a list of what can be pickled.
      In particular, functions are only picklable if they are defined at the top-level of a module."
"""
"""
For
    def IterationStandalone(crossref, captureQ, captureR, iWhile):
  says:
      <<PicklingError: Can't pickle <type 'cv2.VideoCapture'>: it's not the same object as cv2.VideoCapture>>
"""
def IterationStandalone(iWhile):
    crossref = g.crossref;
    captureQ = g.captureQ;
    captureR = g.captureR;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("Entered IterationStandalone(): crossref=%s, captureQ=%s, "\
                            "captureR=%s, refined_crossref=%s, warp_p=%s, "
                            "x0=%s, y0=%s, start=%s, t=%d, iWhile=%d." % \
                        (str(crossref), str(captureQ), str(captureR), \
                         str(g.refined_crossref), str(g.warp_p), \
                         str(g.x0), str(g.y0), str(g.start), g.t, iWhile));
        common.DebugPrint("IterationStandalone(): id(g)=%s" % str(id(g)));

    r_path = captureQ;
    q_path = captureR;


    x0 = g.x0;
    y0 = g.y0;
    """
    x0 = crossref[5: -5, 0].T;
    y0 = crossref[5: -5, 1].T;
    """
    """
    x0 = np.array(range(10));
    y0 = np.array(range(10));
    """

    start = g.start;
    #start = y0.T; #% Alex: So start is related to crossref(:,2) from the end of dp3.m
    #x_init = 0;

    refined_crossref = g.refined_crossref;
    #refined_crossref = crossref.copy();
    ##refined_crossref = np.array([]);

    #% fprintf('\nRetrieval using visual dictionary and inverted index list...(VD method)\n');

    """
    if config.USE_ECC_FROM_OPENCV:
        H = np.eye(3, dtype=np.float32); #% use feature matching if you need a good initialization
    else:
        H = np.eye(3); #% use feature matching if you need a good initialization
    warp_p = H; #%initial warp
    """
    warp_p = g.warp_p; #%initial warp

    t = g.t;
    #t = 0; #%initial subframe correction

    if DONT_USE_MATLAB_FIT_ECC_RECORD_SYNTACTIC_SUGAR == False:
        fit = [];

    #if config.pixel_select == 1:
    #    config.weighted_flag = 0;

    try:
        common.DebugPrint("SpatialAlignment.IterationStandalone(iWhile=%d)\n" % \
                                                                    iWhile);

        """
        if i < 5:
            tic
        """

        if config.affine_time == 0:
            if config.seq2seq == 0:
                common.DebugPrint("Alex: NOT affine temporal model, NOT seq2seq");

                #% frame-to-subframe scheme (one query frame against a reference subsequence
                if DONT_USE_MATLAB_FIT_ECC_RECORD_SYNTACTIC_SUGAR:
                    fitecc = ecc_homo_spacetime.ecc_homo_spacetime( \
                                                img_index=start[iWhile],
                                                tmplt_index=x0[iWhile],
                                                p_init=warp_p,
                                                t0=t,
                                                n_iters=config.iterECC,
                                                levels=config.levelsECC,
                                                r_capture=r_path,
                                                q_capture=q_path,
                                                nof=config.nof,
                                                time_flag=config.time_flag,
                                                weighted_flag=config.weighted_flag,
                                                pixel_select=config.pixel_select,
                                                mode="ecc",
                                                imformat=config.imformat,
                                                save_image=config.verboseECC);
                else:
                    fit[iWhile].ecc = ecc_homo_spacetime.ecc_homo_spacetime( \
                                                img_index=start[iWhile],
                                                tmplt_index=x0[iWhile],
                                                p_init=warp_p,
                                                t0=t,
                                                n_iters=config.iterECC,
                                                levels=config.levelsECC,
                                                r_capture=r_path,
                                                q_capture=q_path,
                                                nof=nof,
                                                time_flag=config.time_flag,
                                                weighted_flag=config.weighted_flag,
                                                pixel_select=config.pixel_select,
                                                mode="ecc",
                                                imformat=config.imformat,
                                                save_image=config.verboseECC);
            else: # seq2seq == 1, affine_time == 0
                common.DebugPrint("Alex: NOT affine temporal model, seq2seq == 1");

                #% sequence-to-sequence alignment (one temporal parameter)
                fit[iWhile].ecc = ecc_homo_spacetime_seq(start[iWhile],
                    x0[iWhile], warp_p, t,
                    iterECC, levels, r_path, q_path, nof, time_flag,
                    weighted_flag, pixel_select, config.imformat, verbose);
        else: #% affine temporal model (two parameters) for the case frame-to-subframe
            common.DebugPrint("Alex: Affine temporal model\n");

            fit[iWhile].ecc = ecc_homo_spacetime.ecc_homo_affine_spacetime( \
                                                img_index=start[iWhile],
                                                tmplt_index=x0[iWhile],
                                                p_init=warp_p,
                                                t0=t,
                                                n_iters=config.iterECC,
                                                levels=config.levelsECC,
                                                r_capture=r_path,
                                                q_capture=q_path,
                                                nof=config.nof,
                                                time_flag=config.time_flag,
                                                weighted_flag=config.weighted_flag,
                                                pixel_select=config.pixel_select,
                                                mode="ecc",
                                                imformat=config.imformat,
                                                save_image=config.verboseECC);

        """
        if i < 5:
            toc;
        """

        iterECC0 = config.iterECC - 1;
        if fitecc[0][iterECC0].t == None:
            while iterECC0 > 0:
                iterECC0 -= 1;
                if fitecc[0][iterECC0].t != None:
                    break;

        #!!!!TODO: understand ecc_homo_spacetime - should we use fitecc[config.levelsECC - 1][iterECC0].t instead of fitecc[0][iterECC0].t? (when levelsECC != 1)?

        #% synchronization correction
        #%synchro with subframe correction
        #refined_crossref(i,2)=refined_crossref(i,2)+fit(i).ecc(1,iter).t;
        if config.USE_ECC_FROM_OPENCV:
            pass;
        else:
            if DONT_USE_MATLAB_FIT_ECC_RECORD_SYNTACTIC_SUGAR:
                refined_crossref[iWhile, 1] = refined_crossref[iWhile, 1] + \
                                            fitecc[0][iterECC0].t;
                                            #fitecc[0][config.iterECC - 1].t;
            else:
                refined_crossref[iWhile, 1] = refined_crossref[iWhile, 1] + \
                                            fit[iWhile].ecc[iterECC0].t;
                                            #fit[iWhile].ecc[0, config.iterECC - 1].t;
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("Exiting IterationStandalone(iWhile=%d)" % iWhile);
    except:
        common.DebugPrintErrorTrace();





def SpatialAlignmentEvangelidis(crossref, captureQ, captureR):
    """
    % Alex:
    % We entere a bogus crossref, since for the 720x480 MIT video Matlab
    %    crashed with out of memory - it ended up using 1GB of data
    %crossref=zeros(1594,2);
    %for i=1:size(crossref,1)
    %    crossref(i,1)=1000+i;
    %    crossref(i,2)=2000+i;
    %end
    %crossref
    """

    #fprintf(1, 'Alex: Entered alignment_script\n');

    #% Alex: tic starts a stopwatch timer to measure performance. The function records the internal time at execution of the tic command. Display the elapsed time with the toc function.
    #tic

    #% alignment parameters

    """
    % Alex: From [TPAMI_2013] paper:
    % "Given a pair (m; n), we consider the temporally local
    %subsequence In-mu;...,In+mu where mu is a small integer. After
    %defining Phi(), we look for the image warped in space and time
    %from the above subsequence that aligns with Im. To this
    %end, we extend the ECC alignment algorithm [3] to the
    %space-time dimensions, i.e., the extended scheme estimates
    %the spatiotemporal parameterspthat maximize the correla-
    %tion coefficient between the input frame Iq(x) and the
    %warped reference subframeIr(Phi(x_hat; p)."
    """
    #nof = 5; #%number of frames for sub-sequences
    #cropflag = 0; #% flag for cropped images (use 0)
    #iterECC = 15; #%iterations of ECC
    #levelsECC = 1; #%levels of multi-resolution ECC (use 1)

    #verboseECC = 1; #% save/see the spatial alignment (and if want, uncomment to get a pause per frame)

    if config.TESTING_IDENTICAL_MATLAB:
        #q_path = "video_data/input/";
        #r_path = "video_data/reference/";
        q_path = "Videos/input/";
        r_path = "Videos/reference/";

        if not os.path.exists(q_path):
            os.makedirs(q_path);

        if not os.path.exists(r_path):
            os.makedirs(r_path);
    else:
        q_path = captureQ;
        r_path = captureR;

    #% reject a few frames in the beginning and at the end, because of the sub-sequence
    #% Alex: crossref comes from the synchro_script (via the global scope)
    #x0=crossref(6:end-5,1)';
    x0 = crossref[5: -5, 0].T;
    #y0=crossref(6:end-5,2)';
    y0 = crossref[5: -5, 1].T;

    #imformat = "png"; #"jpeg"; #%image format

    start = y0.T; #% Alex: So start is related to crossref(:,2) from the end of dp3.m
    x_init = 0;

    refined_crossref = crossref.copy();

    #% fprintf('\nRetrieval using visual dictionary and inverted index list...(VD method)\n');

    if config.USE_ECC_FROM_OPENCV:
        H = np.eye(3, dtype=np.float32); #% use feature matching if you need a good initialization
    else:
        H = np.eye(3); #% use feature matching if you need a good initialization
    warp_p = H; #%initial warp
    t = 0; #%initial subframe correction

    if DONT_USE_MATLAB_FIT_ECC_RECORD_SYNTACTIC_SUGAR == False:
        fit = [];

    #pixel_select=0; #% when 1, it considers only pixels around salient points
    #time_flag=0; #% when 0, it does only spatial alignment even in seq2seq
    #weighted_flag=1; #% when 1, it considers a self-weighted version of ECC, not explained in PAMI paper

    #%fprintf(1, 'Am here before\n');


    if config.pixel_select == 1:
        config.weighted_flag = 0;


    """
    % !!!!NEEDS NORMALLY MEX: concat_3d, etc (UNLESS on Win64, for which we have binaries for the MEXes)
    % when affine_time is 1, an affine temporal model is considered with the frame-to-subframe scheme
    % Alex: affine temporal model means two parameters, for the case frame-to-subframe
    %    Alex: if affine_time == 0 --> it has one temporal parameter. We can also use here sequence-to-sequence alignment
    % Alex: From [TPAMI_2013] paper: "there will be a global affine temporal transformation
    %       t=alpha * t_hat + tau t_hat determines corre-spondences between the
    %       indices t and t_hat, regardless of scene content or camera motion."
    """
    #config.affine_time = 0;

    """
    % Alex: From [TPAMI_2013] paper:
    %"Note, however, that if the input frames are weakly textured,
    %sequence-to-sequence schemes may be preferable to image-to-sequence counterparts."
    % when seq2seq is 1, it considers a sequence-to-sequence alignment (seq2seq in PAMI paper)
    % Alex: the PAMI paper seems to be [1] Y. Caspi and M. Irani,
                Spatio-Temporal Alignment of Se-quences,
                IEEE Trans. Pattern Analysis and Machine Intelligence,
                vol. 24, no. 11, pp. 1409-1424, Nov. 2002
    """
    #config.seq2seq = 0;

    #% if seq2seq==1:
    #%     pixel_select = 0;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("SpatialAlignmentEvangelidis(): crossref.shape = %s" % \
                                                        str(crossref.shape));
        common.DebugPrint("SpatialAlignmentEvangelidis(): crossref = %s" % str(crossref));

        common.DebugPrint("SpatialAlignmentEvangelidis(): x0.shape = %s" % str(x0.shape));
        common.DebugPrint("SpatialAlignmentEvangelidis(): x0 = %s" % str(x0));

        common.DebugPrint("SpatialAlignmentEvangelidis(): y0.shape = %s" % str(y0.shape));
        common.DebugPrint("SpatialAlignmentEvangelidis(): y0 = %s" % str(y0));

        common.DebugPrint("SpatialAlignmentEvangelidis(): start.shape = %s" % \
                                                        str(start.shape));
        common.DebugPrint("SpatialAlignmentEvangelidis(): (used to generate reference " \
                                "frame to read) start = %s" % str(start));

    if config.USE_MULTITHREADING == True:
        global g;
        g.crossref = crossref;
        g.captureQ = captureQ;
        g.captureR = captureR;
        g.x0 = x0;
        g.y0 = y0;
        g.start = start;
        g.refined_crossref = refined_crossref;
        g.warp_p = warp_p;
        g.t = t;

        # We consider only this case:
        assert config.affine_time == 0;
        assert config.seq2seq == 0;

	if common.MY_DEBUG_STDOUT:
            common.DebugPrint("SpatialAlignmentEvangelidis(): id(g)=%s" % str(id(g)));
            common.DebugPrint("SpatialAlignmentEvangelidis(): g.crossref=%s, " \
                                    "g.captureQ=%s, g.captureR=%s." % \
                                    (g.crossref, g.captureQ, g.captureR));
        """
        Start worker processes to use on multi-core processor (able to run
           in parallel - no GIL issue if each core has it's own VM)
        """
        pool = multiprocessing.Pool(processes=config.numProcesses);
        """
        common.DebugPrint("SpatialAlignment(): Spawning a new " \
                            "IterationStandalone(iWhile=%d) thread" % iWhile);
        """
        print("SpatialAlignment(): Spawned a pool of %d workers" % \
                                config.numProcesses);
        if False:
            # The multithreaded solution is not great, since it encounters the GIL limitation
            #NOT_TODO: this is not working here, outside the while loop
            t = Thread(target=IterationStandalone, args=(iWhile, ));
            t.start();
        common.DebugPrint("SpatialAlignment(): __name__ = %s" % str(__name__));

        if True:
            #listParams = [(crossref, captureQ, captureR) for x in range(1, 8 + 1)];
            #listParams = range(1, x0.size + 1);
            listParams = range(0, x0.size);  #!!!!TODO: use counterStep

            #% listParams is the vector with query frame IDs. y0 with the corresponding ref frames.
            # See https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.pool
            res = pool.map(func=IterationStandalone, iterable=listParams, \
                            chunksize=1);
            print("Pool.map returns %s" % str(res)); #x0.size + 1

            """
            From https://medium.com/building-things-on-the-internet/40e9b2b36148
              close the pool and wait for the work to finish
            """
            pool.close();
            pool.join();
            """
            We passed refined_crossref to the workers, so it should be
                updated here. !!!!TODO: check if so
            """
            """
            !!!!TODO: check to see if refined_crossref was updated properly by the workers

            !!!!TODO: seems you need to use muliprocessing.Value - see http://stackoverflow.com/questions/14124588/python-multiprocessing-shared-memory:
                <<Python's multithreading is not suitable for CPU-bound tasks
                   (because of the GIL), so the usual solution in that case is
                    to go on multiprocessing.
                  However, with this solution you need to explicitly share
                    the data, using multiprocessing.Value and
                     multiprocessing.Array.>>

            From http://stackoverflow.com/questions/10721915/shared-memory-objects-in-python-multiprocessing :
                <<If you use an operating system that uses copy-on-write fork() semantics (like any common unix), then as long as you never alter your data structure it will be available to all child processes without taking up additional memory. You will not have to do anything special (except make absolutely sure you don't alter the object).
                The most efficient thing you can do for your problem would be to pack your array into an efficient array structure (using numpy or array), place that in shared memory, wrap it with multiprocessing.Array, and pass that to your functions. This answer shows how to do that.
                If you want a writeable shared object, then you will need to wrap it with some kind of synchronization or locking. multiprocessing provides two methods of doing this: one using shared memory (suitable for simple values, arrays, or ctypes) or a Manager proxy, where one process holds the memory and a manager arbitrates access to it from other processes (even over a network).
                The Manager approach can be used with arbitrary Python objects, but will be slower than the equivalent using shared memory because the objects need to be serialized/deserialized and sent between processes.
                There are a wealth of parallel processing libraries and approaches available in Python. multiprocessing is an excellent and well rounded library, but if you have special needs perhaps one of the other approaches may be better.>>

            From http://en.wikipedia.org/wiki/Fork_%28system_call%29 :
                <<In Unix systems equipped with virtual memory support
                    (practically all modern variants), the fork operation
                    creates a separate address space for the child.
                  The child process has an exact copy of all the memory
                    segments of the parent process, though if copy-on-write
                    semantics are implemented, the physical memory need
                    not be actually copied.
                  Instead, virtual memory pages in both processes may refer to
                    the same pages of physical memory until one of them writes
                    to such a page: then it is copied.>>

             From http://en.wikipedia.org/wiki/Copy-on-write
                <<Copy-on-write finds its main use in virtual memory operating systems; when a process creates a copy of itself, the pages in memory that might be modified by either the process or its copy are marked copy-on-write. When one process modifies the memory, the operating system's kernel intercepts the operation and copies the memory thus a change in the memory of one process is not visible in another's.>>
            """

            return refined_crossref;

    #i=1;
    # IMPORTANT NOTE: We substitute i - 1 --> iWhile (since array numbering
    #     starts with 0, not like in Matlab from 1)
    iWhile = 0;

    #% x0 is the vector with query frame IDs. y0 with the corresponding ref frames.
    #while(i<(numel(x0)+1))
    while iWhile < x0.size:
        try:
            common.DebugPrint("SpatialAlignmentEvangelidis(): Iteration(iWhile=%d)\n" % iWhile);

            """
            if i < 5:
                tic
            """

            #if ~affine_time
            if config.affine_time == 0:
                if config.seq2seq == 0:
                    common.DebugPrint("SpatialAlignmentEvangelidis(): NOT affine temporal model, NOT seq2seq");

                    #% We do spatial alignment for query frame x0(i) and ref frame start(i) = y0(i)
                    #% frame-to-subframe scheme (one query frame against a reference subsequence
                    if DONT_USE_MATLAB_FIT_ECC_RECORD_SYNTACTIC_SUGAR:
                        fitecc = ecc_homo_spacetime.ecc_homo_spacetime( \
                                                    img_index=start[iWhile],
                                                    tmplt_index=x0[iWhile],
                                                    p_init=warp_p,
                                                    t0=t,
                                                    n_iters=config.iterECC,
                                                    levels=config.levelsECC,
                                                    r_capture=r_path,
                                                    q_capture=q_path,
                                                    nof=config.nof,
                                                    time_flag=config.time_flag,
                                                    weighted_flag=config.weighted_flag,
                                                    pixel_select=config.pixel_select,
                                                    mode="ecc",
                                                    imformat=config.imformat,
                                                    save_image=config.verboseECC);
                    else:
                        fit[iWhile].ecc = ecc_homo_spacetime.ecc_homo_spacetime( \
                                                    img_index=start[iWhile],
                                                    tmplt_index=x0[iWhile],
                                                    p_init=warp_p,
                                                    t0=t,
                                                    n_iters=config.iterECC,
                                                    levels=config.levelsECC,
                                                    r_capture=r_path,
                                                    q_capture=q_path,
                                                    nof=config.nof,
                                                    time_flag=config.time_flag,
                                                    weighted_flag=config.weighted_flag,
                                                    pixel_select=config.pixel_select,
                                                    mode="ecc",
                                                    imformat=config.imformat,
                                                    save_image=config.verboseECC);
                else:
                    common.DebugPrint("SpatialAlignmentEvangelidis(): NOT affine temporal model, seq2seq == 1");

                    #% sequence-to-sequence alignment (one temporal parameter)
                    fit[iWhile].ecc = ecc_homo_spacetime_seq(start[iWhile],
                        x0[iWhile], warp_p, t,
                        iterECC, levels, r_path, q_path, nof, time_flag,
                        weighted_flag, pixel_select, config.imformat, verbose);
            else: #% affine temporal model (two parameters) for the case frame-to-subframe
                common.DebugPrint("SpatialAlignmentEvangelidis(): Affine temporal model\n");

                fit[iWhile].ecc = ecc_homo_spacetime.ecc_homo_affine_spacetime( \
                                                    img_index=start[iWhile],
                                                    tmplt_index=x0[iWhile],
                                                    p_init=warp_p,
                                                    t0=t,
                                                    n_iters=config.iterECC,
                                                    levels=config.levelsECC,
                                                    r_capture=r_path,
                                                    q_capture=q_path,
                                                    nof=config.nof,
                                                    time_flag=config.time_flag,
                                                    weighted_flag=config.weighted_flag,
                                                    pixel_select=config.pixel_select,
                                                    mode="ecc",
                                                    imformat=config.imformat,
                                                    save_image=config.verboseECC);

            """
            if i < 5:
                toc;
            """

            #% synchronization correction
            #%synchro with subframe correction
            #refined_crossref(i,2)=refined_crossref(i,2)+fit(i).ecc(1,iter).t;
            if config.USE_ECC_FROM_OPENCV:
                pass;
            else:
                iterECC0 = config.iterECC - 1;
                if fitecc[0][iterECC0].t == None:
                    while iterECC0 > 0:
                        iterECC0 -= 1;
                        if fitecc[0][iterECC0].t != None:
                            break;
                if fitecc[0][iterECC0].t == None:
                    continue;

                #!!!!TODO: understand ecc_homo_spacetime - should we use fitecc[config.levelsECC - 1][iterECC0].t instead of fitecc[0][iterECC0].t? (when levelsECC != 1)?

                if DONT_USE_MATLAB_FIT_ECC_RECORD_SYNTACTIC_SUGAR:
                    refined_crossref[iWhile, 1] = refined_crossref[iWhile, 1] + \
                                                fitecc[0][iterECC0].t;
                else:
                    refined_crossref[iWhile, 1] = refined_crossref[iWhile, 1] + \
                                                fit[iWhile].ecc[0, iterECC0].t;
                common.DebugPrint(
                    "SpatialAlignment(): Finished iteration iWhile=%d" % \
                                                                    iWhile);

            common.DebugPrint("SpatialAlignmentEvangelidis(): warp_p " \
                                  "(at end of iteration) = %s" % \
                                                            str(warp_p));
        # Inspired from http://effbot.org/zone/stupid-exceptions-keyboardinterrupt.htm
        except (KeyboardInterrupt, SystemExit):
            raise;
        except:
            common.DebugPrintErrorTrace();
            quit();

        iWhile += 1;

        """
        We stop the spatial alignment after generating the
            first "pair" of visuals.
        """
        #break;

        #print("SpatialAlignment(): while loop: iWhile = %d" % iWhile);


    #% fit struct containes the results of the spatial alignment.

    #toc
    return refined_crossref;

    #% visualize_space_time_refinement2



def TestGoProCameraVideos():
    """
    Running Spatial Alignment, given the precomputed
        temporal alignment, the cross array.
    """
    path = "/home/asusu/drone-diff_Videos/GoPro_clips/2HD_cuts_from_Lucian/";
    videoPathFileNameQ = path + "GOPR7269_50-55.MP4";
    videoPathFileNameR = path + "GOPR7344_90-95.MP4";

    import ReadVideo
    captureQ, frameCountQ, resVidQ = ReadVideo.OpenVideoCapture(videoPathFileNameQ, 0);
    common.DebugPrint("Alex: frameCountQ = %d" % frameCountQ);

    captureR, frameCountR, resVidR = ReadVideo.OpenVideoCapture(videoPathFileNameR, 1);
    common.DebugPrint("Alex: frameCountR = %d" % frameCountR);

    """
    crossref = np.array([ \
        [  7, 120],
        [  8, 120],
        [  9, 120],
        [ 10, 120],
        [ 11, 120],
        [ 12, 120],
        [ 13, 120],
        [ 14, 120],
        [ 15, 120],
        [ 16, 120],
        [ 17, 120],
        [ 18, 120],
        [ 19, 114],
        [ 20,   4],
        [ 21,   4],
        [ 22,   5],
        [ 23,   5],
        [ 24,   5],
        [ 25,   4],
        [ 26,   4],
        [ 27,   4],
        [ 28,   4],
        [ 29,   4],
        [ 30,   4],
        [ 31,   4],
        [ 32,   4],
        [ 33,   4],
        [ 34,   4],
        [ 35,   4],
        [ 36,   4],
        [ 37,   4],
        [ 38,   0],
        [ 39,   0],
        [ 40,   0],
        [ 41,   5],
        [ 42,   5],
        [ 43,   5],
        [ 44,   5],
        [ 45,   5],
        [ 46,   5],
        [ 47,   5],
        [ 48,   5],
        [ 49,   5],
        [ 50,   5],
        [ 51,   5],
        [ 52,   5],
        [ 53,   5],
        [ 54,   5],
        [ 55,   6],
        [ 56,   6],
        [ 57,   6],
        [ 58,   6],
        [ 59,   6],
        [ 60,   6],
        [ 61,   6],
        [ 62,   6],
        [ 63,   6],
        [ 64,   6],
        [ 65,  78],
        [ 66,   6],
        [ 67,   6],
        [ 68,   6],
        [ 69,   6],
        [ 70,   6],
        [ 71,   6],
        [ 72,   6],
        [ 73,   6],
        [ 74,   7],
        [ 75,   7],
        [ 76,   7],
        [ 77,   7],
        [ 78,   7],
        [ 79,   7],
        [ 80,  13],
        [ 81,  12],
        [ 82,  12],
        [ 83,  12],
        [ 84,  12],
        [ 85,  16],
        [ 86,  16],
        [ 87,  16],
        [ 88,  15],
        [ 89,  22],
        [ 90,  22],
        [ 91,  22],
        [ 92,  22],
        [ 93,  22],
        [ 94,  22],
        [ 95,  22],
        [ 96,  22],
        [ 97,  22],
        [ 98,  22],
        [ 99,  32],
        [100,  32],
        [101,  32],
        [102,  32],
        [103,  32],
        [104,  33],
        [105,  34],
        [106,  34],
        [107,  34],
        [108,  38],
        [109,  38],
        [110,  38],
        [111,  38],
        [112,  43],
        [113,  43],
        [114,  43],
        [115,  43],
        [116,  43],
        [117,  45],
        [118,  41],
        [119,  60],
        [120,  60],
        [121,  60],
        [122,  60],
        [123,  60],
        [124,  60],
        [125,  60],
        [126,  60],
        [127,  60],
        [128,  60],
        [129,  60],
        [130,  60],
        [131,  60],
        [132,  60],
        [133,  60],
        [  0,   4],
        [  1,   4],
        [  2,   4],
        [  3,   4],
        [  4,   4],
        [  5,   4],
        [  6,   4]]);
        """
    crossref = np.array([ \
        [  7, 120],
        [  8, 120],
        [  9, 120],
        [ 10, 120],
        [ 11, 120],
        [ 12, 120],
        [ 13, 120],
        [ 14, 120],
        [ 15, 120],
        [ 16, 120],
        [ 17, 120],
        [ 18, 120],
        [ 19, 114],
        [ 20,   4],
        [ 21,   4],
        [ 22,   5],
        [ 23,   5],
        [ 24,   5],
        [ 25,   4],
        [ 26,   4],
        [ 27,   4],
        [ 28,   4],
        [ 29,   4],
        [ 30,   4],
        [ 31,   4],
        [ 32,   4],
        [ 33,   4],
        [ 34,   4],
        [ 35,   4],
        [ 36,   4],
        [ 37,   4],
        [ 38,   0],
        [ 39,   0],
        [ 40,   0],
        [ 41,   5],
        [ 42,   5],
        [ 43,   5],
        [ 44,   5],
        [ 45,   5],
        [ 46,   5],
        [ 47,   5],
        [ 48,   5],
        [ 49,   5],
        [ 50,   5],
        [ 51,   5],
        [ 52,   5],
        [ 53,   5],
        [ 54,   5],
        [ 55,   6],
        [ 56,   6],
        [ 57,   6],
        [ 58,   6],
        [ 59,   6],
        [ 60,   6],
        [ 61,   6],
        [ 62,   6],
        [ 63,   6],
        [ 64,   6],
        [ 65,  78],
        [ 66,   6],
        [ 67,   6],
        [ 68,   6],
        [ 69,   6],
        [ 70,   6],
        [ 71,   6],
        [ 72,   6],
        [ 73,   6],
        [ 74,   7],
        [ 75,   7],
        [ 76,   7],
        [ 77,   7],
        [ 78,   7],
        [ 79,   7],
        [ 80,  13],
        [ 81,  12],
        [ 82,  12],
        [ 83,  12],
        [ 84,  12],
        [ 85,  16],
        [ 86,  16],
        [ 87,  16],
        [ 88,  15],
        [ 89,  22],
        [ 90,  22],
        [ 91,  22],
        [ 92,  22],
        [ 93,  22],
        [ 94,  22],
        [ 95,  22],
        [ 96,  22],
        [ 97,  22],
        [ 98,  22],
        [ 99,  32],
        [100,  32],
        [101,  32],
        [102,  32],
        [103,  32],
        [104,  33],
        [105,  34],
        [106,  34],
        [107,  34],
        [108,  38],
        [109,  38],
        [110,  38],
        [111,  38],
        [112,  43],
        [113,  43],
        [114,  43],
        [115,  43],
        [116,  43],
        [117,  45],
        [118,  41],
        [119,  60],
        [120,  60],
        [121,  60],
        [122,  60],
        [123,  60],
        [124,  60],
        [125,  60],
        [126,  60],
        [127,  60],
        [128,  60],
        [129,  60],
        [130,  60],
        [131,  60],
        [132,  60],
        [133,  60],
        [  0,   4],
        [  1,   4],
        [  2,   4],
        [  3,   4],
        [  4,   4],
        [  5,   4],
        [  6,   4]]);

    res = SpatialAlignmentEvangelidis(crossref, captureQ, captureR);

    print("Corrected cross from SpatialAlignmentEvangelidis() = %s" % \
                                                                str(res));



import unittest

class TestSuite(unittest.TestCase):
    def runTest(self):
        pass # just to be able to run the test without instantiating the unit test framework, to profile with hotshot, etc


    def testSpatialAlignmentEvangelidis(self):
        """
        We test Evangelidis Spatial Alignment, given the precomputed
            temporal alignment, the cross array.
        """
        videoPathFileNameQ = "Videos/input.avi";
        videoPathFileNameR = "Videos/reference.avi";

        """
        We want to use the same JPEGs Matlab is using - since the algorithm
         is somewhat sensitive to quantisation errors.
        """
        assert config.TESTING_IDENTICAL_MATLAB == True;

        import ReadVideo
        captureQ, frameCountQ, resVidQ = ReadVideo.OpenVideoCapture(videoPathFileNameQ, 0);
        common.DebugPrint("Alex: frameCountQ = %d" % frameCountQ);

        captureR, frameCountR, resVidR = ReadVideo.OpenVideoCapture(videoPathFileNameR, 1);
        common.DebugPrint("Alex: frameCountR = %d" % frameCountR);

        """
        The cross result we obtain for the videos from
                Evangelidis, with step=25 (1fps).
        """
        crossref = [[ 0, 48],
                 [ 1, 48],
                 [ 2, 48],
                 [ 3, 48],
                 [ 4, 48],
                 [ 5, 48],
                 [ 6,  0],
                 [ 7,  0],
                 [ 8, 60],
                 [ 9, 60],
                 [10, 60],
                 [11, 60],
                 [12, 67],
                 [13, 67],
                 [14, 67],
                 [15, 67],
                 [16, 67],
                 [17, 67],
                 [18, 72],
                 [19, 72],
                 [20, 72],
                 [21, 78],
                 [22, 78],
                 [23, 78],
                 [24, 78],
                 [25, 78],
                 [26, 82],
                 [27, 82],
                 [28, 82],
                 [29, 54],
                 [30, 54]];

        crossref = np.array(crossref);

        #SpatialAlignmentEvangelidis(cross, captureQ=None, captureR=None);
        res = SpatialAlignmentEvangelidis(crossref, captureQ, captureR);

        common.DebugPrint("Corrected cross from SpatialAlignmentEvangelidis() = %s" % \
                                                                    str(res));

        """
        aZero = resCross - cross;
        #print("aZero = %s" % str(aZero));
        self.assertTrue( (aZero == 0).all() );
        """


if __name__ == '__main__':
    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    # We use 4 digits precision and suppress using scientific notation.
    np.set_printoptions(precision=4, suppress=True, \
                        threshold=1000000, linewidth=3000);

    TestGoProCameraVideos();
    quit();
    #unittest.main()

    """
    We instantiate the TestSuite, just because we want to individually
        run the test
    """
    ts = TestSuite();

    #if False:
    if True:
        # No profiling performed
        ts.testSpatialAlignmentEvangelidis();
    else:
        import hotshot

        prof = hotshot.Profile("hotshot_edi_stats_spatial_alignment");

        """
        It appears the stats generated like this are NOT very reliable, since
            they were performed during a unit-test that itself quit() program
            when exiting unittest.main().
        #prof.runcall(unittest.main);
        """
        prof.runcall(ts.testSpatialAlignmentEvangelidis);

        print;
        prof.close();


        from hotshot import stats

        s = stats.load("hotshot_edi_stats_spatial_alignment");
        s.sort_stats("time").print_stats();
        #s.print_stats();

