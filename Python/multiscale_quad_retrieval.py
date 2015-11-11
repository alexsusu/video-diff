# !!!!TODO: takeout DBGPRINT - it's already replaced with common.MY_DEBUG_STDOUT (I think):

import math
import numpy as np
from numpy import linalg as npla

import cv2

import common
import config
import findquads
import spatial_consistency
import Matlab


DBGPRINT = False
#DBGPRINT = True

#FILTER = False
FILTER = True

USE_GPS_COORDINATES = False

if config.USE_MULTITHREADING == True:
    import threading
    from threading import Thread

    import multiprocessing

    class Globals:
        r_quadsTree = None;
        r_harlocs = None;
        q_harlocs = None;
        md_threshold = None;
        st_threshold = None;
        all_ori = None;
        all_id = None;
        all_max = None;
        all_cen = None;
        nos = None;
        scale_index = None;
        cropflag = None;
        sequence = None;
        RD_start = None;
        RD_end = None;
        MAXDIS = None;
        MAXORI = None;
        tolers = None;
    g = Globals();


"""
When parallelizing multiscale_quad_retrieval(), we will obtain
  slightly different results for crossref (etc) - see crossref.txt.
  Note that when running in serial (without a POOL), we obtain the same result ALWAYS.
  (less relevant: if we run only 1 process in pool instead of 3, we get
     fewer changes in crossref.txt - NOT sure if this means something).
   It appears the reason is the fact I am running the FLANN KD-tree implementation,
     which is an approximate NN search library, employing randomization
     (see http://answers.opencv.org/question/32664/flannbasedmatcher-returning-different-results/).
     I guess the random number sequence, when having the same seed, evolves
        differently when serial VS in parallel
       threads, so the results tend to be different.
"""
def IterationStandaloneMQR(queryFrame):
    r_quadsTree = g.r_quadsTree;
    r_harlocs = g.r_harlocs;
    q_harlocs = g.q_harlocs;
    md_threshold = g.md_threshold;
    st_threshold = g.st_threshold;
    all_ori = g.all_ori;
    all_id = g.all_id;
    all_max = g.all_max;
    all_cen = g.all_cen;
    nos = g.nos;
    scale_index = g.scale_index;
    cropflag = g.cropflag;
    sequence = g.sequence;
    RD_start = g.RD_start;
    RD_end = g.RD_end;
    MAXDIS = g.MAXDIS;
    MAXORI = g.MAXORI;
    tolers = g.tolers;
    """
    common.DebugPrint( \
              "Entered IterationStandaloneMQR(): crossref=%s, captureQ=%s, "\
                        "captureR=%s, refined_crossref=%s, warp_p=%s, "
                        "x0=%s, y0=%s, start=%s, t=%d, iWhile=%d." % \
                    (str(crossref), str(captureQ), str(captureR), \
                         str(g.refined_crossref), str(g.warp_p), \
                         str(g.x0), str(g.y0), str(g.start), g.t, iWhile));
    common.DebugPrint("IterationStandalone(): id(g)=%s" % str(id(g)));
    """

    # tic

    """
    str1=['load ' q_path QD(q).name]
    eval(str1)
    """

    """
    We make pp reference the desired multiharloc list for the query video
        frame queryFrame
    """
    pp = q_harlocs[queryFrame];
    #pp = np.array(pp);

    #common.DebugPrint("multiscale_quad_retrieval(): pp = %s" % str(pp));

    """
    Alex: for the query frame queryFrame we retrieve, for scale scale_index, the
        harris features in var points.
      Then we build the quads from points.
      Then for each quad (4 float values) we query the corresponding scale
        kd-tree, and we get the indices.
        Then we build the histogram and compute idf, ....!!!!

     Note: scale is 1 for original frame resolution and the higher
        we go we have lower image resolutions (we go higher in the
        Guassian pyramid I think).
    """
    #[qout,qcen,qmaxdis,qori]=findquads(pp(pp(:,3)==scale_index,1:2),md_threshold,0);
    points = pp[pp[:, 2] == scale_index, 0:2];
    qout, qcen, qmaxdis, qori = findquads.findquads(points, md_threshold, 0);

    common.DebugPrint("multiscale_quad_retrieval(): queryFrame = %d, " \
                        "qout.shape = %s" % (queryFrame, str(qout.shape)));

    # disp([num2str(q) ' of ' num2str(length(QD)) ' -> ' num2str(size(qout,1)) ' quads'])

    #space_xy=zeros(size(qcen,1),2*length(RD))+nan;
    #space_xy = np.zeros( (qcen.shape[0], 2 * len(RD)) ) + np.nan;
    space_xy = np.zeros( (qcen.shape[0], 2 * len(r_harlocs)) ) + np.nan;

    #     votes=zeros(length(RD),1)
    #votes=zeros(length(RD),length(tolers));
    #votes = np.zeros( (len(RD), 1) );
    votes = np.zeros( (len(r_harlocs), 1) );

    #nep = np.array([]);
    #m_points = np.array([]);

    assert isinstance(tolers, float);

    """
    We substitute queryFrameQuad - 1 with queryFrameQuad, since we want
        to number arrays from 0 (not from 1 like in Matlab).
    """
    #for queryFrameQuad in range(1, qout.shape[0] + 1):
    for queryFrameQuad in range(qout.shape[0]):
        """
        Matlab's polymorphism is really bugging here: although it's
            normally a float, tolers is considered to be a size 1 vector...
            so len(tolers) == 1
        """
        #for tol_i in range(1, len(tolers) + 1):
        #    tol = tolers[tol_i - 1]
        """
        We substitute tol_i - 1 with tol, since we want
            to number arrays from 0 (not from 1 like in Matlab).
        """
        #for tol_i in range(1, 1 + 1):
        for tol_i in range(1):
            tol = tolers;

            #common.DebugPrint("multiscale_quad_retrieval(): qout[i - 1, :] = %s" % str(qout[i - 1, :]))

            #% default for first PAMI with tol= 0.1 approximately

            # NOTE: SciPy's KDTree finds a few more results, in some cases,
            #    than the Matlab code from Evangelidis.

            #idx, di = kdtree_ball_query(tree, qout(i, :), tol)
            #idx, distKD = kdtree_ball_query(tree, qout[i - 1, :], tol)
            #idx, di = tree.query(x=xQuery, k=4)
            #resPoints = [data[i] for i in resBallIndices]
            # tol is a scalar representing the radius of the ball
            if config.KDTREE_IMPLEMENTATION == 0:
                idx = r_quadsTree.query_ball_point(qout[queryFrameQuad, :], tol);
            elif config.KDTREE_IMPLEMENTATION == 1:
                #pt = qout[queryFrameQuad - 1, :].astype(np.float32);
                pt = qout[queryFrameQuad, :];
                pt = np.array([[pt[0], pt[1], pt[2], pt[3]]], dtype=np.float32);
                retval, idx, dists = r_quadsTree.radiusSearch( \
                                            query=pt, \
                                            radius=(tol**2), \
                                            maxResults=NUM_MAX_ELEMS, \
                                            params=search_params);
                if common.MY_DEBUG_STDOUT and DBGPRINT:
                    """
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                    "retval (number NNs) = %s" % str(retval));
                    """
                    common.DebugPrint( \
                        "multiscale_quad_retrieval(): radiusSearch's retval " \
                        "(at queryFrame=%d, queryFrameQuad=%d) is %d\n" % (queryFrame, queryFrameQuad, retval));
                idx = idx[0];
                dists = dists[0];
                idx = idx[: retval];
                dists = dists[: retval];

            if common.MY_DEBUG_STDOUT and DBGPRINT:
                print("multiscale_quad_retrieval(): " \
                        "qout[queryFrameQuad, :] = %s" % str(qout[queryFrameQuad, :]));
                print("multiscale_quad_retrieval(): " \
                                    "idx = %s" % str(idx));
                print("multiscale_quad_retrieval(): " \
                                    "tol = %s" % str(tol));
                if config.KDTREE_IMPLEMENTATION == 0:
                    print("multiscale_quad_retrieval(): " \
                            "r_quadsTree.data[idx] = %s" % \
                            str(r_quadsTree.data[idx]));

            # We print the distances to the points returned in idx
            a = qout[queryFrameQuad, :];
            if False: #!!!! This is just for debugging purposes
                for myI, index in enumerate(idx):
                    b = r_quadsTree.data[index];
                    """
                    if False:
                        common.DebugPrint("multiscale_quad_retrieval(): distance to " \
                            "%d point (%s) inside ball = %.4f" % \
                            (myI, str(b), npla.norm(a - b)));
                    """
            idx = np.array(idx);


            #if False:
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("multiscale_quad_retrieval(): " \
                            "all_max.shape = %s" % str(all_max.shape));
                common.DebugPrint("multiscale_quad_retrieval(): " \
                            "qmaxdis.shape = %s" % str(qmaxdis.shape));
                common.DebugPrint("multiscale_quad_retrieval(): " \
                                    "qmaxdis = %s" % str(qmaxdis));
                common.DebugPrint("multiscale_quad_retrieval(): " \
                                    "qori.shape = %s" % str(qori.shape));
                common.DebugPrint("multiscale_quad_retrieval(): " \
                                    "qori = %s" % str(qori));

            #dis_idx=abs(qmaxdis(i)-all_max(idx))<MAXDIS;
            if len(idx) == 0:
                # NOT A GOOD IDEA: continue;
                #idx = np.array([]);
                dis_idx = np.array([]);
                ori_idx = np.array([]);
            else:
                #if False:
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "queryFrameQuad = %s" % str(queryFrameQuad));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                        "all_max[idx] = %s" % str(all_max[idx]));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                        "qmaxdis[queryFrameQuad] = %s" % str(qmaxdis[queryFrameQuad]));

                dis_idx = np.abs(qmaxdis[queryFrameQuad] - all_max[idx]) < MAXDIS;

                #if False:
                if common.MY_DEBUG_STDOUT:
                    """
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "idx = %s" % str(idx));
                    """
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "dis_idx = %s" % str(dis_idx));

                #idx=idx(dis_idx)
                idx = idx[dis_idx];

                #if False:
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                        "idx (after idx = idx[dis_idx]) = %s" % str(idx));

                #ori_idx=abs(qori(i)-all_ori(idx))<MAXORI;
                ori_idx = np.abs(qori[queryFrameQuad] - all_ori[idx]) < MAXORI;

                #if False:
                if common.MY_DEBUG_STDOUT:
                    """
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "all_ori = %s" % str(all_ori));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                            "qori[queryFrameQuad] = %s" % str(qori[queryFrameQuad]));

                    """
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "ori_idx = %s" % str(ori_idx));

                #idx=idx(ori_idx);
                idx = idx[ori_idx];


            # IMPORTANT ###################################################
            # IMPORTANT ###################################################
            # IMPORTANT ###################################################
            #% spatio-temporal consistency
            # IMPORTANT ###################################################
            # IMPORTANT ###################################################
            # IMPORTANT ###################################################

            #if numel(idx) > 0:
            if idx.size > 0:
                # Normally cropflag == 0
                if cropflag == 0:
                    dy = qcen[queryFrameQuad, 0] - all_cen[idx, 0];
                    dx = qcen[queryFrameQuad, 1] - all_cen[idx, 1];

                    #D=dy.^2+dx.^2;
                    D = dy**2 + dx**2;

                    co_idx = D < pow(st_threshold, 2);

                    idx = idx[co_idx];
                else:
                    """
                    We substitute iii - 1 with iii, since we want
                        to number arrays from 0 (not from 1 like in Matlab).
                    """
                    #for iii in range(1, len(idx) + 1):
                    for iii in range(len(idx)):
                        #space_xy(i,(all_id(idx(iii))-RD_start)*2+1:(all_id(idx(iii))-RD_start)*2+2) = all_cen(idx(iii),:)
                        space_xy[queryFrameQuad, \
                                (all_id[idx[iii]] - RD_start) * 2: (all_id[idx[iii] - 1] - RD_start) * 2 + 1] = \
                                all_cen[idx[iii], :]

                #hh=hist(all_id(idx),RD_start:RD_end);
                # It has to be an np.array because we multiply it with a scalar
                histoRange = np.array(range(RD_start, RD_end + 1));
                hh = Matlab.hist(x=all_id[idx], binCenters=histoRange);

                #if False:
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "hh = %s" % (str(hh)));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "hh.shape = %s" % (str(hh.shape)));

                    """
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "all_id = %s" % (str(all_id)));
                    """
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                "all_id.shape = %s" % (str(all_id.shape)));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                "idx = %s" % (str(idx)));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                "idx.shape = %s" % (str(idx.shape)));

                # % nz can be computed more optimally
                #nz=find(hh~=0); # nz can be computed more optimally
                # np.nonzero() always returns a tuple, even if it contains 1 element since hh has only 1 dimension
                nz = np.nonzero(hh != 0)[0];
                #if False:
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "nz = %s" % (str(nz)));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "nz.shape = %s" % (str(nz.shape)));

                #if numel(nz) > 0:
                if nz.size > 0:
                    #%%----text-retrieval-like
                    #votes(nz, tol_i) = votes(nz, tol_i) + log10(length(RD) / (length(nz)))^2 #Note: log10(a)^2 means (log10(a))^2 #PREVIOUSLY
                    #myVal = pow(math.log10(float(len(RD)) / len(nz)), 2);
                    myVal = pow(math.log10(float(len(r_harlocs)) / len(nz)), 2);

                    #if False:
                    if common.MY_DEBUG_STDOUT:
                        """
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "len(RD) = %d" % len(RD));
                        """
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "len(r_harlocs) = %d" % len(r_harlocs));
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "len(nz) = %d" % len(nz));
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "myVal = %.5f" % myVal);

                    # PREVIOUSLY
                    votes[nz, tol_i] = votes[nz, tol_i] + myVal;
                    #   votes(nz)=votes(nz)+log10(length(RD)/(length(nz)));
                    #   votes(nz)=votes(nz)+1;

    #if False:
    if common.MY_DEBUG_STDOUT:
        """
        common.DebugPrint("multiscale_quad_retrieval(): " \
                "Votes_space.shape = %s" % (str(Votes_space.shape)));
        common.DebugPrint("multiscale_quad_retrieval(): " \
                "votes.shape = %s" % (str(votes.shape)));
        """

        common.DebugPrint("multiscale_quad_retrieval(): " \
                            "votes.shape = %s" % (str(votes.shape)));
        common.DebugPrint("multiscale_quad_retrieval(): " \
                            "votes = %s" % (str(votes)));

    return (queryFrame, np.ravel(votes));

    # NOT performing these in each worker - the central dispatcher will do these
    if False:
        #Votes_space(:,q)=votes;
        # Gives: "ValueError: output operand requires a reduction, but reduction is not enabled"
        #Votes_space[:, queryFrame - 1] = votes;
        Votes_space[:, queryFrame] = np.ravel(votes);

        if cropflag == 0:
            HH[:, queryFrame] = 1;
        else:
            """
            HH[:, queryFrame] = spatial_consistency.spatial_consistency(space_xy, \
                                        qcen, len(RD), st_threshold, cropflag);
            """
            HH[:, queryFrame] = spatial_consistency.spatial_consistency(space_xy, \
                                        qcen, len(r_harlocs), st_threshold, cropflag);





"""
From http://www.mathworks.com/help/matlab/matlab_prog/symbol-reference.html:
    Dot-Dot-Dot (Ellipsis) - ...
    A series of three consecutive periods (...) is the line continuation operator in MATLAB.
      Line Continuation
      Continue any MATLAB command or expression by placing an ellipsis at the end of the line to be continued:
"""

NUM_MAX_ELEMS = 100000;
search_params = dict(checks=1000000000); # Gives fewer results than scipy's tree.query_ball_point when we have 65K features

# returns Votes_space, HH
# Alex: r_harlocs and q_harlocs are the corresponding lists of harlocs computed
"""
md_threshold = max-distance threshold used to build quads out of Harris features
st_threshold = threshold value for spatio-temporal consistency (coherence)
all_ori, all_id, all_max, all_cen = orientation, reference frame ids, max distances,
                                      respectively centroids coordinates of each
                                      reference quad for scale scale_index
"""
def multiscale_quad_retrieval(r_quadsTree, r_harlocs, q_harlocs, md_threshold, st_threshold, \
            all_ori, all_id, all_max, all_cen, nos, scale_index, cropflag, \
            sequence):
    common.DebugPrint("Entered multiscale_quad_retrieval(): " \
                        "md_threshold = %s, st_threshold = %s." % \
                        (str(md_threshold), \
                        str(st_threshold)));

    assert len(r_harlocs) != 0;
    assert len(q_harlocs) != 0;

    try:
        Votes_space = np.load("Votes_space%d.npz" % scale_index)['arr_0'];
        HH = np.load("HH%d.npz" % scale_index)['arr_0'];
        return Votes_space, HH;
    except:
        common.DebugPrintErrorTrace();

    if common.MY_DEBUG_STDOUT and DBGPRINT:
        common.DebugPrint("multiscale_quad_retrieval(): r_quadsTree = %s" % \
                            str(r_quadsTree));

        common.DebugPrint("multiscale_quad_retrieval(): len(r_harlocs) = %d" % len(r_harlocs));
        common.DebugPrint("multiscale_quad_retrieval(): r_harlocs = %s" % str(r_harlocs));

        common.DebugPrint("multiscale_quad_retrieval(): q_harlocs = %s" % str(q_harlocs));
        common.DebugPrint("multiscale_quad_retrieval(): md_threshold = %s" % str(md_threshold));
        print("multiscale_quad_retrieval(): st_threshold = %s" % str(st_threshold));
        #common.DebugPrint("multiscale_quad_retrieval(): all_ori, all_id, all_max, all_cen, nos, scale_index, cropflag = %s" % str(all_ori, all_id, all_max, all_cen, nos, scale_index, cropflag));
        common.DebugPrint("multiscale_quad_retrieval(): all_id = %s" % str(all_id));
        common.DebugPrint("multiscale_quad_retrieval(): all_id.shape = %s" % (str(all_id.shape)));
        #common.DebugPrint("multiscale_quad_retrieval(): all_max, all_cen, nos, scale_index, cropflag = %s" % str(all_max, all_cen, nos, scale_index, cropflag));
        #common.DebugPrint("multiscale_quad_retrieval(): all_max = %s" % str(all_max));
        #common.DebugPrint("multiscale_quad_retrieval(): all_cen, nos, scale_index, cropflag = %s" % str(all_cen, nos, scale_index, cropflag));
        common.DebugPrint("multiscale_quad_retrieval(): sequence = %s" % str(sequence));
        print("multiscale_quad_retrieval(): cropflag = %s" % str(cropflag));

    t1 = float(cv2.getTickCount());

    if scale_index > nos:
        assert scale_index <= nos;
        #error('Wrong scale index or number-of-scales');

    #QD = dir([q_path "multiharlocs*.mat"])
    #QD = [q_path + "multiharlocs*.mat"]
    #QD = q_harlocs;

    #RD = dir([r_path "multiharlocs*.mat"])
    #RD = [r_path + "multiharlocs*.mat"]
    #RD = r_harlocs;

    #TODO: take out RD_start
    #RD_start = str2num(RD(1).name(end - 9 : end - 4))
    #RD_start = int(RD[0][-9 : -4])
    RD_start = 0;

    #RD_end = str2num(RD(end).name(end - 9 : end - 4))
    #RD_end = int(RD[-1][-9 : -4])
    #RD_end = len(RD) - 1;
    RD_end = len(r_harlocs) - 1;

    if False: # n_d not used anywhere
        #n_d = hist(all_id, RD_start : RD_end)
        #n_d = hist[all_id, RD_start : RD_end]
        n_d = Matlab.hist(x=all_id, \
                      binCenters=np.array(range(RD_start, RD_end + 1)) );

        #cross_indices = np.zeros( (len(QD), 2) );
        cross_indices = np.zeros( (len(q_harlocs), 2) );

    j = 1;

    #tic
    #ORI = np.array([]); # ORI NOT used anywhere

    """
    Inspired from
      https://stackoverflow.com/questions/17559140/matlab-twice-as-fast-as-numpy
        BUT doesn't help in this case:
    Votes_space = np.asfortranarray(np.zeros( (len(RD), len(QD)) ));
    """
    #Votes_space = np.zeros( (len(RD), len(QD)) );
    Votes_space = np.zeros( (len(r_harlocs), len(q_harlocs)) );

    # Make a distinct copy of HH from Votes_space...
    #HH = Votes_space.copy().astype(np.int16); #Votes_space + 0;
    #HH = np.zeros((len(RD), len(QD)), dtype=np.int8);
    HH = np.zeros((len(r_harlocs), len(q_harlocs)), dtype=np.int8); #!!!!TODO use MAYBE even np.bool - OR take it out

    #common.DebugPrint("multiscale_quad_retrieval(): Votes_space = %s,\n       HH = %s" % (str(Votes_space), str(HH)))

    tolers = 0.1 - float(scale_index) / 100.0; # it helps to make more strict the threshold as the scale goes up
    # tolers = 0.15 - float(scale_index) / 100.0;

    MAXDIS = 3 + scale_index;
    MAXORI = 0.25;


    """
    !!!!TODO TODO: I am using multiprocessing.Poll and return votes;
      the dispatcher assembles the results,
        but the results are NOT the same with the serial case - although they
           look pretty decent, but they seem to be suboptimal - dp_Alex returns
             suboptimal cost path for USE_MULTITHREADING == True instead of
             False.
             (Note: running under the same preconditions
                 multiscale_quad_retrieval I got the same results in dp_Alex().
    """
    if False: #config.USE_MULTITHREADING == True:
        global g;
        g.r_quadsTree = r_quadsTree;
        g.r_harlocs = r_harlocs;
        g.q_harlocs = q_harlocs;
        g.md_threshold = md_threshold;
        g.st_threshold = st_threshold;
        g.all_ori = all_ori;
        g.all_id = all_id;
        g.all_max = all_max;
        g.all_cen = all_cen;
        g.nos = nos;
        g.scale_index = scale_index;
        g.cropflag = cropflag;
        g.sequence = sequence;
        g.RD_start = RD_start;
        g.RD_end = RD_end;
        g.MAXDIS = MAXDIS;
        g.MAXORI = MAXORI;
        g.tolers = tolers;

        """
        Start worker processes to use on multi-core processor (able to run
           in parallel - no GIL issue if each core has it's own VM)
        """
        pool = multiprocessing.Pool(processes=config.numProcesses);
        print("multiscale_quad_retrieval(): Spawned a pool of %d workers" % \
                                config.numProcesses);

        listParams = range(0, len(q_harlocs)); #!!!!TODO: use counterStep, config.initFrame[indexVideo]

        #res = pool.map(IterationStandaloneMQR, listParams);
        # See https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.pool
        res = pool.map(func=IterationStandaloneMQR, iterable=listParams, \
                        chunksize=1);

        print("Pool.map returns %s" % str(res)); #x0.size + 1

        """
        From https://medium.com/building-things-on-the-internet/40e9b2b36148
            close the pool and wait for the work to finish
        """
        pool.close();
        pool.join();

        # Doing the "reduce" phase after the workers have finished :)
        assert len(res) == len(q_harlocs);
        for queryFrame, resE in enumerate(res):
            resEIndex = resE[0];
            resE = resE[1];
            assert resEIndex == queryFrame;
            # Gives: "ValueError: output operand requires a reduction, but reduction is not enabled"
            #Votes_space[:, queryFrame - 1] = votes;
            Votes_space[:, queryFrame] = resE;

        for queryFrame in range(len(q_harlocs)):
            if cropflag == 0:
                HH[:, queryFrame] = 1;
            else:
                """
                HH[:, queryFrame] = spatial_consistency.spatial_consistency(space_xy, \
                                            qcen, len(RD), st_threshold, cropflag);
                """
                HH[:, queryFrame] = spatial_consistency.spatial_consistency(space_xy, \
                                            qcen, len(r_harlocs), st_threshold, cropflag);

        try:
            np.savez_compressed("Votes_space%d" % scale_index, Votes_space);
            np.savez_compressed("HH%d" % scale_index, HH);
        except:
            common.DebugPrintErrorTrace();

        return Votes_space, HH;



    """
    We substitute q - 1 with q, since we want
      to number arrays from 0 (not from 1 like in Matlab).
    """
    #for q=1:length(QD)
    #for q in range(1, len(QD) + 1):
    #for queryFrame in range(len(QD)):
    for queryFrame in range(len(q_harlocs)):
        common.DebugPrint("multiscale_quad_retrieval(): Starting iteration queryFrame = %d" % queryFrame);
        # tic

        """
        str1=['load ' q_path QD(q).name]
        eval(str1)
        """

        """
        We make pp reference the desired multiharloc list for the query video
           frame queryFrame
        """
        pp = q_harlocs[queryFrame];
        #pp = np.array(pp);

        #common.DebugPrint("multiscale_quad_retrieval(): pp = %s" % str(pp));

        #[qout,qcen,qmaxdis,qori]=findquads(pp(pp(:,3)==scale_index,1:2),md_threshold,0);
        points = pp[pp[:, 2] == scale_index, 0:2];
        qout, qcen, qmaxdis, qori = findquads.findquads(points, md_threshold, 0);

        if common.MY_DEBUG_STDOUT and DBGPRINT:
            print("multiscale_quad_retrieval(): queryFrame = %d, " \
                          "qout.shape (number of quads for query frame queryFrame) = %s" % \
                                                 (queryFrame, str(qout.shape)));

        # disp([num2str(q) ' of ' num2str(length(QD)) ' -> ' num2str(size(qout,1)) ' quads'])

        #space_xy=zeros(size(qcen,1),2*length(RD))+nan;
        #space_xy = np.zeros( (qcen.shape[0], 2 * len(RD)) ) + np.nan;
        space_xy = np.zeros( (qcen.shape[0], 2 * len(r_harlocs)) ) + np.nan;

        #     votes=zeros(length(RD),1)
        #votes=zeros(length(RD),length(tolers));
        #votes = np.zeros( (len(RD), 1) );
        votes = np.zeros( (len(r_harlocs), 1) );

        #nep = np.array([]);
        #m_points = np.array([]);

        assert isinstance(tolers, float);

        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("multiscale_quad_retrieval(): quads of query frame %d are: " % queryFrame);
            common.DebugPrint("  qout = %s" % str(qout));

        """
        Alex: for each quad (4 floats) of the query frame from Harris feature of scale scale_index
          Note: all_id stores the reference frame id for each quad descriptor.
        """
        """
        We substitute queryFrameQuad - 1 with queryFrameQuad, since we want
            to number arrays from 0 (not from 1 like in Matlab).
        """
        #for queryFrameQuad in range(1, qout.shape[0] + 1):
        for queryFrameQuad in range(qout.shape[0]):
            common.DebugPrint("multiscale_quad_retrieval(): Starting iteration queryFrameQuad = %d" % queryFrameQuad);
            """
            Matlab's polymorphism is really bugging here: although it's
                normally a float, tolers is considered to be a size 1 vector...
                so len(tolers) == 1
            """
            #for tol_i in range(1, len(tolers) + 1):
            #    tol = tolers[tol_i - 1]
            """
            We substitute tol_i - 1 with tol, since we want
                to number arrays from 0 (not from 1 like in Matlab).
            """
            #for tol_i in range(1, 1 + 1):
            for tol_i in range(1):
                tol = tolers;

                """
                # TODO: done below - take out this dbg print
                if DBGPRINT:
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "qout[queryFrameQuad, :] = %s" % \
                                        str(qout[queryFrameQuad, :]))
                """

                #% default for first PAMI with tol= 0.1 approximately

                # NOTE: SciPy's KDTree finds a few more results, in some cases,
                #    than the Matlab code from Evangelidis.

                #idx, di = kdtree_ball_query(tree, qout(i, :), tol)
                #idx, distKD = kdtree_ball_query(tree, qout[i - 1, :], tol)
                #idx, di = tree.query(x=xQuery, k=4)
                #resPoints = [data[i] for i in resBallIndices]
                # tol is a scalar representing the radius of the ball
                if config.KDTREE_IMPLEMENTATION == 0:
                    idx = r_quadsTree.query_ball_point(qout[queryFrameQuad, :], tol);
                elif config.KDTREE_IMPLEMENTATION == 1:
                    #pt = qout[queryFrameQuad - 1, :].astype(np.float32);
                    pt = qout[queryFrameQuad, :];
                    pt = np.array([[pt[0], pt[1], pt[2], pt[3]]], dtype=np.float32);
                    retval, idx, dists = r_quadsTree.radiusSearch( \
                                                query=pt, \
                                                radius=(tol**2), \
                                                maxResults=NUM_MAX_ELEMS, \
                                                params=search_params);
                    if common.MY_DEBUG_STDOUT and DBGPRINT:
                        """
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                        "retval (number NNs) = %s" % str(retval));
                        """
                        common.DebugPrint( \
                            "multiscale_quad_retrieval(): radiusSearch's retval " \
                            "(at queryFrame=%d, queryFrameQuad=%d) is %d" % (queryFrame, queryFrameQuad, retval));

                    idx = idx[0];
                    dists = dists[0];
                    """
                    Note: retval is the number of neighbors returned from the radiusSearch().
                      But the idx and the dists can have more elements than the returned retval.
                    """
                    idx = idx[: retval];
                    dists = dists[: retval];

                if common.MY_DEBUG_STDOUT and DBGPRINT:
                    print("multiscale_quad_retrieval(): " \
                            "qout[queryFrameQuad, :] = %s" % str(qout[queryFrameQuad, :]));
                    print("multiscale_quad_retrieval(): " \
                                      "idx = %s" % str(idx));
                    print("multiscale_quad_retrieval(): " \
                                      "dists = %s" % str(dists));
                    print("multiscale_quad_retrieval(): " \
                                      "tol = %s" % str(tol));
                    if config.KDTREE_IMPLEMENTATION == 0:
                        print("multiscale_quad_retrieval(): " \
                                "r_quadsTree.data[idx] = %s" % \
                                str(r_quadsTree.data[idx]));

                # We print the distances to the points returned in idx
                if common.MY_DEBUG_STDOUT and DBGPRINT: # This is just for debugging purposes
                    a = qout[queryFrameQuad, :];
                    if config.KDTREE_IMPLEMENTATION == 0:
                        for myI, index in enumerate(idx):
                            b = r_quadsTree.data[index];
                            """
                            if False:
                                common.DebugPrint("multiscale_quad_retrieval(): distance to " \
                                    "%d point (%s) inside ball = %.4f" % \
                                    (myI, str(b), npla.norm(a - b)));
                            """
                    else:
                        pass;
                idx = np.array(idx);

                #if False:
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                "all_max.shape = %s" % str(all_max.shape));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                "qmaxdis.shape = %s" % str(qmaxdis.shape));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                      "qmaxdis = %s" % str(qmaxdis));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                      "qori.shape = %s" % str(qori.shape));
                    common.DebugPrint("multiscale_quad_retrieval(): " \
                                      "qori = %s" % str(qori));

                #dis_idx=abs(qmaxdis(i)-all_max(idx))<MAXDIS;
                if len(idx) == 0:
                    # NOT A GOOD IDEA: continue;
                    #idx = np.array([]);
                    dis_idx = np.array([]);
                    ori_idx = np.array([]);
                else:
                    if common.MY_DEBUG_STDOUT and DBGPRINT:
                        print("multiscale_quad_retrieval(): " \
                                            "queryFrameQuad = %s" % str(queryFrameQuad));
                        print("multiscale_quad_retrieval(): " \
                            "all_max[idx] = %s" % str(all_max[idx]));
                        print("multiscale_quad_retrieval(): " \
                            "qmaxdis[queryFrameQuad] = %s" % str(qmaxdis[queryFrameQuad]));

                    if USE_GPS_COORDINATES:
                        # We look only at a part of the reference video
                        """
                        Since in some cases the video temporal alignment is
                            difficult to do due to similar portions in the
                            trajectory (see the drone videos, clip 3_some_lake)
                            we "guide" the temporal alignment by restricting
                            the reference frame search space - this is useful
                            when we have the geolocation (GPS) coordinate for
                            each frame.
                        """
                        if common.MY_DEBUG_STDOUT and DBGPRINT:
                            print("multiscale_quad_retrieval(): " \
                                "all_id = %s" % str(all_id));

                        if True:
                            #assert (all_id.ndim == 2) and (all_id.shape[1] == 1);
                            if all_id.ndim == 2:
                                #!!!!TODO TODO: put this at the beginning of the function
                                assert all_id.shape[1] == 1;
                                """
                                We flatten the array all_id
                                  Note: We don't use order="F" since it's
                                        basically 1-D array
                                """
                                all_id = np.ravel(all_id);

                        #!!!!TODO: put start and end frame in config - or compute it from geolocation
                        sub_idx = np.logical_and( (all_id[idx] >= 2030 - 928), \
                                                    (all_id[idx] <= 2400 - 928) );
                        idx = idx[sub_idx];

                        if common.MY_DEBUG_STDOUT and DBGPRINT:
                            print("multiscale_quad_retrieval(): " \
                                "all_id = %s" % str(all_id));
                            print("multiscale_quad_retrieval(): " \
                                "sub_idx = %s" % str(sub_idx));
                            print("multiscale_quad_retrieval(): " \
                                "idx = %s" % str(idx));

                    if FILTER:
                        dis_idx = np.abs(qmaxdis[queryFrameQuad] - all_max[idx]) < MAXDIS;

                        #if False:
                        if common.MY_DEBUG_STDOUT:
                            """
                            common.DebugPrint("multiscale_quad_retrieval(): " \
                                                "idx = %s" % str(idx));
                            """
                            common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "dis_idx = %s" % str(dis_idx));

                        #idx=idx(dis_idx)
                        idx = idx[dis_idx];

                    #if False:
                    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                            "idx (after idx = idx[dis_idx]) = %s" % str(idx));

                    if FILTER:
                        #ori_idx=abs(qori(i)-all_ori(idx))<MAXORI;
                        ori_idx = np.abs(qori[queryFrameQuad] - all_ori[idx]) < MAXORI;

                        #if False:
		        if common.MY_DEBUG_STDOUT:
                            """
                            common.DebugPrint("multiscale_quad_retrieval(): " \
                                                    "all_ori = %s" % str(all_ori));
                            common.DebugPrint("multiscale_quad_retrieval(): " \
                                    "qori[queryFrameQuad] = %s" % str(qori[queryFrameQuad]));

                            """
                            common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "ori_idx = %s" % str(ori_idx));

                        #idx=idx(ori_idx);
                        idx = idx[ori_idx];


                # IMPORTANT ###################################################
                # IMPORTANT ###################################################
                # IMPORTANT ###################################################
                #% spatio-temporal consistency
                # IMPORTANT ###################################################
                # IMPORTANT ###################################################
                # IMPORTANT ###################################################

                #if numel(idx) > 0:
                if idx.size > 0:
                    if cropflag == 0:
                        if FILTER:
                            """
                            Alex: this is a simple procedure of eliminating False
                            Positive (FP) matches, as presented in Section 4.2 of
                            TPAMI 2013 paper.
                            Basically it filters out quad matches that have
                            centroids st_threshold away from the query quad.
                            Note: all_cen are the controids of all reference
                                quads.
                            """
                            dy = qcen[queryFrameQuad, 0] - all_cen[idx, 0];
                            dx = qcen[queryFrameQuad, 1] - all_cen[idx, 1];

                            #D=dy.^2+dx.^2;
                            D = dy**2 + dx**2;

                            co_idx = D < pow(st_threshold, 2);

                            idx = idx[co_idx];
                    else:
                        """
                        We substitute iii - 1 with iii, since we want
                            to number arrays from 0 (not from 1 like in Matlab).
                        """
                        #for iii in range(1, len(idx) + 1):
                        for iii in range(len(idx)):
                            #space_xy(i,(all_id(idx(iii))-RD_start)*2+1:(all_id(idx(iii))-RD_start)*2+2) = all_cen(idx(iii),:)
                            space_xy[queryFrameQuad, \
                                    (all_id[idx[iii]] - RD_start) * 2: (all_id[idx[iii] - 1] - RD_start) * 2 + 1] = \
                                    all_cen[idx[iii], :];

                    #hh=hist(all_id(idx),RD_start:RD_end);
                    # It has to be an np.array because we multiply it with a scalar
                    histoRange = np.array(range(RD_start, RD_end + 1));
                    hh = Matlab.hist(x=all_id[idx], binCenters=histoRange);

                    #if False:
                    #if True:
		    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "hh = %s" % (str(hh)));
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "hh.shape = %s" % (str(hh.shape)));

                        """
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                            "all_id = %s" % (str(all_id)));
                        """
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                    "all_id.shape = %s" % (str(all_id.shape)));
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                    "idx = %s" % (str(idx)));
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                    "idx.shape = %s" % (str(idx.shape)));

                    # % nz can be computed more optimally
                    #nz=find(hh~=0); # nz can be computed more optimally
                    # np.nonzero() always returns a tuple, even if it contains 1 element since hh has only 1 dimension
                    nz = np.nonzero(hh != 0)[0];
                    #if False:
	    	    if common.MY_DEBUG_STDOUT:
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                          "nz = %s" % (str(nz)));
                        common.DebugPrint("multiscale_quad_retrieval(): " \
                                          "nz.shape = %s" % (str(nz.shape)));

                    #if numel(nz) > 0
                    if nz.size > 0:
                        #%%----text-retrieval-like
                        #votes(nz, tol_i) = votes(nz, tol_i) + log10(length(RD) / (length(nz)))^2 #PREVIOUSLY
                        #myVal = pow(math.log10(float(len(RD)) / len(nz)), 2);
                        myVal = pow(math.log10(float(len(r_harlocs)) / len(nz)), 2);
                        """
                        try:
                            myVal = pow(math.log10(float(len(r_harlocs)) / len(nz)), 2);
                        except:
                            print("Error: len=%d len(nz)=%d nz.size=%d" % \
                                            (len(r_harlocs), len(nz), nz.size));
                            common.DebugPrintErrorTrace();
                        """

                        #if False:
		        if common.MY_DEBUG_STDOUT:
                            """
                            common.DebugPrint("multiscale_quad_retrieval(): " \
                                              "len(RD) = %d" % len(RD));
                            """
                            common.DebugPrint("multiscale_quad_retrieval(): " \
                                              "len(r_harlocs) = %d" % len(r_harlocs));
                            common.DebugPrint("multiscale_quad_retrieval(): " \
                                              "len(nz) = %d" % len(nz));
                            common.DebugPrint("multiscale_quad_retrieval(): " \
                                               "myVal = %.5f" % myVal);

                        # PREVIOUSLY
                        votes[nz, tol_i] = votes[nz, tol_i] + myVal;
                        #   votes(nz)=votes(nz)+log10(length(RD)/(length(nz)));
                        #   votes(nz)=votes(nz)+1;

        if common.MY_DEBUG_STDOUT and DBGPRINT:
            """
            common.DebugPrint("multiscale_quad_retrieval(): " \
                    "Votes_space.shape = %s" % (str(Votes_space.shape)));
            common.DebugPrint("multiscale_quad_retrieval(): " \
                    "votes.shape = %s" % (str(votes.shape)));
            """

            print("multiscale_quad_retrieval(): " \
                              "votes.shape = %s" % (str(votes.shape)));
            if (np.abs(votes) < 1.0e-10).all():
                print( \
                      "multiscale_quad_retrieval(): votes = 0 (all zeros)");
            else:
                print("multiscale_quad_retrieval(): " \
                              "votes = %s" % (str(votes)));

        #Votes_space(:,q)=votes;
        # Gives: "ValueError: output operand requires a reduction, but reduction is not enabled"
        #Votes_space[:, queryFrame - 1] = votes;
        # Note: since votes is basically a 1-D vector, we don't use the Fortran order
        Votes_space[:, queryFrame] = np.ravel(votes); # order="F");


        if cropflag == 0:
            HH[:, queryFrame] = 1;
        else:
            """
            HH[:, queryFrame] = spatial_consistency.spatial_consistency(space_xy, \
                                        qcen, len(RD), st_threshold, cropflag);
            """
            HH[:, queryFrame] = spatial_consistency.spatial_consistency(space_xy, \
                                        qcen, len(r_harlocs), st_threshold, cropflag);


    if common.MY_DEBUG_STDOUT and DBGPRINT:
        print("multiscale_quad_retrieval(scale_index=%d): " \
                            "Votes_space =\n%s" % (scale_index, str(Votes_space)));

    try:
        np.savez_compressed("Votes_space%d" % scale_index, Votes_space);
        np.savez_compressed("HH%d" % scale_index, HH);
    except:
        common.DebugPrintErrorTrace();

    t2 = float(cv2.getTickCount());
    myTime = (t2 - t1) / cv2.getTickFrequency();
    print("multiscale_quad_retrieval() took %.6f [sec]" % myTime);
    """
    common.DebugPrint("multiscale_quad_retrieval(): " \
                        "%d corresponding frames retrieved in %.6f secs" % \
                        (len(q_harlocs), myTime));
    """

    return Votes_space, HH;

