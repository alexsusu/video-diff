"""
findquads finds the relevant quads among all given Points
    (multi-scale Harris features), given parameter threshold.
To efficiently solve the problem we perform NNS (Nearest-Neighbor Search)
    using kd-tree.

More exactly, we generate quads (quads points) for each point
    from Points (we use NN for Points) and filter out the ones that have
        max-distance >= threshold.
    Then we generate hashes of 4 floats from the coordinates of these quads
                points.

Note: because we use a lot of DebugPrint() calls, the evaluated arguments
    slow down considerably the performance of the findquads() function,
    independently of the value of common.MY_DEBUG_STDOUT.
    Reason: we print a lot of rather big arrays/matrices.
  Therefore, we guard the evaluation of these DebugPrint() calls with
    conditional: if common.MY_DEBUG_STDOUT .
"""

import common
import config
import Matlab

import cv2
import math
import numpy as np
import numpy.matlib
from numpy import linalg as npla

if config.KDTREE_IMPLEMENTATION == 0:
    from scipy import spatial


"""
FOOBAR is where I tried to use WITHOUT SUCCESS np.c_ and np.r_ instead of
    np.hstack and np.vstack, respectively.
"""
FOOBAR = False


"""
if Points is np.array
    Points.shape returns e.g., (2, 2)
"""

# Used by multiscale_quad_tree.m, used by synchro....m

# Alex: determine quads for the given Harris points, for a given scale, for a given video frame
#function [Quads, Centroids, maxDistances, Orientation]=findquads(Points,threshold,reflect_flag)
def findquads(Points, threshold, reflect_flag):
    t1 = float(cv2.getTickCount())

    """
    Alex: Points contains M 2D points.
          More exactly, Points is a list of Harris features for the current (reference of query) video frame, for a given scale.
        We find the quadruples that are valid (when
            the distance between the most-widely separated points within a
            quadruple is below the scalar threshold).

    % Alex: the quads are basically VALID tuples of 4 points from Points (Points are the extracted keypoints)
    %    "We apply the recentquad descriptor [7] for rough video synchronization.
    %       This expressive, low dimen-sional local feature descriptor allows for very efficient retrieval
    %           schemes based on kD-trees and, thus, establishes frame correspondences much faster than
    %           standard bag-of-words (BoW) approaches with high-dimensional descriptors [5], [6], [12]."
    %    [7] D. Lang, D.W. Hogg, K. Mierle, M. Blanton, and S. Roweis, "Astrometry.net: Blind Astrometric Calibration of Arbitrary Astronomical Images," The Astronomical J.,vol. 37, pp. 1782-2800, 2010.

    % [QUADS,CENTROIDS, MAXDISTANCES, ORIENTATION] = FINDQUADS (POINTS, THRESHOLD, REFLECT_FLAG)
    % This function finds the quad descriptors of the points in Mx2 POINTS matrix
    % that contains the coordinates (y,x) of M points.
    % Points = [y1 x1; y2 x2;....;yM xM];
    % The quadruples are valid
    % when the distance between the most-widely separated points within a quadruple
    % is below the scalar THRESHOLD. REFLECT_FLAG is good to be enabled (equal to 1)
    % when the descriptors are computed for the reference image or the database of
    % images that is used as reference (it can be disabled (0) for a query image
    % that has to be compared with the reference one)
    %
    % The N valid quad codes are returned to Nx4 matrix QUADS with their centroids stored
    % in the Nx2 matrix CENTROIDS. The distance and the orientation of each
    % valid quad are stored in the length-N vectors MAXDISTANCES and ORIENTATION.
    %--------------------------------
    % Comment about REFLECT_FLAG: Given that q is the quad code of a quadruple A,B,C,D
    % with A,B the control points, i.e. A corresponds to (0,0) and B corresponds to (1,1),
    % the code of the quadruple by considering the control points as B and A (reverse order)
    % is 1-q. If REFLECT_FLAG is true, then we compute both codes for each quadruple.

    Alex:
      From 2013 paper:
        Section 4:
          "Suppose a quadruple ("quad") of interest points yi; i = {1;2;3;4} as shown in Fig. 3a.
          The points y1;y2 are the control points defined as the most widely separated pair of points.
          Let d denote the distance (diameter) between the control points,
              phi the orientation of the diameter, and c the centroid of this quad"

          "We then consider a local coordinate systemOXYoriented and centered with respect
          to the control points y1;y2, so that they coincide with
          the points (0;0) and (1;1), respectively. This allows us to
          encode the quad structure in terms of the new coordinates of
          the remaining points y3;y4. Accordingly, any quad of four
          points can be represented by means of a 4D vectorq, which
          is called aquad descriptor, or simply aquad. In essence, such a
          coding realizes asimilarity normalization transform, i.e., the
          descriptor is invariant to any scale, rotation, and translation
          of points."


    Points is an M x 2 array.
    We then choose certain meaningful quads out of these points (we use thershold to filter, etc)
    //
    Quads is an Q x 4 array (from various combination of 4 points (8 floats - vars out and out2) we generate 4 floats.
    Centroids is an Q x 4 array.
    maxDistances is an Q x 1 array.
    Orientation is an Q x 1 array.
    """

    # if False:
    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("findquads(): Points = %s" % str(Points))
        common.DebugPrint("findquads(): Points.shape = %s" % str(Points.shape))

    #if size(Points, 2) != 2:
    if Points.shape[1] != 2:
        #common.DebugPrint("findquads(): Points should be concatenated in an Mx2 matrix")
        quit();

    locs = Points;

    if locs.shape[0] > 3:
        out = np.zeros( (locs.shape[0], 8) );
        distance = np.zeros( (locs.shape[0], 4) );
        centro = np.zeros( (locs.shape[0], 2) );

        if config.KDTREE_IMPLEMENTATION == 0:
            mytree = spatial.KDTree(locs);
            #common.DebugPrint("findquads(): mytree.data = %s" % str(mytree.data));
        elif config.KDTREE_IMPLEMENTATION == 1:
            locs = locs.astype(np.float32);
            #!!!!TODO: try to use exact NN-search for the kd-tree - see http://docs.opencv.org/trunk/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html
            mytree = cv2.flann_Index(features=locs, params=config.FLANN_PARAMS);

        # Alex: locs is a list of Harris features for a video frame
        #for i=1:numel(locs(:,1))
        for i in range(locs.shape[0]):
            #idx, dists = kdtree_k_nearest_neighbors(mytree, locs[i - 1, :], 4);
            xQuery = locs[i, :];
            #common.DebugPrint("findquads(): xQuery = %s" % str(xQuery));
            if config.KDTREE_IMPLEMENTATION == 0:
                dists, idx = mytree.query(x=xQuery, k=4);
            elif config.KDTREE_IMPLEMENTATION == 1:
                #TODO!!!!Put in global scope
                search_params = dict(checks=1000000000);
                idx, dists = mytree.knnSearch(query=xQuery, knn=4, \
                                                params=search_params);
                idx = idx[0];
                dists = dists[0];
                dists = np.sqrt(dists); #TODO: we can avoid np.sqrt - maybe make threshold**2

            """
            It seems kdtree_k_nearest_neighbors Matlab MEX is returning
                sorted decreasing dists (and idx correspondingly), while
                Python's query() returns increasing dists, so we reverse
                the result lists.
            if False:
                print "findquads(): dists (before) = %s" % dists
                print "findquads(): idx (before) = %s" % idx

            distsL = dists.tolist();
            distsL.reverse();
            dists = np.array(distsL);

            idxL = idx.tolist();
            idxL.reverse();
            idx = np.array(idxL);
            """

            #if False:
            #if True:
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("findquads(): dists = %s" % dists);
                common.DebugPrint("findquads(): idx = %s" % idx);
                #pass;

            # if (s_D(1)<threshold)&&(sum(diff(s_D(1:4))==0)==0)%&(sqrt(s_D(2))>2)
            s_D = dists;
            if (s_D[3] < threshold):
                # The following predicate turns out to always be True so we just put an assert
                #and \((np.diff(a=s_D[0: 3], n=1, axis=0) == 0).sum() == 0): #%&(sqrt(s_D(2))>2):
                #assert (np.diff(a=s_D[0: 3], n=1, axis=0) == 0).sum() < 1.0e-3; #== 0;
                valDiff = (np.diff(a=s_D[0: 3], n=1, axis=0) == 0).sum(); #== 0;

                common.DebugPrint("findquads(): valDiff = %s" % str(valDiff));

                if valDiff >= 1.0e-3:
                    common.DebugPrint("findquads(): valDiff = %.7f" % valDiff);
                else:
                    """
                    % Alex: temp is a 2 * 4 matrix with value [x1 x2 x3 x4; y1 y2 y3 y4] (coordinates of the quad)
                    #temp=[locs(idx,1)';locs(idx,2)'];
                    """
                    if FOOBAR:
                        #NOT GOOD since we need to do a transposed, etc (see below :) ):
                        temp = np.r_[locs[idx, 0].T, locs[idx, 1].T];
                    else:
                        #common.DebugPrint("locs[idx, 0].T.shape = %s" % str(locs[idx, 0].T.shape));
                        #common.DebugPrint("locs[idx, 1].T.shape = %s" % str(locs[idx, 1].T.shape));
                        #common.DebugPrint("blabla = %s" % str(np.r_[locs[idx, 0].T, locs[idx, 1].T]));
                        temp = np.vstack( ( locs[idx, 0].T, locs[idx, 1].T ) );
                        #common.DebugPrint("temp = %s" % str(temp));
                        #common.DebugPrint("np.ravel(temp[:].T) = %s" % str(np.ravel(temp[:].T)));

                    #common.DebugPrint("findquads(): temp = %s" % temp)

                    #% Alex: out is a matrix with 8 colums - all the (x, y) coordinates of the poins in the quad
                    #out(i,:)=temp(:)';
                    out[i, :] = np.ravel(temp[:].T); # We really have to use np.ravel(), otherwise we get: "ValueError: operands could not be broadcast together with shapes (8) (4,2)"
                    #common.DebugPrint("findquads(): out = %s" % out)

                    #% Alex: we compute the centroid coordinates for the quad, using the (x, y) coordinates of the poins in the quad
                    #centro(i,:)=[mean(out(i,1:2:end)),mean(out(i,2:2:end))];
                    centro[i, :] = [out[i, 0: : 2].mean(), out[i, 1: : 2].mean()];
                    #common.DebugPrint("findquads(): centro = %s" % centro)

                    # %distances between i and 3 neighbors in descending order
                    #% Alex: obviously nearest point to i is i itself
                    distance[i, :] = s_D.T;
                    #common.DebugPrint("findquads(): distance = %s" % distance)

            # temp is a 2 * 4 matrix with value [x1 x2 x3 x4; y1 y2 y3 y4] (coordinates of the quad)
            #temp = np.vstack( ( locs[idx - 1, 0].conj().T, locs[idx - 1, 1].conj().T ) )

        #kdtree_delete(mytree);
        if config.KDTREE_IMPLEMENTATION == 1:
            mytree.release();

        #% toc

        """
        out contains for each point pt from Points
            the coordinates of the 4 closest points to pt from Points.

        out has rows of form (y1,x1,....,y4,x4).
        We reverse the y and x columns in order to obtain rows of the form
            (x1,y1,....,x4,y4)
        """
        """
        Alex: we don't execute this code since I consider inefficient to compute
            permutation by multiplying matrices.
        #% reverse the order of points and the distances
        #perper=fliplr(per);
        #out=out*perper; % points are now ordered as (x1,y1,....,x4,y4)
        """
        #out = out[:, ::-1]
        for i, row in enumerate(out):
            out[i, :] = [row[1], row[0], row[3], row[2], row[5], row[4], row[7], row[6]];
        #common.DebugPrint("findquads(): out = %s" % str(out));
        common.DebugPrint("findquads(): out.shape = %s" % str(out.shape));


        #% Alex: Compute the sum of the elements in each row.
        w = out.sum(1);
        w = np.ravel(w); #w.flatten() #IMPORTANT: flatten() returns a 2D array in the end... :(
        common.DebugPrint("findquads(): w = %s" % str(w))

        #% Alex: We take out the quads whose sum of coordinates are 0
        index = np.nonzero(w != 0);
        index = index[0]; # we take only the 1st result from the tuple result

        out2 = out[index, :];
        dis2 = distance[index, :];
        c2 = centro[index, :];

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): out2.shape = %s" % str(out2.shape))
            common.DebugPrint("findquads(): out2 = %s" % str(out2))
            common.DebugPrint("findquads(): dis2.shape = %s" % str(dis2.shape))
            common.DebugPrint("findquads(): dis2 = %s" % str(dis2))
            common.DebugPrint("findquads(): c2.shape = %s" % str(c2.shape))
            common.DebugPrint("findquads(): c2 = %s" % str(c2))

        """
        We take the furthest distance w.r.t. to point 1 from quad
                                        (1st quad control point).
            Indeed, when we search for a point pt in Points the closest
                4 points the 1st found is pt itself.
        """
        max_dis2 = dis2[:, 3];
        #common.DebugPrint("findquads(): max_dis2 = %s" % str(max_dis2))

        # find the longest distance out of six possible distances in a quad
        ori = 0 * max_dis2;
        #ind10 = ones(size(out2,1), 1)
        ind10 = np.ones( (out2.shape[0], 1) );


        common.DebugPrint("findquads(): out2.shape = %s" % str(out2.shape))
        ##common.DebugPrint("findquads(): out2 (before for) = %s" % str(out2))
        for i in range(out2.shape[0]):
            # Alex: We compute the distances of the pair of points not containing the first point of the quad
            d = np.array( (npla.norm(out2[i, 2: 4] - out2[i, 4: 6]),
                    npla.norm(out2[i, 2: 4] - out2[i, 6: 8]),
                    npla.norm(out2[i, 4: 6] - out2[i, 6: 8]) ) );

            #if False:
            if common.MY_DEBUG_STDOUT:
                #common.DebugPrint("findquads(): out2[i] = %s" % str(out2[i]));
                #common.DebugPrint("findquads(): out2[i, 2:4] = %s" % str(out2[i, 2:4]));
                #common.DebugPrint("findquads(): out2[i, 4:6] = %s" % str(out2[i, 4:6]));
                """
                common.DebugPrint("findquads(): npla.norm(out2[i, 2:4] - " \
                    "out2[i, 4:6] = %s" % str(npla.norm(out2[i, 2:4] - out2[i, 4:6])));
                """
                pass;
            #if False:
            if common.MY_DEBUG_STDOUT:
                common.DebugPrint("findquads(): d**2 = %s" % str(d**2));
            #common.DebugPrint("findquads(): d = %s" % str(d));

            #[sd,d_i]=max(d);
            d_i = d.argmax();
            sd = d[d_i];

            """
            Now sd is the biggest distance between the 3 points of this quad
                (except point[i] itself), while dis2 is the distance from
                 the current point (out2[0: 2]) to all the other 3 points.
            """
            if sd < threshold:
                max_dis2[i] = max([max_dis2[i], sd]);

                if sd < dis2[i, 3]:
                    #[0, 1, 6, 7, 2, 3, 4, 5]
                    #out2[i, :] = out2[i, :] * per1
                    l = out2[i, :];
                    out2[i, :] = [l[0], l[1], l[6], l[7], l[2], l[3], l[4], l[5]];
                    #disp('4')
                else:
                    if d_i == 0:
                        #out2[i, :] = out2[i, :] * per2
                        l = out2[i, :];
                        out2[i, :] = [l[4], l[5], l[2], l[3], l[0], l[1], l[6], l[7]];
                    elif d_i == 1:
                        #out2[i, :] = out2[i, :] * per3
                        l = out2[i, :]
                        out2[i, :] = [l[6], l[7], l[2], l[3], l[4], l[5], l[0], l[1]];
                    else:
                        #out2[i, :] = out2[i, :] * per4
                        l = out2[i, :];
                        out2[i, :] = [l[6], l[7], l[4], l[5], l[2], l[3], l[0], l[1]];
            else:
                ind10[i] = 0;
                #disp('5')

        #%%%Remove quads with max-distance above threshold
        #ind10=logical(ind10);
        ind10 = ind10.astype(int);
        ind10 = np.ravel(ind10); #.flatten() #reshape(1, 0)

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): ind10 = %s" % str(ind10));
            #common.DebugPrint("findquads(): max_dis2**2 = %s" % str(max_dis2**2));
            common.DebugPrint("findquads(): max_dis2 = %s" % str(max_dis2));
            #common.DebugPrint("findquads(): out2 (before) = %s" % str(out2));

        try:
            #out2=out2(ind10,:);
            #out2 = out2[ind10, :] # We put the rows that correspond to the 1s/Trues in ind10
            out2 = out2[np.nonzero(ind10)];

            #common.DebugPrint("findquads(): out2 (after) = %s" % str(out2))

            #c2=c2[ind10,:];
            c2 = c2[np.nonzero(ind10)];

            #max_dis2=max_dis2[ind10,:];
            max_dis2 = max_dis2[np.nonzero(ind10)];

            #ori=ori[ind10,:];
            ori = ori[np.nonzero(ind10)];
        except:
            common.DebugPrint("findquads(): ind10 = %s" % str(ind10));
            common.DebugPrint("findquads(): np.nonzero(ind10) = %s" % str(np.nonzero(ind10)));
            common.DebugPrint("findquads(): out2 = %s" % str(out2));
            common.DebugPrint("findquads(): c2 = %s" % str(c2));
            common.DebugPrint("findquads(): max_dis2 = %s" % str(max_dis2));
            common.DebugPrint("findquads(): ori = %s" % str(ori));

            if out2.size == 0:
                return np.array([]), np.array([]), np.array([]), np.array([]);
            """
            Quads = out2;
            Centroids = c2;
            maxDistances = max_dis2;
            Orientation = ori;
            """

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): c2 = %s" % str(c2));
            common.DebugPrint("findquads(): out2.shape = %s" % str(out2.shape));
            common.DebugPrint("findquads(): out2 = %s" % str(out2));



        #%Normalize coordinates (find the quad coding for each quadruple)
        #ox = out2[:, 1:2:end]
        ox = out2[:, 0::2]; # ox is the x coordinates of the 4 quad points
        #oy = out2[:, 2:2:end]
        oy = out2[:, 1::2]; # oy is the y coordinates of the 4 quad points

        #common.DebugPrint("findquads(): ox (orig) = %s" % str(ox));
        #common.DebugPrint("findquads(): oy (orig) = %s" % str(oy));

        # From http://wiki.scipy.org/NumPy_for_Matlab_Users
        #ox=ox-repmat(ox[:,1], 1, 4)
        """
        From http://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
         numpy.tile(A, reps)[source]
            Construct an array by repeating A the number of times given by reps.
        """
        #ox=ox-repmat(ox(:,1),1,4);
        ox = ox - np.tile(ox[:, 0], (4, 1)).T;

        #oy=oy-repmat(oy[:,1],1,4);
        oy = oy - np.tile(oy[:, 0], (4, 1)).T;

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): ox.shape = %s" % str(ox.shape));
            common.DebugPrint("findquads(): ox = %s" % str(ox));
            common.DebugPrint("findquads(): oy.shape = %s" % str(oy.shape));
            common.DebugPrint("findquads(): oy = %s" % str(oy));

        #common.DebugPrint("findquads(): ox[:, 1] = %s" % str(ox[:, 1]));
        #common.DebugPrint("findquads(): oy[:, 1] = %s" % str(oy[:, 1]));

        """
        From Matlab help:
          P = atan2(Y,X) returns an array P the same size as X and Y containing
            the element-by-element, four-quadrant inverse tangent (arctangent)
            of Y and X, which must be real.

        From http://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html
         <<numpy.arctan2(x1, x2[, out]) = <ufunc 'arctan2'>
            Element-wise arc tangent of x1/x2 choosing the quadrant correctly.>>
        """
        #alpha=atan2(oy[:,2],ox[:,2]);
        alpha = np.arctan2(oy[:, 1], ox[:, 1]);

        theta = math.pi / 4.0 - alpha;
        ori = alpha;

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): alpha.shape = %s" % str(alpha.shape))
            common.DebugPrint("findquads(): alpha = %s" % str(alpha))
            common.DebugPrint("findquads(): theta.shape = %s" % str(theta.shape))
            common.DebugPrint("findquads(): theta = %s" % str(theta))
            common.DebugPrint("findquads(): ori.shape = %s" % str(ori.shape))
            common.DebugPrint("findquads(): ori = %s" % str(ori))

        # IMPORTANT NOTE: theta is a vector

        #matcos=repmat(cos(theta),1,4);
        matcos = np.tile(np.cos(theta), (4, 1)).T;

        #matsin=repmat(sin(theta),1,4);
        matsin = np.tile(np.sin(theta), (4, 1)).T;

        #common.DebugPrint("findquads(): ox.shape = %s" % str(ox.shape));
        #common.DebugPrint("findquads(): oy.shape = %s" % str(oy.shape));
        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): matcos.shape = %s" % str(matcos.shape));
            common.DebugPrint("findquads(): matcos = %s" % str(matcos));
            common.DebugPrint("findquads(): matsin.shape = %s" % str(matsin.shape));
            common.DebugPrint("findquads(): matsin = %s" % str(matsin));

        # This gives error: "ValueError: matrices are not aligned"
        #print "findquads(): ox * matcos.shape = %s" % str((ox * matcos).shape);

        """
        common.DebugPrint("findquads(): ox * matcos.shape = %s" % \
                            str(np.multiply(ox, matcos).shape));
        """
        #common.DebugPrint("findquads(): out2.shape = %s" % str(out2.shape))
        """
        common.DebugPrint("findquads(): out2[:, 1::2] = %s" % \
                             str(out2[:, 1::2]));
        """
        """
        common.DebugPrint("findquads(): out2[:, 1::2].shape = %s" % \
                            str(out2[:, 1::2].shape));
        """

        #common.DebugPrint("findquads(): out2 = %s" % str(out2));
        #common.DebugPrint("findquads(): out2.shape = %s" % str(out2.shape));

        #out2(:,2:2:end)=ox.*matcos-oy.*matsin
        out2[:, 1::2] = np.multiply(ox, matcos) - np.multiply(oy, matsin);
        # This gives error: "ValueError: matrices are not aligned", since numpy considers them matrices...
        #out2[:, 1::2] = ox * matcos - oy * matsin;

        #out2(:,1:2:end)=ox.*matsin+oy.*matcos
        out2[:, 0::2] = np.multiply(ox, matsin) + np.multiply(oy, matcos)

        #if False:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): out2.shape!!!! (before /) = %s" % str(out2.shape));
            common.DebugPrint("findquads():out2[:, 2].shape = %s" % \
                                        str(out2[:, 2].shape));
            common.DebugPrint("findquads():out2[:, 2].shape = %s" % \
                                        str(out2[:, 2].shape));
            common.DebugPrint("findquads(): np.tile(out2[:, 2], (8, 1)).T.shape = %s" % \
                                        str(np.tile(out2[:, 2], (8, 1)).T.shape));
            common.DebugPrint("findquads(): np.tile(out2[:, 2], (8, 1)).T = %s" % \
                                        str(np.tile(out2[:, 2], (8, 1)).T));
            #common.DebugPrint("findquads(): np.tile(out2[:, 2], (8, 1)).T = %s" % \
            #                            str(np.tile(out2[:, 2], (8, 1)).T));
            common.DebugPrint("findquads(): out2!!!! (before /) = %s" % str(out2));

        #out2=out2./repmat(out2(:,3),1,8);
        """
        From http://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html :
          "If A.ndim < d, A is promoted to be d-dimensional by prepending new axes.
           So a shape (3,) array is promoted to (1, 3) for 2-D replication,
            or shape (1, 1, 3) for 3-D replication.
           If this is not the desired behavior,
             promote A to d-dimensions manually before calling this function."
        So np.tile(out2[:, 2], (8, 1)) is exactly what we want:
            3rd column of out2, replicated in 8 columns.
         This is exactly what the Matlab repmat operation does.
        """
        out2 = out2 / np.tile(out2[:, 2], (8, 1)).T; # element wise divide

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): out2.shape (after normalize) = %s" % \
                            str(out2.shape));
            common.DebugPrint("findquads(): out2 (after normalize) = %s" % \
                            str(out2));

        #if False:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): out2!!!! (after /) = %s" % str(out2));


        #% Keep hashing points only y_c, x_c, y_d, x_d
        #out2=out2(:,5:8);
        out2 = out2[:, 4: 8];

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): out2.shape (after slicing) = %s" % \
                                                            str(out2.shape));
            common.DebugPrint("findquads(): out2 (after slicing) = %s" % str(out2));

        #% Roweis constraint (C and D must be inside the circle)
        #s = out2 .^ 2 - out2;
        #s = out2**2 - out2; # element-wise exponentiation  #Gives error: "ValueError: input must be a square array" #numpy seems to consider out2 a matrix
        #NOT GOOD: s = np.multiply(out2, 2) - out2;
        s = out2**2 - out2;

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): s.shape = %s" % str(s.shape));
            common.DebugPrint("findquads(): s = %s" % str(s));

        #index = ((s[:, 1] + s[:, 2]) > 0) | ((s[:, 3] + s[:,4]) > 0);
        index = ((s[:, 0] + s[:, 1]) > 0) | ((s[:, 2] + s[:, 3]) > 0);

        # qq=sum(out2,2);
        # qq_i=diff([qq(1);qq])==0;
        # index=index|qq_i;

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): index (after |) = %s" % str(index));

        notIndex = np.logical_not(index); #.ravel() - not necessary
        #notIndex = notIndex[0];
        #common.DebugPrint("findquads(): notIndex = %s" % str(notIndex));

        # Alex: ~index means Logical NOT of index elements
        #out2=out2(~index,:);
        #out2 = out2[np.logical_not(index), :];
        #out2 = out2[notIndex, :];

        #out2 = out2[notIndex, :]; # THIS IS THE CULPRIT - IT doesn't keep all the columns but only the first column of out2, since index is a column vector...

        try:
            indices = np.nonzero(notIndex);
            #common.DebugPrint("findquads(): indices = %s" % str(indices));
            out2 = out2[indices, :];

            #common.DebugPrint("findquads(): out2 (after index) = %s" % (str(out2)))

            #c2=c2(~index,:);
            #c2 = c2[np.logical_not(index), :]; # IT doesn't keep all the columns but only the first column of out2, since index is a column vector...
            c2 = c2[indices, :];

            #max_dis2=max_dis2(~index,:);
            #max_dis2 = max_dis2[np.logical_not(index), :] # IT doesn't keep all the columns but only the first column of out2, since index is a column vector...
            max_dis2 = max_dis2[indices, :];

            #ori=ori(~index,:);
            #ori = ori[np.logical_not(index), :]; # IT doesn't keep all the columns but only the first column of out2, since index is a column vector...
            ori = ori[indices, :];

            #common.DebugPrint("findquads(): c2 (before) = %s" % str(c2))
            """
            common.DebugPrint("findquads(): max_dis2 (after index) = %s" % \
                                                            (str(max_dis2)));
            """
            #common.DebugPrint("findquads(): ori (after index) = %s" % (str(ori)));

            #    size(out2)
        except:
            common.DebugPrint("findquads(): indices = %s" % str(indices));
            common.DebugPrint("findquads(): out2 = %s" % str(out2));
            common.DebugPrint("findquads(): c2 = %s" % str(c2));
            common.DebugPrint("findquads(): max_dis2 = %s" % str(max_dis2));
            common.DebugPrint("findquads(): ori = %s" % str(ori));

            if out2.size == 0:
                return np.array([]), np.array([]), np.array([]), np.array([]);
            """
            Quads = out2;
            Centroids = c2;
            maxDistances = max_dis2;
            Orientation = ori;
            """






        #% Remove quads with equal centroids
        c2 = np.round(c2 * 1000) / 1000.0;

        #[c2, c2i] = unique(c2, "rows", "first")

        c2 = c2[0]; # c2 is list of list of list.

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): c2.shape (before sort) = %s" % \
                                                        str(c2.shape))
            common.DebugPrint("findquads(): c2 (before sort) = %s" % str(c2))
        c2, c2i = Matlab.unique(c2);

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): c2.shape = %s" % (str(c2.shape)));
            common.DebugPrint("findquads(): c2 (after sort, unique) = %s" % \
                                                                (str(c2)));
            common.DebugPrint("findquads(): c2i.shape = %s" % (str(c2i.shape)));
            common.DebugPrint("findquads(): c2i (after sort, unique) = %s" % \
                                                                (str(c2i)));
        #common.DebugPrint("findquads(): out2 = %s" % (str(out2)));
        #common.DebugPrint("findquads(): out2.shape = %s" % (str(out2.shape)));
        #common.DebugPrint("findquads(): ori.shape = %s" % (str(ori.shape)));

        """
        After operation array = array[indices, :], on numpy 1.6.1 (but NOT on
          numpy 1.8.1),
          out2 has a shape (1, num_rows, 4) instead of (num_rows, 4)
        If we don't check for the .ndim it can lead to runtime exceptions such as
          IndexError: 0-d arrays can only use a single () or a list of newaxes (and a single ...) as an index
            (at max_dis2 = max_dis2[c2i, :];).
        """
        if out2.ndim == 3:
            out2 = out2[0];

        # max_dis2 has a shape (1, num_rows) instead of (num_rows)
        if max_dis2.ndim == 2:
            max_dis2 = max_dis2[0];

        # ori has a shape (1, num_rows) instead of (num_rows)
        if ori.ndim == 2:
            ori = ori[0];

        assert dis2.ndim == 2;

        """
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): out2.shape = %s" % (str(out2.shape)));
            common.DebugPrint("findquads(): out2 = %s" % (str(out2)));
            common.DebugPrint("findquads(): max_dis2.shape = %s" % \
                                            (str(max_dis2.shape)));
            common.DebugPrint("findquads(): max_dis2 = %s" % \
                                            (str(max_dis2)));
            common.DebugPrint("findquads(): dis2.shape = %s" % \
                                            (str(dis2.shape)));
        """

        if c2i.size == 0:
            out2 = np.array([]);
            max_dis2 = np.array([]);
            ori = np.array([]);
            dis2 = np.array([]);
        else:
            #out2=out2(c2i,:);
            #out2 = out2[c2i, :];
            out2 = out2[c2i];

            #max_dis2=max_dis2(c2i,:);
            #max_dis2 = max_dis2[c2i, :];
            max_dis2 = max_dis2[c2i];

            #ori=ori(c2i, :);
            #ori = ori[c2i, :];
            ori = ori[c2i];

            #dis2=dis2(c2i,:);
            dis2 = dis2[c2i, :];

        #if True:
        if common.MY_DEBUG_STDOUT:
            common.DebugPrint("findquads(): out2.shape (before qq) = %s" % \
                                                            (str(out2.shape)));
            common.DebugPrint("findquads(): out2 = %s" % (str(out2)));

        #    size(out2)

        #% remove double quads (if two consecutive quadruples have the same centroids)
        #if size(out2,1)>1
        if out2.shape[0] > 1:
            qq = out2.sum(1);

            #qq_i = (diff([0; qq]) == 0)
            #TODO: change to np.c_ or even do simple arithmetic operation
            qq_i = (np.diff(np.hstack((0, qq))) == 0);

            qq_iN = np.logical_not(qq_i);

            #out2 = out2(~qq_i, :)
            #out2 = out2[np.logical_not(qq_i), :];
            out2 = out2[qq_iN, :];

            #c2 = c2(~qq_i, :)
            #c2 = c2[np.logical_not(qq_i), :];
            c2 = c2[qq_iN, :];

            #max_dis2 = max_dis2(~qq_i, :)
            #max_dis2 = max_dis2[np.logical_not(qq_i), :];
            # numpy 1.8.1 is a bit stricter than numpy 1.6.1 and complains here since max_dis2 is a 1D array
            #max_dis2 = max_dis2[np.logical_not(qq_i)];
            max_dis2 = max_dis2[qq_iN];

            #ori = ori(~qq_i, :)
            #ori = ori[np.logical_not(qq_i), :];
            # numpy 1.8.1 is a bit stricter than numpy 1.6.1 and complains here since ori is a 1D array
            #ori = ori[np.logical_not(qq_i)];
            ori = ori[qq_iN];

        #  size(out2)

        #%% Lang et al paper constraint about symmetry break
        #index=out2(:,2)<out2(:,4);
        # index=(out2(:,2)+out2(:,4))<=1;
        # index=logical(index);
        # out2=out2(index,:);
        # c2=c2(index,:);
        # max_dis2=max_dis2(index,:);
        # ori=ori(index,:);


        #% the folowing part is executed only for database codes construction(kd-tree)
        #% this part produces the reflection codes of the previous part
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if reflect_flag == 1:
            #all_out = np.hstack((out2, 1 - out2))
            #all_out = np.vstack((out2, 1 - out2));
            all_out = np.r_[out2, 1 - out2];

            out2 = all_out;

            #c2=[c2;c2];
            #c2 = np.hstack((c2, c2));
            #c2 = np.vstack((c2, c2));
            c2 = np.r_[c2, c2];

            #max_dis2=[max_dis2;max_dis2];
            #max_dis2 = np.vstack((max_dis2, max_dis2));
            if FOOBAR:
                max_dis2 = np.c_[max_dis2, max_dis2];
            else:
                max_dis2 = np.hstack((max_dis2, max_dis2));

            #ori = np.ravel(ori);

            # We make a distinct copy of ori in ori2
            ori2 = ori.copy(); #ori + 0

            #common.DebugPrint("findquads(): ori = %s" % str(ori));
            #common.DebugPrint("findquads(): ori>0 %s" % str(ori > 0));

            #ori2(ori>0)=ori2(ori>0)-pi;
            ori2[ori > 0] = ori2[ori > 0] - math.pi;

            #ori2(ori<0)=ori2(ori<0)+pi;
            ori2[ori < 0] = ori2[ori < 0] + math.pi;

            #ori=[ori;ori2];
            #ori = np.vstack((ori, ori2));
            if FOOBAR:
                ori = np.c_[ori, ori2];
            else:
                ori = np.hstack((ori, ori2));
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









        """
        out2 = np.array([]);
        c2 = np.array([]);
        max_dis2 = np.array([]);
        ori = np.array([]);
        """
    else:
        out2 = np.array([]);
        c2 = np.array([]);
        max_dis2 = np.array([]);
        ori = np.array([]);

    Quads = out2;
    Centroids = c2;
    maxDistances = max_dis2;
    Orientation = ori;

    #if False:
    #if True:
    if common.MY_DEBUG_STDOUT:
        assert Quads.shape[0] == Centroids.shape[0];
        assert Quads.shape[0] == maxDistances.shape[0];
        assert Quads.shape[0] == Orientation.shape[0];
        common.DebugPrint("findquads(): Quads.shape = %s" % str(Quads.shape));
        common.DebugPrint("findquads(): Quads = %s" % Quads);
        #
        common.DebugPrint("findquads(): Centroids.shape = %s" % str(Centroids.shape));
        common.DebugPrint("findquads(): Centroids = %s" % Centroids);
        #
        common.DebugPrint("findquads(): maxDistances.shape = %s" % str(maxDistances.shape));
        common.DebugPrint("findquads(): maxDistances = %s" % maxDistances);
        #
        common.DebugPrint("findquads(): Orientation.shape = %s" % str(Orientation.shape));
        common.DebugPrint("findquads(): Orientation = %s" % Orientation);

    t2 = float(cv2.getTickCount());
    myTime = (t2 - t1) / cv2.getTickFrequency();
    ##common.DebugPrint("findquads() took %.6f [sec]" % myTime);
    common.DebugPrint("findquads() took %.6f [sec]" % myTime);

    return Quads, Centroids, maxDistances, Orientation;







import unittest

#print "Am here"

class TestSuite(unittest.TestCase):
    def NOTtestFindquads1(self):
        Points = [[ 143.9875,   5.2982],
              [136.2446,   9.2081],
              [179.1950,  11.0207],
              [153.6399,  12.9662],
              [132.6722,  28.5853],
              [151.9186,  29.2249],
              [112.8424,  31.6131],
              [126.9981,  41.3808],
              [139.8522,  42.3463],
              [122.4435,  66.4544],
              [191.3109,  69.0574],
              [111.4996,  75.3657],
              [ 86.4415,  80.2474],
              [127.5941,  80.8626],
              [149.8920,  81.1514],
              [ 97.1595,  82.5343],
              [123.3499,  94.6041],
              [149.7408,  98.9250],
              [ 81.9263, 104.3929],
              [166.2624, 105.2797],
              [ 91.9423, 106.2993],
              [ 96.7691, 107.0736],
              [134.9731, 107.6504],
              [119.6984, 109.6571],
              [120.6563, 117.2485],
              [ 82.7040, 120.6820],
              [120.4497, 125.1228],
              [134.2411, 126.8005],
              [ 98.3621, 129.2707],
              [142.8167, 132.2306],
              [177.1780, 131.8914],
              [ 83.6934, 136.9983],
              [166.6887, 143.0564],
              [129.9157, 144.8874],
              [150.6505, 146.8775],
              [ 82.0735, 153.3622],
              [ 99.1306, 152.8677],
              [130.1120, 156.9786],
              [ 89.6809, 161.0089],
              [116.6665, 163.5011],
              [154.2077, 171.1634],
              [121.3037, 173.5902],
              [ 98.8005, 174.6614],
              [148.4043, 182.6575],
              [ 99.7286, 188.9187],
              [153.4122, 189.9850],
              [ 86.5865, 193.4515],
              [117.9247, 197.3077],
              [125.2097, 197.0618],
              [136.7434, 199.0549],
              [123.6612, 214.6136],
              [126.4476, 225.7383],
              [122.2756, 233.9473],
              [117.0578, 241.5690],
              [137.3026, 249.3599],
              [115.7963, 255.1546],
              [128.3769, 266.2148],
              [120.0243, 269.1458],
              [124.5191, 276.0504],
              [153.6535, 277.2812],
              [131.4233, 314.4435]]

        Points = np.array(Points);

        threshold = 101;
        reflect_flag = 1;

        findquads(Points, threshold, reflect_flag);

        #self.assertTrue((aZero == 0).all());
        self.assertTrue(True);


    def testFindquads2(self):
        """
        harlocsR[89] from _testdata.py (from Matlab, with step = 25, videos
                   from Evangelidis, the 90th ref frame considered).
        """
        Points  = [
              [100.313002, 5.936060, 1.000000],
              [142.308832, 6.062830, 1.000000],
              [76.578778, 10.168268, 1.000000],
              [132.278082, 9.705039, 1.000000],
              [159.938633, 14.917468, 1.000000],
              [132.181898, 28.797677, 1.000000],
              [162.187220, 29.343137, 1.000000],
              [126.291731, 37.137437, 1.000000],
              [126.102350, 42.172245, 1.000000],
              [104.217773, 51.676411, 1.000000],
              [130.557293, 61.590168, 1.000000],
              [142.035865, 63.933301, 1.000000],
              [152.851379, 63.905084, 1.000000],
              [75.643873, 64.665077, 1.000000],
              [98.751256, 66.461474, 1.000000],
              [117.955529, 68.159664, 1.000000],
              [141.753703, 82.341534, 1.000000],
              [104.697987, 91.224013, 1.000000],
              [157.871267, 95.861982, 1.000000],
              [141.507346, 96.978915, 1.000000],
              [116.866147, 98.608169, 1.000000],
              [167.367705, 106.525015, 1.000000],
              [122.710332, 109.255311, 1.000000],
              [155.525476, 122.794894, 1.000000],
              [142.545304, 138.715011, 1.000000],
              [127.648209, 141.347323, 1.000000],
              [133.669520, 144.032605, 1.000000],
              [128.321257, 154.829765, 1.000000],
              [126.040677, 185.195243, 1.000000],
              [149.370481, 230.939712, 1.000000],
              [111.004354, 236.335058, 1.000000],
              [80.654702, 239.596226, 1.000000],
              [96.632221, 246.142941, 1.000000],
              [129.916844, 253.467008, 1.000000],
              [80.624791, 255.333053, 1.000000],
              [106.063790, 259.348836, 1.000000],
              [156.161547, 261.062360, 1.000000],
              [154.279293, 270.003750, 1.000000],
              [159.141369, 290.575220, 1.000000],
              [162.013060, 299.200725, 1.000000],
              [165.179485, 309.869079, 1.000000],
              [100.391362, 5.560034, 2.000000],
              [132.187725, 9.411956, 2.000000],
              [76.626043, 10.702369, 2.000000],
              [160.213799, 15.607047, 2.000000],
              [126.694391, 40.806061, 2.000000],
              [141.496975, 62.789095, 2.000000],
              [76.087424, 64.080214, 2.000000],
              [98.857167, 66.461925, 2.000000],
              [118.281704, 68.462719, 2.000000],
              [142.147474, 82.232078, 2.000000],
              [157.312256, 95.289006, 2.000000],
              [142.814833, 96.492844, 2.000000],
              [117.232617, 96.791485, 2.000000],
              [167.194531, 106.750393, 2.000000],
              [155.588857, 122.867061, 2.000000],
              [125.783582, 138.757770, 2.000000],
              [128.682009, 155.462619, 2.000000],
              [126.669369, 184.977174, 2.000000],
              [148.555098, 230.668795, 2.000000],
              [110.890296, 235.533015, 2.000000],
              [80.883991, 240.046103, 2.000000],
              [128.250031, 253.713436, 2.000000],
              [81.133976, 255.310049, 2.000000],
              [105.279321, 259.929190, 2.000000],
              [156.657178, 261.245340, 2.000000],
              [155.189156, 270.716910, 2.000000],
              [159.236460, 288.203153, 2.000000],
              [132.082758, 9.196925, 3.000000],
              [77.143226, 11.077524, 3.000000],
              [161.562641, 16.372035, 3.000000],
              [127.116798, 39.925660, 3.000000],
              [140.689181, 61.977743, 3.000000],
              [76.947712, 63.569522, 3.000000],
              [98.700133, 66.398537, 3.000000],
              [118.968190, 68.836204, 3.000000],
              [156.790180, 94.913526, 3.000000],
              [118.562504, 96.033263, 3.000000],
              [133.190074, 145.139665, 3.000000],
              [127.216183, 185.035064, 3.000000],
              [144.250606, 228.684977, 3.000000],
              [110.281582, 234.435191, 3.000000],
              [81.274478, 240.476505, 3.000000],
              [126.581194, 254.096457, 3.000000],
              [81.773802, 255.018958, 3.000000],
              [104.277865, 260.765892, 3.000000],
              [155.802530, 271.219380, 3.000000],
              [159.701891, 288.030866, 3.000000],
              [78.127396, 11.564526, 4.000000],
              [165.005847, 16.539178, 4.000000],
              [128.099989, 39.736784, 4.000000],
              [139.862153, 61.069754, 4.000000],
              [119.864572, 69.325379, 4.000000],
              [154.611576, 94.634613, 4.000000],
              [119.065673, 96.072289, 4.000000],
              [134.085682, 144.881898, 4.000000],
              [128.874240, 184.469753, 4.000000],
              [144.235118, 227.344207, 4.000000],
              [109.657801, 233.454048, 4.000000],
              [82.123402, 241.566226, 4.000000],
              [103.083049, 261.903964, 4.000000],
              [160.308515, 287.494072, 4.000000],
              [138.991462, 59.824525, 5.000000],
              [83.458102, 63.687614, 5.000000],
              [150.946716, 94.030645, 5.000000],
              [119.404737, 95.978283, 5.000000],
              [135.708985, 144.921802, 5.000000],
              [131.293835, 183.964861, 5.000000],
              [143.082725, 226.302151, 5.000000],
              [108.886465, 232.302297, 5.000000],
              [83.229426, 246.107643, 5.000000],
              [101.510799, 263.274439, 5.000000],
              [160.305256, 282.889333, 5.000000]];

        pp = np.array(Points);
        assert pp.shape == (113, 3);

        scale_index = 1;

        points = pp[pp[:, 2] == scale_index, 0:2];
        assert points.shape == (41, 2);

        threshold = 101;
        reflect_flag = 1;

        out, cen, maxdis, ori = findquads(points, threshold, reflect_flag);

        common.DebugPrint("out.shape = %s" % str(out.shape));
        common.DebugPrint("out = %s" % str(out));
        #
        common.DebugPrint("cen.shape = %s" % str(cen.shape));
        common.DebugPrint("cen = %s" % str(cen));
        #
        common.DebugPrint("maxdis.shape = %s" % str(maxdis.shape));
        common.DebugPrint("maxdis = %s" % str(maxdis));
        #
        common.DebugPrint("ori.shape = %s" % str(ori.shape));
        common.DebugPrint("ori = %s" % str(ori));

        # Result from the Matlab code
        resOut = [ \
            [ 0.02711088,  0.69631276,  1.11223866,  0.47665969],
            [ 0.33911093,  0.62730067,  1.08724363,  0.31401688],
            [ 0.50463133,  0.46581425,  0.39448411,  1.13836039],
            [ 1.00585667,  0.28526845,  0.54853698,  0.54327512],
            [ 0.24555794,  0.24971583,  0.64106754,  0.35935873],
            [ 0.96541455,  0.32749252,  0.27830757,  1.02167589],
            [ 1.02159427,  0.25494919,  0.41687997,  0.82285937],
            [ 0.85106422,  0.04408734,  0.79206928,  0.50133076],
            [ 0.74876582,  0.41919753,  0.90403754,  0.57081559],
            [ 0.66690618,  0.72007942,  0.80448864,  1.08324955],
            [ 0.82643252,  0.87210841,  0.67316151,  0.46297661],
            [ 0.31286832,  0.60460894,  0.11189213,  0.98902212],
            [ 0.24877075,  0.32970138,  0.98324323,  0.56863287],
            [ 0.36658934,  0.59746129,  1.05312442,  0.15346072],
            [ 1.06111385,  0.72835276,  0.76712532,  0.40506457],
            [ 0.41743512, -0.06048622,  1.02871038,  0.42365196],
            [ 0.21724509,  0.59443119,  0.19287858,  0.86691924],
            [-0.02676006,  0.58710761,  0.63120442,  0.56114979],
            [ 0.42571696, -0.12503031,  0.72533225,  0.4238274 ],
            [ 0.72497692,  0.73661992,  0.49335617,  0.52895766],
            [ 0.97288912,  0.30368724, -0.11223866,  0.52334031],
            [ 0.66088907,  0.37269933, -0.08724363,  0.68598312],
            [ 0.49536867,  0.53418575,  0.60551589, -0.13836039],
            [-0.00585667,  0.71473155,  0.45146302,  0.45672488],
            [ 0.75444206,  0.75028417,  0.35893246,  0.64064127],
            [ 0.03458545,  0.67250748,  0.72169243, -0.02167589],
            [-0.02159427,  0.74505081,  0.58312003,  0.17714063],
            [ 0.14893578,  0.95591266,  0.20793072,  0.49866924],
            [ 0.25123418,  0.58080247,  0.09596246,  0.42918441],
            [ 0.33309382,  0.27992058,  0.19551136, -0.08324955],
            [ 0.17356748,  0.12789159,  0.32683849,  0.53702339],
            [ 0.68713168,  0.39539106,  0.88810787,  0.01097788],
            [ 0.75122925,  0.67029862,  0.01675677,  0.43136713],
            [ 0.63341066,  0.40253871, -0.05312442,  0.84653928],
            [-0.06111385,  0.27164724,  0.23287468,  0.59493543],
            [ 0.58256488,  1.06048622, -0.02871038,  0.57634804],
            [ 0.78275491,  0.40556881,  0.80712142,  0.13308076],
            [ 1.02676006,  0.41289239,  0.36879558,  0.43885021],
            [ 0.57428304,  1.12503031,  0.27466775,  0.5761726 ],
            [ 0.27502308,  0.26338008,  0.50664383,  0.47104234]];
        resOut = np.array(resOut);

        aZero = resOut - out;
        common.DebugPrint("aZero (quads) = %s" % str(aZero));

        # Result has to be identical with the one in Matlab (except the fact Matlab normally has 4 decimals precision :) ):
        # State our expectation
        #self.assertTrue(lF == res);
        self.assertTrue(out.shape == (40, 4));
        if config.KDTREE_IMPLEMENTATION == 0:
            self.assertTrue( (np.abs(aZero) < 1.0e-07).all() );
        else:
            self.assertTrue( (np.abs(aZero) < 1.0e-06).all() );

        # Result from the Matlab code
        resCen = [ \
            [ 89.1880,   33.1110],
            [ 90.9940,  250.1050],
            [ 92.2290,  244.3520],
            [ 99.1420,   62.7410],
            [110.7560,   91.3870],
            [110.9040,  248.8230],
            [111.7570,   57.1170],
            [121.4450,   99.0170],
            [128.7830,   42.4240],
            [128.9200,  156.3510],
            [129.2140,   29.4530],
            [133.0460,  144.7310],
            [133.2650,   20.4260],
            [136.6130,  245.4510],
            [139.8470,  136.7220],
            [149.1780,   15.0070],
            [149.8750,  268.7770],
            [152.1250,   95.4270],
            [155.8270,  115.9740],
            [160.1530,  292.4120],
            [ 89.1880,   33.1110],
            [ 90.9940,  250.1050],
            [ 92.2290,  244.3520],
            [ 99.1420,   62.7410],
            [110.7560,   91.3870],
            [110.9040,  248.8230],
            [111.7570,   57.1170],
            [121.4450,   99.0170],
            [128.7830,   42.4240],
            [128.9200,  156.3510],
            [129.2140,   29.4530],
            [133.0460,  144.7310],
            [133.2650,   20.4260],
            [136.6130,  245.4510],
            [139.8470,  136.7220],
            [149.1780,   15.0070],
            [149.8750,  268.7770],
            [152.1250,   95.4270],
            [155.8270,  115.9740],
            [160.1530,  292.4120]];
        #ans =    40     1

        # Result from the Matlab code
        resMaxdis = [ \
            63.6998,
            32.1837,
            35.8307,
            42.4557,
            49.0444,
            34.0809,
            36.5794,
            37.2565,
            32.8327,
            43.8774,
            33.0493,
            21.4944,
            34.9597,
            51.4841,
            33.4863,
            35.7800,
            47.2344,
            35.2267,
            45.5112,
            41.3287,
            63.6998,
            32.1837,
            35.8307,
            42.4557,
            49.0444,
            34.0809,
            36.5794,
            37.2565,
            32.8327,
            43.8774,
            33.0493,
            21.4944,
            34.9597,
            51.4841,
            33.4863,
            35.7800,
            47.2344,
            35.2267,
            45.5112,
            41.3287];
        #ans = 40 1;

        # Result from the Matlab code
        resOri = [ \
             2.7439,
            -2.2316,
            -1.0119,
             1.4884,
            -2.6312,
            -1.7874,
            -0.8446,
            -1.7259,
             3.0921,
             3.1049,
            -0.1880,
            -0.7232,
             2.6657,
             1.0698,
            -0.9836,
             0.9898,
            -2.4745,
             0.8141,
            -0.3435,
             0.2669,
            -0.3977,
             0.9100,
             2.1297,
            -1.6532,
             0.5104,
             1.3542,
             2.2970,
             1.4157,
            -0.0495,
            -0.0366,
             2.9536,
             2.4184,
            -0.4759,
            -2.0718,
             2.1580,
            -2.1518,
             0.6671,
            -2.3275,
             2.7981,
            -2.8747];

        resCen = np.array(resCen);
        aZero = resCen - cen;
        common.DebugPrint("aZero (cen-troids) = %s" % str(aZero));
        #if config.KDTREE_IMPLEMENTATION == 0:
        self.assertTrue( (np.abs(aZero) < 1.0e-07).all() );

        resMaxdis = np.array(resMaxdis);
        aZero = resMaxdis - maxdis;
        common.DebugPrint("aZero (maxdis) = %s" % str(aZero));
        self.assertTrue( (np.abs(aZero) < 1.0e-04).all() );

        resOri = np.array(resOri);
        aZero = resOri - ori;
        common.DebugPrint("aZero (ori-entation) = %s" % str(aZero));
        self.assertTrue( (np.abs(aZero) < 1.0e-04).all() );


if __name__ == '__main__':
    unittest.main()

    if False:
        import hotshot

        prof = hotshot.Profile("hotshot_edi_stats_findquads");
        prof.runcall(findquads, Points, threshold, reflect_flag);
        print;
        prof.close();

        from hotshot import stats

        s = stats.load("hotshot_edi_stats_findquads");
        s.sort_stats("time").print_stats();
        #s.print_stats()

