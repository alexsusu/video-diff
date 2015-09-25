"""
The most efficient would be to use OpenCV's
    cv::flann::hierarchicalClustering .
   But we do NOT have Python bindings to it.

See if you have the time:
    http://opencvpython.blogspot.ro/2013/01/k-means-clustering-3-working-with-opencv.html

Other ideas at:
    - https://stackoverflow.com/questions/1793532/how-do-i-determine-k-when-using-k-means-clustering
        - "Basically, you want to find a balance between two variables:
            the number of clusters (k) and the average variance of the clusters."
        - "First build a minimum spanning tree of your data. Removing the K-1 most expensive edges splits the tree into K clusters,
            so you can build the MST once, look at cluster spacings / metrics for various K, and take the knee of the curve.
            This works only for Single-linkage_clustering, but for that it's fast and easy. Plus, MSTs make good visuals."

    - https://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters


https://en.wikipedia.org/wiki/Variance

NOT useful:
    http://classroom.synonym.com/calculate-average-variance-extracted-2842.html
"""

import cv2
import numpy as np
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt

import common
import config




colors = ['b', 'r', 'g', 'y', "w", "magenta", "brown", "pink", "orange", \
                                                                    "purple"]


def sqr(r):
    return r * r


"""
This uses SciPy - see scipy-ref.pdf, Section 3.1.1 .
I guess this is similar to cv::flann::hierarchicalClustering() .
Note: the number of clusters is dependent on the threshold var defined below.

IMPORTANT: We return only the elements from Z that are part of the MEANINGFUL
    clusters obtained with hierarchicalClustering().

See if you have the time:
  http://nbviewer.ipython.org/github/herrfz/dataanalysis/tree/master/data/
    http://nbviewer.ipython.org/github/herrfz/dataanalysis/blob/master/week4/clustering_example.ipynb
"""
def HierarchicalClustering(Z, N):
    if False:
        print "Z = %s" % str(Z);

    """
    Z = []
      Traceback (most recent call last):
        File "ReadAVI.py", line 365, in <module>
          Main()
        File "ReadAVI.py", line 295, in Main
          res = MatchFrames.Main_img2(img2, counter2)
        File "/home/alexsusu/drone-diff/02/MatchFrames.py", line 776, in Main_img2
          res = match_and_draw("Image Match")
        File "/home/alexsusu/drone-diff/02/MatchFrames.py", line 593, in match_and_draw
          nonp1 = ClusterUnmatchedKeypoints(nonp1)
        File "/home/alexsusu/drone-diff/02/MatchFrames.py", line 110, in ClusterUnmatchedKeypoints
          Z = Clustering.HierarchicalClustering(Z, N)
        File "/home/alexsusu/drone-diff/02/Clustering.py", line 64, in HierarchicalClustering
          dSch = sch.distance.pdist(Z)
        File "/usr/lib/python2.7/dist-packages/scipy/spatial/distance.py", line 1173, in pdist
          raise ValueError('A 2-dimensional array must be passed.')
      ValueError: A 2-dimensional array must be passed.

    Z = [[ 265.  127.]]
      Traceback (most recent call last):
        File "ReadAVI.py", line 365, in <module>
          Main()
        File "ReadAVI.py", line 295, in Main
          res = MatchFrames.Main_img2(img2, counter2)
        File "/home/alexsusu/drone-diff/02/MatchFrames.py", line 776, in Main_img2
          res = match_and_draw("Image Match")
        File "/home/alexsusu/drone-diff/02/MatchFrames.py", line 593, in match_and_draw
          nonp1 = ClusterUnmatchedKeypoints(nonp1)
        File "/home/alexsusu/drone-diff/02/MatchFrames.py", line 110, in ClusterUnmatchedKeypoints
          Z = Clustering.HierarchicalClustering(Z, N)
        File "/home/alexsusu/drone-diff/02/Clustering.py", line 65, in HierarchicalClustering
          dSch = sch.distance.pdist(Z)
        File "/usr/lib/python2.7/dist-packages/scipy/spatial/distance.py", line 1173, in pdist
          raise ValueError('A 2-dimensional array must be passed.')
      ValueError: A 2-dimensional array must be passed.

    Z = [[ 430.           61.        ]
         [ 265.          127.        ]
         [ 300.           79.        ]
         [ 481.           54.        ]
         [ 450.00003052   91.20000458]
         [ 327.62884521  143.07841492]
         [ 261.27365112   99.53282166]
         [ 292.62652588  119.43939209]
         [ 313.52841187  152.28521729]
         [ 358.31817627  101.52348328]]
      Traceback (most recent call last):
        File "ReadAVI.py", line 365, in <module>
          Main()
        File "ReadAVI.py", line 295, in Main
          res = MatchFrames.Main_img2(img2, counter2)
        File "/home/alexsusu/drone-diff/02/MatchFrames.py", line 776, in Main_img2
          res = match_and_draw("Image Match")
        File "/home/alexsusu/drone-diff/02/MatchFrames.py", line 593, in match_and_draw
          nonp1 = ClusterUnmatchedKeypoints(nonp1)
        File "/home/alexsusu/drone-diff/02/MatchFrames.py", line 110, in ClusterUnmatchedKeypoints
          Z = Clustering.HierarchicalClustering(Z, N)
        File "/home/alexsusu/drone-diff/02/Clustering.py", line 145, in HierarchicalClustering
          numElems[e] += 1
      IndexError: list index out of range
    """

    if False:
        Z = [[ 430.,          61.        ],
             [ 265.,         127.        ],
             [ 300.,          79.        ],
             [ 481.,          54.        ],
             [ 450.00003052,  91.20000458],
             [ 327.62884521, 143.07841492],
             [ 261.27365112,  99.53282166],
             [ 292.62652588, 119.43939209],
             [ 313.52841187, 152.28521729],
             [ 358.31817627, 101.52348328]];

    N = len(Z);
    common.DebugPrint("HierarchicalClustering(): N = %d" % N);

    # Note: Z is not standard list, but a numpy array
    if len(Z) < 10: #or Z == []:
        common.DebugPrint("HierarchicalClustering(): Bailing out of hierarchical " \
              "clustering since too few elements provided (and I guess we" \
              "could have issues)");
        return [];

    # Vector of (N choose 2) pairwise Euclidian distances
    dSch = sch.distance.pdist(Z);
    dMax = dSch.max();

    if False:
        common.DebugPrint("Z = %s" % str(Z));

    # This parameter is CRUCIAL for the optimal number of clusters generated
    #threshold = 0.1 * dMax;
    threshold = 0.05 * dMax; # This parameter works better for the videos from Lucian

    """
    I did not find much information on the linkage matrix (linkageMatrix), but
        from my understanding it is the direct result of the hierarchical
        clustering, which is performed by recursively splitting clusters,
        forming a dendrogram forest of trees (see if you have time
            https://stackoverflow.com/questions/5461357/hierarchical-k-means-in-opencv-without-knowledge-of-k
            "a forest of hierarchical clustering trees").
      The linkage matrix is stores on each row data for a clustered point:
            - the last element in the row is the leaf in the dendrogram tree forrest
                the point belongs to. The leaf does not really tell you to which
                final cluster the point belongs to - (IMPORTANT) for this, we
                have the function sch.fcluster().
      See if you have the time (for some better understanding):
        https://stackoverflow.com/questions/11917779/how-to-plot-and-annotate-hierarchical-clustering-dendrograms-in-scipy-matplotlib

      See doc:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

      See if you have the time:
        https://stackoverflow.com/questions/16883412/how-do-i-get-the-subtrees-of-dendrogram-made-by-scipy-cluster-hierarchy
    """
    linkageMatrix = sch.linkage(dSch, "single");
    common.DebugPrint("linkageMatrix = %s" % str(linkageMatrix));

    # Inspired from https://stackoverflow.com/questions/7664826/how-to-get-flat-clustering-corresponding-to-color-clusters-in-the-dendrogram-cre
    indexCluster = sch.fcluster(linkageMatrix, threshold, "distance");
    common.DebugPrint("indexCluster = %s" % str(indexCluster));

    cMax = -1;

    numElems = [0] * (N + 1); # We "truncate" later the ending zeros from numElems
    # IMPORTANT: It appears the ids of the clusters start from 1, not 0
    for e in indexCluster:
        #print "e = %s" % str(e)
        numElems[e] += 1;
        if cMax < e:
            cMax = e;
    #cMax += 1

    # cMax is the MAXIMUM optimal number of clusters after the Hierarchical clustering

    common.DebugPrint("cMax (the MAX id of final clusters) = %d" % cMax);

    numElems = numElems[0 : cMax + 1];
    common.DebugPrint("numElems = %s" % str(numElems));
    """
    # We can also use:
    numElems.__delslice__(cMax + 1, len(numElems))
    but it's sort of deprecated
        - see http://docs.python.org/release/2.5.2/ref/sequence-methods.html
    """

    numClusters = 0;
    for e in numElems:
        if e != 0:
            numClusters += 1;

    common.DebugPrint("numClusters (the optimal num of clusters) = %d" % \
                                                                numClusters);
    assert numClusters == cMax;

    numClustersAboveThreshold = 0;
    for i in range(cMax + 1):
        if numElems[i] >= \
                        config.THRESHOLD_NUM_NONMATCHED_ELEMENTS_IN_CLUSTER:
            common.DebugPrint("numElems[%d] = %d" % (i, numElems[i]));
            numClustersAboveThreshold += 1;

    common.DebugPrint("numClustersAboveThreshold = %d" % \
                                                    numClustersAboveThreshold)

    RETURN_ONLY_BIGGEST_CLUSTER = False; #True;
    if RETURN_ONLY_BIGGEST_CLUSTER == True:
        # !!!!TODO: find biggest cluster - sort them after numElems, etc
        res = [];
        for i in range(N):
            if indexCluster[i] == numClusters: # We start numbering the clusters from 1
                res.append(Z[i]);
    else:
        if False:
            # We return only the elements from the MEANINGFUL clusters
            res = [];
            for i in range(N):
                if numElems[indexCluster[i]] >= \
                            config.THRESHOLD_NUM_NONMATCHED_ELEMENTS_IN_CLUSTER:
                    res.append(Z[i]);
        else:
            res = {};
            for i in range(N):
                if numElems[indexCluster[i]] >= \
                            config.THRESHOLD_NUM_NONMATCHED_ELEMENTS_IN_CLUSTER:
                    if indexCluster[i] not in res:
                        res[indexCluster[i]] = [];
                    res[indexCluster[i]].append(Z[i]);

    if config.USE_GUI and config.DISPLAY_PYTHON_CLUSTERING:
        # We clear the figure and the axes
        plt.clf();
        plt.cla();

        # Plot the data
        for i in range(N): #indexCluster:
            #print "Z[i, 0] = %.2f, Z[i, 1] = %.2f" % (Z[i, 0], Z[i, 1])
            if False:
                # We plot only the "interesting" clusters
                if numElems[indexCluster[i]] >= \
                        config.THRESHOLD_NUM_NONMATCHED_ELEMENTS_IN_CLUSTER:
                    plt.scatter(Z[i, 0], Z[i, 1], c=colors[indexCluster[i]]);
            else:
                try:
                    colCluster = colors[indexCluster[i]];
                except: # IndexError: list index out of range
                    colCluster = 2;
                plt.scatter(Z[i, 0], Z[i, 1], c=colCluster)

        plt.xlabel("Height. (numClusters = %d, numClustersAboveThreshold = %d)" % \
                                    (numClusters, numClustersAboveThreshold));

        plt.ylabel("Weight");

        # From http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axis
        v = plt.axis();
        # We invert the y to have 0 up and x axis to have 0
        v = (0, v[1], v[3], 0);
        #plt.gca().invert_yaxis()
        plt.axis(v);

        plt.show();
        if False:
            plt.savefig("plot-%s.png" % (prefix));

        if False:
            sch.dendrogram(linkageMatrix,
                       truncate_mode='lastp',
                       color_threshold=1,
                       show_leaf_counts=True)

            plt.show();
            if False:
                plt.savefig("plot-%s.png" % (prefix));

    return res;



"""
This uses Python and OpenCV's cv2 module.
Note: unfortunately, cv::flann::hierarchicalClustering()
    doesn't have Python bindings, so we have to sort of
    implement it :) .
"""
def HierarchicalClusteringWithCV2_UNFINISHED(Z, N):
    #X = np.random.randint(25,50,(25,2))
    #Y = np.random.randint(60,85,(25,2))
    #Z = np.vstack((X, Y))

    # We choose an ~optimal number of clusters
    #k = 4

    minValidity = 1000000
    minValidityK = -1

    for k in range(2, 10 + 1):
        A = [None] * k
        #avg = [0] * k

        """
        Inspired a bit from
            https://www.google-melange.com/gsoc/project/google/gsoc2013/abidrahman2/43002,
            \source\py_tutorials\py_ml\py_kmeans\py_kmeans_opencv\py_kmeans_opencv.rst
        """

        # Define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        t1 = float(cv2.getTickCount())

        #ret, label, center = cv2.kmeans(Z, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        ret, label, center = cv2.kmeans(Z, k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        t2 = float(cv2.getTickCount())
        myTime = (t2 - t1) / cv2.getTickFrequency()
        common.DebugPrint( \
            "HierarchicalClusteringWithCV2_UNFINISHED(): " \
            "cv2.kmeans() took %.5f [sec]" % myTime)

        common.DebugPrint("ret = %s" % str(ret))
        common.DebugPrint("label = %s" % str(label))
        common.DebugPrint("center = %s" % str(center))

        # Now separate the data, Note the flatten()
        for i in range(k):
            A[i] = Z[label.ravel() == i]

        common.DebugPrint("A[0] = %s" % str(A[0]))
        common.DebugPrint("A[0][:, 0] = %s" % str(A[0][:, 0]))
        common.DebugPrint("A[0][:, 1] = %s" % str(A[0][:, 1]))

        """
        Following Section 3.2 from http://www.csse.monash.edu.au/~roset/papers/cal99.pdf :
        See if you have the time, for further ideas:
            https://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters
        """
        intra = 0
        for i in range(k):
            # Gives exception: "TypeError: only length-1 arrays can be converted to Python scalars"
            #for y in range(A[i]):
            for x in range(len(A[i])):
                #intra += sqr(x[0] - center[:,0])
                intra += np.square(A[i][x, 0] - center[i, 0]) + \
                            np.square(A[i][x, 1] - center[i, 1])
                #intra += sqr(A[i][x, 0] - center[i, 0]) + sqr(A[i][x, 1] - center[i, 1])
                #avg[i] += A[i]
        intra /= N

        distMin = 1000000
        for i in range(k):
            for j in range(i + 1, k):
                dist = np.square(center[i, 0] - center[j, 0]) + \
                        np.square(center[i, 1] - center[j, 1])
                """
                dist = sqr(center[i, 0] - center[j, 0]) + \
                        sqr(center[i, 1] - center[j, 1])
                """
                common.DebugPrint("dist = %s" % str(dist))
                if dist < distMin:
                    distMin = dist
        inter = distMin

        """
        We want to minimize intra (clusters be dense) and
            maximize inter (clusters be distant from one another).
        """
        validity = intra / inter

        if minValidity > validity:
            minValidity = validity
            minValidityK = k

        if config.USE_GUI:
            # We clear the figure and the axes
            plt.clf()
            plt.cla()

            # Plot the data
            for i in range(k):
                """
                Note: A[0][:,0] (i.e., [:,0] is a numpy-specific
                    "split"-operator, not working for standard Python lists.
                """
                plt.scatter(A[i][:,0], A[i][:,1], c=colors[i])

            plt.scatter(center[:,0], center[:,1], s=80, c="b", marker="s")

            plt.xlabel(
                "Height. Also: k=%d, intra=%.1f, inter=%.1f, validity = %.4f" % \
                (k, intra, inter, validity))

            plt.ylabel("Weight")

            plt.show()
            if False:
                plt.savefig("plot-%s.png" % (prefix))

        """
        TODO!!!! Implement section 4 from http://www.csse.monash.edu.au/~roset/papers/cal99.pdf:
         See "when we require the number of
            clusters to be increased, we split the cluster
            having maximum variance, so the k-means
            procedure is given good starting cluster centres."
        """
        # !!!!TODO: .... DO THE IMPLEMENTATION, WHITE BOY

    common.DebugPrint("IMPORTANT: minValidityK = %d" % minValidityK)

