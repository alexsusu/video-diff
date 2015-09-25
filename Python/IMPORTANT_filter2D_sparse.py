# TODO: for the Python implementation I still need to do: resize, interp2, mean2, gradient

import numpy as np



"""
  From http://docs.opencv.org/modules/imgproc/doc/filtering.html#filter2d
    dst(x,y) = \sum_{0 <= x' < kernel.cols; 0 <= y' < kernel.rows} kernel(x',y') * src(x + x'- anchor.x, y + y'- anchor.y)
"""
def filter2D(src, dst, kernel):
    # The standard way of computing filter2D
    assert src.shape == dst.shape

    rowsSrc, colsSrc = src.shape
    rowsK, colsK = dx.shape
    print rowsK, colsK

    ay = rowsK / 2;
    ax = colsK / 2;

    for yd in range(rowsSrc):
        for xd in range(colsSrc):
            # we compute the value for the dst point (yd, xd)
            res = 0;
            for yk in range(rowsK):
                ys = yd + yk - ay;
                if not ((ys >= 0) and (ys < rowsSrc)):
                    continue;
                #print "Am here0";
                for xk in range(colsK):
                    xs = xd + xk - ax;
                    if not ((xs >= 0) and (xs < colsSrc)):
                        continue;
                    res += kernel[yk][xk] * src[ys][xs];
                    #print "Am here";
            dst[yd][xd] = res;




def filter2D_sparse(src, kernel):
    # The inverse way of computing filter2D
    #assert src.shape == dst.shape

    rowsSrc, colsSrc = src.shape
    rowsK, colsK = dx.shape
    print rowsK, colsK

    ay = rowsK / 2;
    ax = colsK / 2;

    # We need to initialize now dst to 0 matrix
    """
    for (y = 0; y < rowsSrc; y++)
        for (x = 0; x < colsSrc; x++)
            dst[y][x] = 0;
    """
    dst = np.zeros(src.shape);

    for ys in range(rowsSrc):
        for xs in range(colsSrc):
            # For each source matrix point we add its contribution to the dst
            #     matrix - for this we need to compute various kernel positions
            #     to properly overlap over the point (yx, xs)
            for yk_start in range(-rowsK + 1, 0 + 1):
                #for yk in range(rowsK):
                #for xk in range(rowsK):

                # (yd, xd) is the center of the kernel starting at (yk_start, xk_start)
                yd = ys + yk_start + ay;
                if not ((yd >= 0) and (yd < colsSrc)):
                    continue;

                for xk_start in range(-colsK + 1, 0 + 1):
                    xd = xs + xk_start + ax;
                    if not ((xd >= 0) and (xd < colsSrc)):
                        continue;
                    dst[yd][xd] += kernel[-yk_start][-xk_start] * src[ys][xs];

    return dst;



def warpPerspective():
"""
          // From http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#warpperspective
            dst(x, y) = src (\frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} , \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}})

         - the most efficient way to compute warpPerspective (well, there are 26 operations: 10 subst, 14 muls, 2 divs for each sparse matrix element)
            // Keep in mind that when using dense matrices we perform per each dst pixel these operations: 8 adds, 4 muls, 2 divs
                So we need to be competitive - well, the sparse matrix should have a sparsity factor of ~10% so we're
                Of course, what matters is the data dependency between these ops when paralelizing this loop iteration:
                    - if running at full parallelism we would have an iteration latency of
                        - the dense implementation: t_{MUL} + t_{DIV}.
                        - the sparse implementation: 2 * t_{MUL} + t_{DIV}.
            For each given (known) x_src and y_src (we need to access in ct time each element of the source sparse matrix):
                x_src = (M_{11} x + M_{12} y + M_{13}) / (M_{31} x + M_{32} y + M_{33})
                y_src = (M_{21} x + M_{22} y + M_{23}) / (M_{31} x + M_{32} y + M_{33})
            we compute the x and y of the dst matrix.
            SpMat warpPerspective(SpMat src) {
                // IMPORTANT: assuming we use INTER_NEAREST
                SpMat dst; //create dst sparse matrix (in CSC format)
                for each point (xsrc, ysrc) from the src sparse matrix {
                    /* Using Cramer's formula we compute x and y
                        (see http://www.emathhelp.net/notes/algebra-2/trigonometry/system-of-two-linear-equations-with-two-variables-second-order-determinants/, or http://www.math.fsu.edu/~fusaro/EngMath/Ch6/DCR.html):
                    */
                    a11 = M11 - M31 * xsrc;
                    a21 = M21 - M31 * ysrc;
                    a12 = M12 - M32 * xsrc;
                    a22 = M22 - M32 * ysrc;
                    b1 = M33 * xsrc - M13;
                    b2 = M33 * ysrc - M23;
                    //so for dst we have:
                    x = (b1 * a22 - a12 * b2) / (a11 * a22 - a12 * a21);
                    y = (b2 * a11 - a21 * b1) / (a11 * a22 - a12 * a21);
                    //
                    // !!!!todo: check if x, y is out of bounds
                    dst[x, y] = src[xsrc, ysrc];
                }
            }
         - another way to compute this is to use a DOK representation of the src sparse matrix.
         BUT the biggest disadvantage is that we would have to query each point of the dense dst matrix (in the end dst could become sparse, but clearly this standard way of computing warpPerspective() is at least as inefficient as the complete dense method)
            - less relevant: otherwise, the next best is to use CSC representation and do binary search over the elements of a column

"""



def GaussianBlur():
    pass


def TestFilter2D():
    dst = np.zeros(shp);

    #Matx13f dx(-0.5f, 0.0f, 0.5f);
    #dx = np.array([[-0.5, 0.0, 0.5]]);
    dx = np.array([[-0.5, 2.4, 1.3]]);
    print dx

    filter2D(src, dst, dx);
    print("filter2D() = %s\n\n" % str(dst));

    #dst2 = np.zeros(shp);
    dst2 = filter2D_sparse(src, dx);
    print("filter2D_sparse() = %s\n\n" % str(dst2));

    aZero = dst2 - dst;
    #self.assertTrue(np.abs(aZero == 0).all());
    #self.assertTrue(np.abs(aZero < 1.0e-4).all());
    assert np.abs(aZero < 1.0e-4).all()


    import cv2
    #cv2.filter2D(src, dst, dx);
    # Note: CvPoint is (x, y)
    # See details on borders at http://docs.opencv.org/modules/imgproc/doc/filtering.html
    resF = cv2.filter2D(src=src, ddepth=-1, kernel=dx, anchor=(-1, -1), \
                        borderType=cv2.BORDER_REPLICATE); #cv2.BORDER_ISOLATED); #BORDER_DEFAULT) #BORDER_CONSTANT) #cv2.BORDER_TRANSPARENT)
    #print resF;
    print("resF = %s" % str(resF));

    aZero = resF - dst;
    #self.assertTrue(np.abs(aZero == 0).all());
    #self.assertTrue(np.abs(aZero < 1.0e-4).all());
    assert np.abs(aZero < 1.0e-4).all()



#shp = (1080, 1920);
shp = (10, 10);
src = np.ones(shp) + np.eye(shp[0]);
TestFilter2D();

