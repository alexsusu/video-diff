#import cv
import cv2
import math
import numpy as np
from numpy import linalg as npla
import sys
import scipy.interpolate
import scipy.sparse
import scipy.weave
import unittest

import common
import config




def CompareMatricesWithNanElements(M1, M2):
    assert M1.shape == M2.shape;
    assert M1.ndim == 2;
    for r in range(M1.shape[0]):
        for c in range(M2.shape[0]):
            if np.isnan(M1[r, c]):
                if not np.isnan(M2[r, c]):
                    return False;
            else:
                if M1[r, c] != M2[r, c]:
                    return False;

    return True;


def ConvertCvMatToNPArray(cvmat):
    m = [];
    for r in range(cvmat.rows):
        mR = [cvmat[r, c] for c in range(cvmat.cols)];
        m.append(mR);
    return np.array(m);


def Repr3DMatrix(m):
    assert m.ndim == 3;
    res = "";
    for i in range(m.shape[2]):
        res += ("\n[:, :, %d] = " % i) + str(m[:, :, i]); # + "\n";

    return res;

"""
From http://www.mathworks.com/help/matlab/matlab_prog/symbol-reference.html:
    Dot-Dot-Dot (Ellipsis) - ...
    A series of three consecutive periods (...) is the line continuation operator in MATLAB.
      Line Continuation
      Continue any MATLAB command or expression by placing an ellipsis at the end of the line to be continued:
"""

def fix(x):
    """
    From http://www.mathworks.com/help/matlab/ref/fix.html
      fix
        Round toward zero

      Syntax:
        B = fix(A)

      Description:
        B = fix(A) rounds the elements of A toward zero,
            resulting in an array of integers.
            For complex A, the imaginary and real parts are
                rounded independently.

        Examples:
        a = [-1.9, -0.2, 3.4, 5.6, 7.0, 2.4+3.6i]

        a =
        Columns 1 through 4
        -1.9000        -0.2000        3.4000        5.6000

        Columns 5 through 6
        7.0000        2.4000 + 3.6000i

        fix(a)

        ans =
        Columns 1 through 4
        -1.0000        0             3.0000        5.0000

        Columns 5 through 6
        7.0000        2.0000 + 3.0000i
    """
    if x < 0:
        return math.ceil(x);
    else:
        return math.floor(x);


# eps() is used by fspecial().
def eps(val=1.0):
    """
    Following http://wiki.scipy.org/NumPy_for_Matlab_Users,
        eps is equivalent to spacing(1).
      Note: Matlab's double precision is numpy's float64.
    """
    """
    From NumPy help (see also
      http://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html)
    >>> np.info(np.spacing)
    spacing(x[, out])

    Return the distance between x and the nearest adjacent number.

    Parameters
    ----------
    x1: array_like
        Values to find the spacing of.

    Returns
    -------
    out : array_like
        The spacing of values of `x1`.

    Notes
    -----
    It can be considered as a generalization of EPS:
    ``spacing(np.float64(1)) == np.finfo(np.float64).eps``, and there
    should not be any representable number between ``x + spacing(x)`` and
    x for any finite x.

    Spacing of +- inf and nan is nan.
    """

    """
    From http://www.mathworks.com/help/matlab/ref/eps.html

    """
    epsRes = np.spacing(val);
    return epsRes;


def max(A):
    """
    From http://www.mathworks.com/help/matlab/ref/max.html:
        C = max(A) returns the largest elements along different dimensions of an array.
        If A is a vector, max(A) returns the largest element in A.
        [C,I] = max(...) finds the indices of the maximum values of A,
                and returns them in output vector I.
                If there are several identical maximum values,
                    the index of the first one found is returned.
    """
    assert A.ndim == 1;

    for i in range(A.shape[0]):
        if np.isnan(A[i]):
            A[i] = -1.0e-300;

    C = np.max(A);

    # We find now the index(indices) of C in A
    I = np.nonzero(A == C)[0];

    if False:
        common.DebugPrint("MatlabMax(): a = %s" % str(A));
        common.DebugPrint("MatlabMax(): C = %s" % str(C));
        common.DebugPrint("MatlabMax(): I.shape = %s" % str(I.shape));

    # We want only 1 element, so we make the index also an int
    I = I[0];

    return C, I;


def fliplr(M):
    #fliplr(M);
    return M[:, ::-1];


"""
We convert a tuple of (2 or 3) array indices (or array or indices) into a
    linear (scalar) index (respectively, array of linear indice)
"""
def sub2ind(matrixSize, rowSub, colSub, dim3Sub=None):
    """
    Note that this is a limited implementation of Matlab's sub2ind,
        in the sense we support only 2 and 3 dimensions.
      BUT it is easy to generalize it.
    """
    assert (len(matrixSize) == 2) or (len(matrixSize) == 3);

    """
    Inspired from https://stackoverflow.com/questions/15230179/how-to-get-the-linear-index-for-a-numpy-array-sub2ind
        (see also http://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel_multi_index.html)

    From Matlab help of sub2ind (http://www.mathworks.com/help/matlab/ref/sub2ind.html):
        Convert subscripts to linear indices
        Syntax
            linearInd = sub2ind(matrixSize, rowSub, colSub)
            linearInd = sub2ind(arraySize, dim1Sub, dim2Sub, dim3Sub, ...)

    # Determines whether the multi-index should be viewed as indexing in
    #            C (row-major) order or FORTRAN (column-major) order.
    """
    #return np.ravel_multi_index((rowSub - 1, colSub - 1), dims=matrixSize, order="F");
    if dim3Sub == None:
        res = np.ravel_multi_index((rowSub, colSub), dims=matrixSize, order="F");
    else:
        res = np.ravel_multi_index((rowSub, colSub, dim3Sub), dims=matrixSize, order="F");

    return res;


def find(X):
    """
    find   Find indices of nonzero elements.
    I = find(X) returns the linear indices corresponding to
    the nonzero entries of the array X.  X may be a logical expression.
    Use IND2SUB(SIZE(X),I) to calculate multiple subscripts from
    the linear indices I.

    I = find(X,K) returns at most the first K indices corresponding to
    the nonzero entries of the array X.  K must be a positive integer,
    but can be of any numeric type.

    I = find(X,K,'first') is the same as I = find(X,K).

    I = find(X,K,'last') returns at most the last K indices corresponding
    to the nonzero entries of the array X.

    [I,J] = find(X,...) returns the row and column indices instead of
    linear indices into X. This syntax is especially useful when working
    with sparse matrices.  If X is an N-dimensional array where N > 2, then
    J is a linear index over the N-1 trailing dimensions of X.

    [I,J,V] = find(X,...) also returns a vector V containing the values
    that correspond to the row and column indices I and J.

    Example:
       A = magic(3)
       find(A > 5)

    finds the linear indices of the 4 entries of the matrix A that are
    greater than 5.

       [rows,cols,vals] = find(speye(5))

    finds the row and column indices and nonzero values of the 5-by-5
    sparse identity matrix.

    See also sparse, ind2sub, relop, nonzeros.
    """

    """
    Alex: caution needs to be taken when translating
        find() - in Matlab when find() is supposed to return 1 array the
        indices are of the elements numbered in
        Fortran order (column-major order), while np.nonzero() returns
        invariably a tuple of 2 arrays, the first for the rows, the second
        for the columns;
        but when find is supposed to return 2 arrays, for row and
        column we don't need to worry about this.
    """

    """
    Retrieving indices in "Fortran" (column-major) order, like in Matlab.
    We do this in order to get the harris points sorted like in Matlab.
    """
    c, r = np.nonzero(X.T);

    #return c, r;
    return sub2ind(c, r, X.shape);



"""
This version is at least 50 times faster than ordfilt2_vectorized().
  It is very efficient. It also makes a few assumptions which were
    respected in the code of Evangelidis - check below.
"""
def ordfilt2(A, order, domain):
    """
    common.DebugPrint("Entered Matlab.ordfilt2(order=%d, domain=%s): " \
                        "A.dtype = %s" % \
                                (order, str(domain), str(A.dtype)));
    """
    common.DebugPrint("Entered Matlab.ordfilt2(order=%d): " \
                        "A.dtype = %s" % \
                                (order, str(A.dtype)));

    assert A.ndim == 2;
    assert domain.shape[0] == domain.shape[1];
    assert order == domain.shape[0] * domain.shape[0];
    assert np.abs((domain - 1.0) < 1.0e-5).all(); # !!!!TODO: this is time consuming - take it out if there are issues

    """
    (Documented from http://stackoverflow.com/questions/16685071/implementation-of-matlab-api-ordfilt2-in-opencv)
    See http://docs.opencv.org/modules/imgproc/doc/filtering.html#dilate
        cv2.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
    Inspired from
      http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#dilation
    """
    #(order == domain.shape[0] * domain.shape[0])
    kernel = np.ones(domain.shape, np.uint8);
    res = cv2.dilate(A, kernel, iterations=1);
    return res;



    if False:
        # From http://docs.opencv.org/modules/imgproc/doc/filtering.html#erode
        if (order == 0):
            res = cv2.dilate(A, kernel, iterations=1);
        return res;

    #res = np.zeros(A.shape, dtype=np.int32);
    if False:
        res = np.empty(A.shape, dtype=np.float64); #np.int64);
    else:
        res = np.empty(A.shape, dtype=A.dtype); #np.int64);

    """
    PyArray_Descr *PyArray_DESCR(PyArrayObject* arr)
        Returns a borrowed reference to the dtype property of the array.

    PyArray_Descr *PyArray_DTYPE(PyArrayObject* arr)
        New in version 1.7.
        A synonym for PyArray_DESCR, named to be consistent with the .dtype. usage within Python.
    """

    if False:
        assert A.dtype == np.float64; #np.int64;
        assert res.dtype == np.float64; #np.int64;
    else:
        assert (A.dtype == np.float32) or (A.dtype == np.float64);
        assert res.dtype == A.dtype;

    if A.dtype == np.float32:
        dtypeSize = 4; # np.float32 is 4 bytes
    elif A.dtype == np.float64:
        dtypeSize = 8; # np.float64 is 8 bytes

    common.DebugPrint("Matlab.ordfilt2(): dtypeSize = %d" % dtypeSize);

    # We check to have the matrices in row-major order-style
    assert A.strides == (A.shape[1] * dtypeSize, dtypeSize);
    assert res.strides == (res.shape[1] * dtypeSize, dtypeSize);

    # See http://wiki.scipy.org/Weave, about how to handle NP array in Weave
    CPP_code = """
    int r, c;
    int rd, cd;
    int rdu, cdu;
    int center = domain_array->dimensions[0] / 2;
    int numRows, numCols;

    //int myMax;
    double myMax;

    // On various alternatives of accessing the elements of np.array: https://stackoverflow.com/questions/7744149/typecasting-pyarrayobject-data-to-a-c-array
    //#define elem(row, col) (((double *)npa_array->data)[(row) * npa_array->dimensions[1] + (col)])

    #define MYMIN(a, b) ((a) < (b) ? (a) : (b))
    #define MYMAX(a, b) ((a) > (b) ? (a) : (b))

    //assert(A_array->ndims == 2);
    numRows = A_array->dimensions[0];
    numCols = A_array->dimensions[1];

    #define elemA(row, col) (((double *)A_array->data)[(row) * numCols + (col)])
    #define elemRes(row, col) (((double *)res_array->data)[(row) * numCols + (col)])

    //#define elemA(row, col) (((int *)A_array->data)[(row) * numCols + (col)])
    //#define elemRes(row, col) (((int *)res_array->data)[(row) * numCols + (col)])

    //#define elemA(row, col) (((long long *)A_array->data)[(row) * numCols + (col)])
    //#define elemRes(row, col) (((long long *)res_array->data)[(row) * numCols + (col)])

    /*
    for (r = 0; r < numRows; r++) {
        for (c = 0; c < numCols; c++) {
            printf("A[%d, %d] = %lld ", r, c, elemA(r, c));
        }
        printf("%c", 10);
    }
    */

    for (r = 0; r < numRows; r++) {
        for (c = 0; c < numCols; c++) {
            if (r - center < 0)
                rd = 0;
            else
                rd = -center;

            myMax = -1;
            rdu = MYMIN(center, numRows - r - 1);

            for (; rd <= rdu; rd++) {
                cd = -center;
                if (c - center < 0)
                    cd = 0;
                else
                    cd = -center;

                cdu = MYMIN(center, numCols - c - 1);
                for (; cd <= cdu; cd++) {
                    //printf("r=%d, c=%d, rd = %d, cd = %d\\n", r, c, rd, cd);
                    myMax = MYMAX(myMax, elemA(r + rd, c + cd));
                }
            }
            elemRes(r, c) = myMax;
        }
    }


    //printf("res[1, 0] = %d", elemRes(1, 0));

    //printf("sum = %.2f, npa = %d", sum, PyArray_NDIM(A_array));
    printf("A.shape = %ld, %ld\\n", A_array->dimensions[0], A_array->dimensions[1]);
    //printf("npa[0, 0] = %.3f", ((double *)PyArray_DATA(A_array))[0]);
    //printf("npa[0, 0] = %.3f", ((double *)A_array->data)[0]);
    //printf("npa[1, 0] = %.3f", ((double *)A_array->data)[200]);

    //printf("npa[1, 0] = %.3f", *((double *)PyArray_GETPTR2(A_array, 1, 0)));
    //printf("res[1, 0] = %.3f", *((int *)PyArray_GETPTR2(res_array, 1, 0)));
    """

    if A.dtype == np.float32:
        CPP_code = CPP_code.replace("double", "float");

    #common.DebugPrint("Matlab.ordfilt2(): CPP_code = %s" % CPP_code);

    scipy.weave.inline(CPP_code, ['A', 'res', 'domain']);

    #common.DebugPrint("res[1, 0] = %d" % res[1, 0]);
    #common.DebugPrint("\n\nres = %s" % str(res));

    return res;


# This version is at least 10x faster than the ordfilt2_4_nested_loops()
def ordfilt2_vectorized(A, order, domain):
    """
    OUR ordfilt2 IMPLEMENTATION IS LIMITED w.r.t. the one in Matlab
        - SEE below for details!
    """

    """
    From http://www.mathworks.com/help/images/ref/ordfilt2.html
        2-D order-statistic filtering
        expand all in page
        Syntax

        B = ordfilt2(A, order, domain)
        B = ordfilt2(A, order, domain, S)
        B = ordfilt2(..., padopt)
        Description

        B = ordfilt2(A, order, domain) replaces each element in A by the
                orderth element in the sorted set of neighbors specified
                by the nonzero elements in domain.

        B = ordfilt2(A, order, domain, S) where S is the same size as domain,
                uses the values of S corresponding to the nonzero values of
                domain as additive offsets.

        B = ordfilt2(..., padopt) controls how the matrix boundaries are padded.
            Set padopt to 'zeros' (the default) or 'symmetric'.
            If padopt is 'zeros', A is padded with 0's at the boundaries.
            If padopt is 'symmetric', A is symmetrically extended at the
                    boundaries.

        Class Support

        The class of A can be logical, uint8, uint16, or double.
            The class of B is the same as the class of A, unless the additive
                offset form of ordfilt2 is used, in which case the class of B
                is double.

        Examples
          This examples uses a maximum filter with a [5 5] neighborhood.
              This is equivalent to imdilate(image,strel('square',5)).

          A = imread('snowflakes.png');
          B = ordfilt2(A,25,true(5));
          figure, imshow(A), figure, imshow(B)

        References

        [1] Haralick, Robert M., and Linda G. Shapiro, Computer and Robot Vision,
                            Volume I, Addison-Wesley, 1992.

        [2] Huang, T.S., G.J.Yang, and G.Y.Tang.
            "A fast two-dimensional median filtering algorithm.",
            IEEE transactions on Acoustics, Speech and Signal Processing,
            Vol ASSP 27, No. 1, February 1979

    From type ordfilt2 (extra info):
    %
    %   Remarks
    %   -------
    %   DOMAIN is equivalent to the structuring element used for
    %   binary image operations. It is a matrix containing only 1's
    %   and 0's; the 1's define the neighborhood for the filtering
    %   operation.
    %
    %   For example, B=ORDFILT2(A,5,ONES(3,3)) implements a 3-by-3
    %   median filter; B=ORDFILT2(A,1,ONES(3,3)) implements a 3-by-3
    %   minimum filter; and B=ORDFILT2(A,9,ONES(3,3)) implements a
    %   3-by-3 maximum filter.  B=ORDFILT2(A,4,[0 1 0; 1 0 1; 0 1 0])
    %   replaces each element in A by the maximum of its north, east,
    %   south, and west neighbors.
    %
    %   See also MEDFILT2.

    %   Copyright 1993-2011 The MathWorks, Inc.
    """

    """
    OUR ordfilt2 IMPLEMENTATION IS LIMITED w.r.t. the one in Matlab.

    We assume that:
        - the matrix A contains only positive elements;
        - domain is all ones and;
        - order = num. elements of domain - i.e., we look for the
            maximum in the domain.

    To use their terminology we implement ONLY a N-by-N neighborhood
        maximum filter, where N = domain.shape[0] == domain.shape[1].

    To implement a generic order-th filter we can use:
        - k-th order statistic algorithm - time complexity O(N lg N)
        OR
        - use a MIN-heap of size k - time complexity O(k lg k)
       We can alternate using one or the other depending on the values of
            k and N :))
    """

    # We consider PADOPT is 'zeros', which pads A with zeros at the boundaries.


    res = np.empty(A.shape);

    assert domain.shape[0] == domain.shape[1];
    assert order == domain.shape[0] * domain.shape[1];

    center = domain.shape[0] / 2;

    common.DebugPrint("Matlab.ordfilt2(): domain.shape = %s" % str(domain.shape));
    common.DebugPrint("Matlab.ordfilt2(): center = %s" % str(center));

    numRows, numCols = A.shape;
    #N = numRows;

    common.DebugPrint("ordfilt2(): (numRows, numCols) = %s" % \
                                                str((numRows, numCols)));

    # Vectorized implemetation
    c, r = np.meshgrid(range(numCols), range(numRows));
    common.DebugPrint("Matlab.ordfilt2(): c.shape = %s" % str(c.shape));
    common.DebugPrint("Matlab.ordfilt2(): r.shape = %s" % str(r.shape));

    assert c.shape == A.shape;
    assert r.shape == A.shape;

    res = np.zeros(A.shape);

    nonZeroDomain = np.nonzero(domain == 1);
    #common.DebugPrint("Matlab.ordfilt2(): nonZeroDomain = %s" % str(nonZeroDomain));
    """
    rd = -center;
    while (rd <= center):
        cd = -center;
        while (cd <= center):
            if domain[rd + center, cd + center] == 1:
    """

    if True:
        if True:
            for i in range(len(nonZeroDomain[0])):
                rd = nonZeroDomain[0][i] - center;
                cd = nonZeroDomain[1][i] - center;

                #if (rd == 0) and (cd == 0):
                #    res = ...

                """
                if True:
                    common.DebugPrint("ordfilt2(): (rd, cd) = %s" % \
                                                str((rd, cd)));
                    common.DebugPrint("ordfilt2(): r + rd = %s" % str(r + rd));
                """

                rf = r + rd;
                cf = c + cd;

                """
                if True:
                    common.DebugPrint("ordfilt2(): rf = %s" % str(rf));
                    common.DebugPrint("ordfilt2(): cf = %s" % str(cf));
                    #common.DebugPrint("ordfilt2(): rf[rf < 0] = %s" % str(rf[rf < 0]));
                """

                indRZeroBelow = np.nonzero(rf < 0);
                #common.DebugPrint("ordfilt2(): indRZeroBelow = %s" % str(indRZeroBelow));

                indRZeroAbove = np.nonzero(rf >= numRows);
                #common.DebugPrint("ordfilt2(): indRZeroAbove = %s" % str(indRZeroAbove));

                indCZeroBelow = np.nonzero(cf < 0);
                #common.DebugPrint("ordfilt2(): indCZeroBelow = %s" % str(indCZeroBelow));

                indCZeroAbove = np.nonzero(cf >= numCols);
                #common.DebugPrint("ordfilt2(): indCZeroAbove = %s" % str(indCZeroAbove));

                #common.DebugPrint("ordfilt2(): c + cd = %s" % str(c + cd));

                """
                Note: if we give negative values in matrices as indices,
                    it acts as PADOPT=='symmetric' in ordfilt2.
                BUT if we have values in rf or cf > N then A[rf, cf]
                    gives exception like:
                  "IndexError: index (4) out of range (0<=index<3) in dimension 1"
                """
                rf[indRZeroAbove] = 0;
                cf[indCZeroAbove] = 0;

                """
                if True:
                    common.DebugPrint("ordfilt2(): rf = %s" % str(rf));
                    common.DebugPrint("ordfilt2(): cf = %s" % str(cf));
                """

                #crtD = A[r + rd, c + cd];
                crtD = A[rf, cf];

                # We rectify at the boundaries with 0, since PADOPT is 'zeros'
                indicesR = np.r_[indRZeroBelow[0], indRZeroAbove[0], \
                                            indCZeroBelow[0], indCZeroAbove[0]];
                indicesC = np.r_[indRZeroBelow[1], indRZeroAbove[1], \
                                            indCZeroBelow[1], indCZeroAbove[1]];
                crtD[indicesR, indicesC] = 0.0;

                """
                if True:
                    #common.DebugPrint("ordfilt2(): A[r + rd, c + cd] = %s" % \
                    #                        str(A[r + rd, c + cd]));
                    common.DebugPrint("ordfilt2(): crtD = %s" % str(crtD));
                """

                # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.maximum.html
                res = np.maximum(res, crtD);

            #cd += 1; #!!!!TODO: remove
        #rd += 1; #!!!!TODO: remove

    #common.DebugPrint("ordfilt2(): res = %s" % str(res));
    return res;


def ordfilt2_4_nested_loops(A, order, domain):
    import __builtin__
    res = np.empty(A.shape);

    assert domain.shape[0] == domain.shape[1];
    assert order == domain.shape[0] * domain.shape[1];

    center = domain.shape[0] / 2;

    common.DebugPrint("Matlab.ordfilt2(): domain.shape = %s" % str(domain.shape));
    common.DebugPrint("Matlab.ordfilt2(): center = %s" % str(center));

    numRows, numCols = A.shape;
    #N = numRows;

    common.DebugPrint("ordfilt2(): (numRows, numCols) = %s" % \
                                                str((numRows, numCols)));

    r = 0;
    while (r < numRows):
        c = 0;
        while (c < numCols):
            """
            rd = -center;
            if r + rd < 0: # rd < -r
                rd = 0; #-r;
            """
            if r - center < 0:
                rd = 0; #-r;
            else:
                rd = -center;
            #rdl = - min(center, center + r);

            myMax = -1;
            rdu = min(center, A.shape[0] - r - 1);
            while (rd <= rdu): #center):
                cd = -center;
                """
                if c + cd < 0: # cd < -c
                    cd = 0; #-c;
                """
                if c - center < 0:
                    cd = 0; #-r;
                else:
                    cd = -center;

                cdu = min(center, A.shape[1] - c - 1);
                while (cd <= cdu): #center):
                    #if True:
                    if False:
                        common.DebugPrint("ordfilt2(): (r, c) = %s" % str((r, c)));
                        common.DebugPrint("ordfilt2(): (r+rd, c+cd) = %s" % \
                                                    str((r+rd, c+cd)));

                    myMax = __builtin__.max(myMax, A[r + rd, c + cd]);
                    #if c == A.shape[1] - 1:
                    if False:
                        common.DebugPrint("ordfilt2(): (r, c) = %s" % str((r, c)));
                        common.DebugPrint("ordfilt2(): (r+rd, c+cd) = %s" % \
                                                    str((r+rd, c+cd)));
                        common.DebugPrint("ordfilt2(): A[r + rd, c + cd] = %s" % \
                                                str(A[r + rd, c + cd]));
                        common.DebugPrint("ordfilt2(): myMax = %s" % str(myMax));
                    cd += 1;
                rd += 1;
            res[r, c] = myMax;
            c += 1;
        r += 1;

    return res;


def mean2(x):
    """
    function y = mean2(x) %#codegen
    %MEAN2 Average or mean of matrix elements.
    %   B = MEAN2(A) computes the mean of the values in A.
    %
    %   Class Support
    %   -------------
    %   A can be numeric or logical. B is a scalar of class double.
    %
    %   Example
    %   -------
    %       I = imread('liftingbody.png');
    %       val = mean2(I)
    %
    %   See also MEAN, STD, STD2.

    %   Copyright 1993-2013 The MathWorks, Inc.

    y = sum(x(:),'double') / numel(x);
    """
    y = float(x.sum()) / x.size;
    return y;


def testEqualMatrices(res, res2):
    # Note: res2 is supposed to be correct
    rows, cols = res.shape;

    numMismatches = 0;
    for r in range(rows):
        for c in range(cols):
            if np.isnan(res[r, c]) or np.isnan(res[r, c]):
                if not (np.isnan(res2[r, c]) and np.isnan(res[r, c])):
                    #match = False;
                    numMismatches += 1;
                    common.DebugPrint("testEqualMatrices(): Mismatching nan at r=%d, c=%d, res2[r,c]=%s, res[r,c]=%s" % \
                            (r, c, str(res2[r, c]), str(res[r, c])));
            elif (abs(res2[r, c] - res[r, c]) > 1.e-4):
                #match = False;
                numMismatches += 1;
                common.DebugPrint("testEqualMatrices(): Mismatching vals at r=%d, c=%d, res2[r,c]=%s, res[r,c]=%s" % \
                            (r, c, str(res2[r, c]), str(res[r, c])));

    #V = V[:10, :10];
    #Xq = Xq[:10, :10];
    #Yq = Yq[:10, :10];
    if numMismatches > 0: # match == False:
        common.DebugPrint("testEqualMatrices(): Mismatch with canonical implementation!!!! numMismatches = %d" % numMismatches);
    return numMismatches


def interp2(V, Xq, Yq, interpolationMethod="linear"):
    common.DebugPrint("Entered Matlab.interp2(): " \
                        "V.dtype = %s" % \
                                (str(V.dtype)));

    #res = np.zeros(A.shape, dtype=np.int32);
    if False:
        res = np.empty(V.shape, dtype=np.float64); #np.int64);
    else:
        res = np.empty(V.shape, dtype=V.dtype); #np.int64);

    """
    PyArray_Descr *PyArray_DESCR(PyArrayObject* arr)
        Returns a borrowed reference to the dtype property of the array.

    PyArray_Descr *PyArray_DTYPE(PyArrayObject* arr)
        New in version 1.7.
        A synonym for PyArray_DESCR, named to be consistent with the .dtype. usage within Python.
    """

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("interp2(): V.strides = %s" % str(V.strides));
        common.DebugPrint("interp2(): V.shape = %s" % str(V.shape));
        common.DebugPrint("interp2(): V.dtype = %s" % str(V.dtype));

        common.DebugPrint("interp2(): Xq.strides = %s" % str(Xq.strides));
        common.DebugPrint("interp2(): Xq.shape = %s" % str(Xq.shape));
        common.DebugPrint("interp2(): Xq.dtype = %s" % str(Xq.dtype));

        common.DebugPrint("interp2(): Yq.strides = %s" % str(Yq.strides));
        common.DebugPrint("interp2(): Yq.shape = %s" % str(Yq.shape));
        common.DebugPrint("interp2(): Yq.dtype = %s" % str(Yq.dtype));

        common.DebugPrint("interp2(): res.strides = %s" % str(res.strides));
        common.DebugPrint("interp2(): res.shape = %s" % str(res.shape));
        common.DebugPrint("interp2(): res.dtype = %s" % str(res.dtype));

    if False:
        assert V.dtype == np.float64; #np.int64;
        assert Xq.dtype == np.float64; #np.int64;
        assert Yq.dtype == np.float64; #np.int64;
    else:
	if common.MY_DEBUG_STDOUT:
            common.DebugPrint("interp2(): V.dtype (again) = %s" % str(V.dtype));
            common.DebugPrint("interp2(): V.dtype == np.float32 is %s" % str(V.dtype == np.float32));
            common.DebugPrint("interp2(): V.dtype == float is %s" % str(V.dtype == float));
        """
        if False:
        """
        assert (V.dtype == np.float32) or (V.dtype == np.float64); #np.int64;
        assert Xq.dtype == V.dtype; #np.int64;
        assert Yq.dtype == V.dtype; #np.int64;

    assert V.ndim == 2;
    assert Xq.ndim == 2;
    assert Yq.ndim == 2;

    if V.dtype == np.float32:
        dtypeSize = 4; # np.float32 is 4 bytes
    elif V.dtype == np.float64:
        dtypeSize = 8; # np.float64 is 8 bytes

    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html
    # We check if we have the matrices in row-major order-style
    assert V.strides == (V.shape[1] * dtypeSize, dtypeSize);
    assert res.strides == (res.shape[1] * dtypeSize, dtypeSize);

    """
    # We check if we have the matrices in column-major order (Fortran) style
    assert Xq.strides == (dtypeSize, Xq.shape[0] * dtypeSize);
    assert Yq.strides == (dtypeSize, Yq.shape[0] * dtypeSize);
    """

    assert interpolationMethod == "linear"; # Bilinear interpolation

    CPP_code2_prefix_Row_Major_Order = """
    #define elemXq(row, col) (((double *)Xq_array->data)[(row) * numC + (col)])
    #define elemYq(row, col) (((double *)Yq_array->data)[(row) * numC + (col)])
    """

    CPP_code2_prefix_Fortran_Major_Order = """
    #define elemXq(row, col) (((double *)Xq_array->data)[(col) * numR + (row)])
    #define elemYq(row, col) (((double *)Yq_array->data)[(col) * numR + (row)])
    """

    if (Xq.strides == (dtypeSize, Xq.shape[0] * dtypeSize)):
        assert Yq.strides == (dtypeSize, Yq.shape[0] * dtypeSize);
        CPP_prefix = CPP_code2_prefix_Fortran_Major_Order;
    else:
        CPP_prefix = CPP_code2_prefix_Row_Major_Order;


    # See http://wiki.scipy.org/Weave, about how to handle NP array in Weave

    CPP_code2 = """
    int r, c;
    int rp, cp;
    int numR, numC;

    double x, y;
    double x1, y1;

    double val;

    numR = V_array->dimensions[0];
    numC = V_array->dimensions[1];

    #define elemV(row, col) (((double *)V_array->data)[(row) * numC + (col)])
    #define elemRes(row, col) (((double *)res_array->data)[(row) * numC + (col)])

    for (r = 0; r < numR; r++) {
        for (c = 0; c < numC; c++) {
            x = elemXq(r, c);
            y = elemYq(r, c);

            //printf("interp2(): (initial) x = %.5f, y = %.5f\\n", x, y);

            if ((x < 0.0) or (y < 0.0) or (x >= numC) or (y >= numR)) {
                elemRes(r, c) = NAN;
                continue;
            }

            // Need to check in which square unit (x, y) falls into
            rp = (int)y;
            cp = (int)x;

            // Adjust (x, y) relative to this particular unit square where it falls into
            y -= rp;
            x -= cp;

            assert((x <= 1) && (x >= 0));
            assert((y <= 1) && (y >= 0));

            //printf("interp2(): x = %.5f, y = %.5f\\n", x, y);

            if ((x == 0.0) && (y == 0.0)) {
                val = elemV(r, c);
            }
            else {
                /*
                common.DebugPrint("interp2(): r = %d, c = %d" % (r, c));
                common.DebugPrint("           rp = %d, cp = %d" % (rp, cp));
                common.DebugPrint("           x = %.3f, y = %.3f" % (x, y));
                */

                // First index of f is the col, 2nd index is row
                /*
                f00 = V[rp, cp];
                f01 = V[rp + 1, cp];
                f10 = V[rp, cp + 1];
                f11 = V[rp + 1, cp + 1];

                # As said in https://en.wikipedia.org/wiki/Bilinear_interpolation
                val = f00 * (1 - x) * (1 - y) + f10 * x * (1 - y) + \
                    f01 * (1 - x) * y + f11 * x * y;
                */
                x1 = 1 - x;
                y1 = 1 - y;

                if (
                        //(rp < numR) &&
                        (rp >= 0) &&

                        //(cp < numC) &&
                        (cp >= 0) &&

                        (cp + 1 < numC) &&
                        //(cp + 1 >= 0) &&

                        (rp + 1 < numR) &&
                        //(rp + 1 >= 0) &&

                        (cp + 1 < numC)
                        //&& (cp + 1 >= 0)
                       )
                    {

                    /*
                    printf("interp2(): elemV(rp, cp) = %.5f\\n", elemV(rp, cp));
                    printf("interp2(): elemV(rp, cp + 1) = %.5f\\n", elemV(rp, cp + 1));
                    printf("interp2(): elemV(rp + 1, cp) = %.5f\\n", elemV(rp + 1, cp));
                    printf("interp2(): elemV(rp + 1, cp + 1) = %.5f\\n", elemV(rp + 1, cp + 1));
                    printf("interp2(): at val, x = %.5f, y = %.5f\\n", x, y);
                    printf("interp2(): at val, x1 = %.5f, y1 = %.5f\\n", x1, y1);
                    */
                    val = elemV(rp, cp) * x1 * y1 + elemV(rp, cp + 1) * x * y1 +
                        elemV(rp + 1, cp) * x1 * y + elemV(rp + 1, cp + 1) * x * y;
                }
                else {
                    // If out of the array bounds, assign NaN
                    /*
                    This portion of code can be executed even
                        if we checked for out-of-bounds of (x,y).
                    */
                    // From http://www.cplusplus.com/reference/cmath/NAN/
                    val = NAN;
                }
            }

            elemRes(r, c) = val;
        }
    }
    """

    CPP_code = CPP_prefix + CPP_code2;

    if V.dtype == np.float32:
        CPP_code = CPP_code.replace("double", "float");

    #common.DebugPrint("Matlab.interp2(): CPP_code = %s" % CPP_code);

    #scipy.weave.inline(CPP_prefix + CPP_code2, ["V", "Xq", "Yq", "res"]);
    scipy.weave.inline(CPP_code, ["V", "Xq", "Yq", "res"]);


    #common.DebugPrint("res[1, 0] = %d" % res[1, 0]);
    #common.DebugPrint("\n\nres = %s" % str(res));

    return res;


def interp2_vectorized(V, Xq, Yq, interpolationMethod="linear"):
    #common.DebugPrint("Entered interp2().");
    #common.DebugPrint("interp2(): Xq = %s" % str(Xq));

    #TODO: think if really have to do nan or can replace it with 0 and also change the callers a bit
    """
    From http://www.mathworks.com/help/matlab/ref/interp2.html :
      Vq = interp2(V,Xq,Yq) assumes a default grid of sample points.
        The default grid points cover the rectangular region,
            X=1:n and Y=1:m, where [m,n] = size(V).
        Use this syntax to when you want to conserve memory and are
            not concerned about the absolute distances between points.

      Vq = interp2(X,Y,V,Xq,Yq) returns interpolated values of a function
        of two variables at specific query points using linear interpolation.
        The results always pass through the original sampling of the function.
        X and Y contain the coordinates of the sample points.
        V contains the corresponding function values at each sample point.
        Xq and Yq contain the coordinates of the query points.

      Vq = interp2(___,method) specifies an optional, trailing input
        argument that you can pass with any of the previous syntaxes.
        The method argument can be any of the following strings that
          specify alternative interpolation methods:
          'linear', 'nearest', 'cubic', or 'spline'.
          The default method is 'linear'.

    Alex:
        So, we give the query coordinates xi, yi and
            bidimensional function wimg_time
    """
    assert interpolationMethod == "linear"; # Bilinear interpolation

    if False:
        common.DebugPrint("interp2(): V.shape = %s" % str(V.shape));
        common.DebugPrint("interp2(): Xq[:20, :20] = %s" % str(Xq[:20, :20]));
        common.DebugPrint("interp2(): Xq.shape = %s" % str(Xq.shape));
        common.DebugPrint("interp2(): Yq[:20, :20] = %s" % str(Yq[:20, :20]));
        common.DebugPrint("interp2(): Yq.shape = %s" % str(Yq.shape));

    """
    From https://en.wikipedia.org/wiki/Bilinear_interpolation
        - unit square case
    If we choose a coordinate system in which the four points where
        f is known are (0, 0), (0, 1), (1, 0), and (1, 1), then the
        interpolation formula simplifies to
    f(x, y) = f(0, 0) (1-x) (1-y) + f(1, 0) x (1-y) + f(0, 1)(1-x)y + f(1, 1) xy

    Or equivalently, in matrix operations:
      f(x,y) = [1-x, x] [f(0,0) f(0,1); f(1,0) f(1,1)] [1-y; y]

    One could use the more complex, (yet more accurate I guess),
        barycentric interpolation from
        http://classes.soe.ucsc.edu/cmps160/Fall10/resources/barycentricInterpolation.pdf
    """

    res = np.zeros( V.shape );

    numR, numC = V.shape;

    yM = Yq.copy();
    xM = Xq.copy();

    indY = np.nonzero( np.logical_or(yM < 0, yM > numR - 1) );
    #common.DebugPrint("ind from Yq = %s" % str(ind));
    yM[indY] = 0.0;

    #common.DebugPrint("Xq = %s" % str(Xq));
    indX = np.nonzero( np.logical_or(xM < 0, xM > numC - 1) );
    #common.DebugPrint("ind from Xq = %s" % str(ind));
    xM[indX] = 0.0;

    rpM = yM.astype(int);
    cpM = xM.astype(int);

    #y = Yq[r, c];
    #x = Xq[r, c];

    # Now they contain only fractional part
    yM = yM - rpM;
    xM = xM - cpM;

    y1M = 1.0 - yM;
    x1M = 1.0 - xM;

    if False:
        common.DebugPrint("interp2(): yM = %s" % str(yM));
        common.DebugPrint("interp2(): xM = %s" % str(xM));

        common.DebugPrint("interp2(): y1M = %s" % str(y1M));
        common.DebugPrint("interp2(): x1M = %s" % str(x1M));

    # nan is stronger than any other number - can't mix with other numbers
    zeroRow = np.zeros( (1, V.shape[1]) ); #+ np.nan;
    zeroCol = np.zeros( (V.shape[0], 1) ); #+ np.nan;

    """
    if False:
        V4 = np.zeros((V.shape[0], V.shape[1])); #+ np.nan;
        V4[:-1, :-1] = V[1:, 1:] * xM * yM;
    else:
        zeroCol1 = np.zeros((V.shape[0] - 1, 1)); #+ np.nan;
        #V4 = np.c_[V[1:, 1:] * xM[:-1, :-1] * yM[:-1, :-1], zeroCol1];
        #V4 = np.r_[V4, zeroRow];
        V4 = np.hstack( (V[1:, 1:] * xM[:-1, :-1] * yM[:-1, :-1], zeroCol1) );
        V4 = np.vstack( (V4, zeroRow) );
    """

    """
    Note that each element in Xq and Yq is NOT just x \in [0,1) away from the
        value col, respectively raw of that element.
        Therefore, we need to compute for each result element res[r, c] its
          four neighbors: V00[r, c], V01[r, c], V10[r, c], V11[r, c].
    """
    V00 = V[rpM, cpM];

    # When FASTER is true we have an improvement of 1.13 secs / 10000.
    FASTER = False; #True;

    if FASTER:
        V01 = np.zeros(V.shape);
        V01[:, :-1] = V[:, 1:];
        V01 = V01[rpM, cpM];
    else:
        V01 = np.c_[V[:,1:], zeroCol][rpM, cpM];

    if FASTER:
        V10 = np.zeros(V.shape);
        V10[:-1, :] = V[1:, :];
        V10 = V10[rpM, cpM];
    else:
        V10 = np.r_[V[1:,:], zeroRow][rpM, cpM];

    if FASTER:
        V11 = np.zeros(V.shape);
        V11[:-1, :-1] = V[1:, 1:];
        V11 = V11[rpM, cpM];
    else:
        zeroCol1 = np.zeros((V.shape[0] - 1, 1));
        V11 = np.c_[V[1:, 1:], zeroCol1];
        V11 = np.r_[V11, zeroRow][rpM, cpM];

    if False:
        V01 = np.c_[V00[:,1:], zeroCol];
        V10 = np.r_[V00[1:,:], zeroRow];
        #
        V11 = np.c_[V00[1:, 1:], zeroCol1];
        V11 = np.r_[V11, zeroRow];

    if False:
        common.DebugPrint("interp2(): V = %s" % str(V));
        common.DebugPrint("interp2(): rpM = %s" % str(rpM));
        common.DebugPrint("interp2(): cpM = %s" % str(cpM));
        common.DebugPrint("interp2(): V00 = %s" % str(V00));
        common.DebugPrint("interp2(): V01 = %s" % str(V01));
        common.DebugPrint("interp2(): V10 = %s" % str(V10));
        common.DebugPrint("interp2(): V11 = %s" % str(V11));
        #common.DebugPrint("interp2(): xM = %s" % str(V11));

    """
    x = Xq[r, c];
    y = Yq[r, c];

    rp = int(y);
    cp = int(x);

    y -= rp;
    x -= cp;

    x1 = 1 - x;
    y1 = 1 - y;

    val = V[rp, cp] * x1 * y1 + \
            V[rp, cp + 1] * x * y1 + \
            V[rp + 1, cp] * x1 * y + \
            V[rp + 1, cp + 1] * x * y;
    """

    if False:
        common.DebugPrint("V[:,1:].shape = %s" % str(V[:,1:].shape));
        common.DebugPrint("xM[:,:-1].shape = %s" % str(xM[:,:-1].shape));
        common.DebugPrint("y1M[:,:-1].shape = %s" % str(y1M[:,:-1].shape));
        common.DebugPrint("nanCol.shape = %s" % str(nanCol.shape));

        common.DebugPrint("V[1:,:].shape = %s" % str(V[1:,:].shape));
        common.DebugPrint("x1M[:-1,:].shape = %s" % str(x1M[:-1,:].shape));
        common.DebugPrint("yM[:-1,:].shape = %s" % str(yM[:-1,:].shape));
        common.DebugPrint("nanRow.shape = %s" % str(nanRow.shape));

    if False:
        res = V00 * x1M * y1M + \
            np.c_[V01[:,1:] * xM[:,:-1] * y1M[:,:-1], zeroCol] + \
            np.r_[V10[1:,:] * x1M[:-1,:] *  yM[:-1,:], zeroRow] + \
            V11;
    res = V00 * x1M * y1M + \
            V01 * xM * y1M + \
            V10 * x1M * yM + \
            V11 * xM * yM;

    if False:
        #common.DebugPrint("Yq = %s" % str(Yq));
        ind = np.nonzero( np.logical_or(Yq < 0, Yq > numR - 1) );
        #common.DebugPrint("ind from Yq = %s" % str(ind));
        res[ind] = np.nan;

        #common.DebugPrint("Xq = %s" % str(Xq));
        ind = np.nonzero( np.logical_or(Xq < 0, Xq > numC - 1) );
        #common.DebugPrint("ind from Xq = %s" % str(ind));
        res[ind] = np.nan;

    res[indY] = np.nan;
    res[indX] = np.nan;

    """
    # Following is a DIDACTIC example of ONE bilinear interpolation:

    x = Xq[0, 0]; # - 1;
    y = Yq[0, 0]; # - 1;

    # First index of f is the col, 2nd index is row
    f00 = V[0, 0];
    f01 = V[1, 0];
    f10 = V[0, 1];
    f11 = V[1, 1];

    # As said, https://en.wikipedia.org/wiki/Bilinear_interpolation
    val = f00 * (1 - x) * (1 - y) + f10 * x * (1 - y) + f01 * (1 - x) * y + f11 * x * y;
    common.DebugPrint("interp2-related: val (bilinear interpolation) = %.5f" % val);
    """

    """
    # Another a bit more complex formula for bilinear interpolation,
        for the general (NOT unit square) case:
        from http://www.ajdesigner.com/phpinterpolation/bilinear_interpolation_equation.php#ajscroll
    #       it has even a calculator
    x1 = 0.0;
    x2 = 1.0;
    y1 = 0.0;
    y2 = 1.0;

    Q11 = V[0, 0];
    Q12 = V[1, 0];
    Q21 = V[0, 1];
    Q22 = V[1, 1];

    common.DebugPrint("Q11 = %s" % str(Q11));
    common.DebugPrint("Q21 = %s" % str(Q21));
    common.DebugPrint("Q12 = %s" % str(Q12));
    common.DebugPrint("Q22 = %s" % str(Q22));

    nom = (x2 - x1) * (y2 - y1);
    common.DebugPrint("interp2-related: nom (bilinear interpolation) = %.5f" % nom);
    val2 = (x2 - x) * (y2 - y) / ( (x2 - x1) * (y2 - y1) ) * Q11 + \
            (x - x1) * (y2 - y) / ( (x2 - x1) * (y2 - y1) ) * Q21 + \
            (x2 - x) * (y - y1) / ( (x2 - x1) * (y2 - y1) ) * Q12 + \
            (x - x1) * (y - y1) / ( (x2 - x1) * (y2 - y1) ) * Q22;

    common.DebugPrint("interp2-related: val2 (bilinear interpolation) = %.5f" % val2);
    """

    return res;


def interp2VectorizedWithTest(V, Xq, Yq, interpolationMethod="linear"):
    common.DebugPrint("Entered interp2VectorizedWithTest()");
    res = interp2Orig(V, Xq, Yq, interpolationMethod);
    res2 = interp2_nested_loops(V, Xq, Yq, interpolationMethod);
    resN = testEqualMatrices(res, res2);

    if resN != 0:
        common.DebugPrint("testEqualMatrices(): V.shape = %s" % str(V.shape));
        common.DebugPrint("testEqualMatrices(): V = %s" % str(V));

        common.DebugPrint("testEqualMatrices(): Xq.shape = %s" % str(Xq.shape));
        common.DebugPrint("testEqualMatrices(): Xq = %s" % str(Xq));

        common.DebugPrint("testEqualMatrices(): Yq.shape = %s" % str(Yq.shape));
        common.DebugPrint("testEqualMatrices(): Yq = %s" % str(Yq));

    return res;

#TEST_INTERP2 = True
TEST_INTERP2 = False
if TEST_INTERP2:
    interp2Orig = interp2;
    interp2 = interp2VectorizedWithTest;
else:
    interp2Orig = interp2;


# TODO: remove
# This implementation is BAD: it assumes each query is placed in "its own square"
def interp2_BAD(V, Xq, Yq, interpolationMethod="linear"):
    common.DebugPrint("Entered interp2().");
    #common.DebugPrint("interp2(): Xq = %s" % str(Xq));

    """
    From http://www.mathworks.com/help/matlab/ref/interp2.html :
      Vq = interp2(V,Xq,Yq) assumes a default grid of sample points.
        The default grid points cover the rectangular region,
            X=1:n and Y=1:m, where [m,n] = size(V).
        Use this syntax to when you want to conserve memory and are
            not concerned about the absolute distances between points.

      Vq = interp2(X,Y,V,Xq,Yq) returns interpolated values of a function
        of two variables at specific query points using linear interpolation.
        The results always pass through the original sampling of the function.
        X and Y contain the coordinates of the sample points.
        V contains the corresponding function values at each sample point.
        Xq and Yq contain the coordinates of the query points.

      Vq = interp2(___,method) specifies an optional, trailing input
        argument that you can pass with any of the previous syntaxes.
        The method argument can be any of the following strings that
          specify alternative interpolation methods:
          'linear', 'nearest', 'cubic', or 'spline'.
          The default method is 'linear'.

    Alex:
        So, we give the query coordinates xi, yi and
            bidimensional function wimg_time
    """
    assert interpolationMethod == "linear"; # Bilinear interpolation


    if False:
        common.DebugPrint("interp2(): V.shape = %s" % str(V.shape));
        common.DebugPrint("interp2(): Xq[:20, :20] = %s" % str(Xq[:20, :20]));
        common.DebugPrint("interp2(): Xq.shape = %s" % str(Xq.shape));
        common.DebugPrint("interp2(): Yq[:20, :20] = %s" % str(Yq[:20, :20]));
        common.DebugPrint("interp2(): Yq.shape = %s" % str(Yq.shape));

    """
    From https://en.wikipedia.org/wiki/Bilinear_interpolation
        - unit square case
    If we choose a coordinate system in which the four points where
        f is known are (0, 0), (0, 1), (1, 0), and (1, 1), then the
        interpolation formula simplifies to
    f(x, y) = f(0, 0) (1-x) (1-y) + f(1, 0) x (1-y) + f(0, 1)(1-x)y + f(1, 1) xy

    Or equivalently, in matrix operations:
      f(x,y) = [1-x, x] [f(0,0) f(0,1); f(1,0) f(1,1)] [1-y; y]

    One could use the more complex, (yet more accurate I guess),
        barycentric interpolation from
        http://classes.soe.ucsc.edu/cmps160/Fall10/resources/barycentricInterpolation.pdf
    """

    res = np.zeros( V.shape );

    numR, numC = V.shape;

    yM = Yq.copy();
    xM = Xq.copy();

    rpM = yM.astype(int);
    cpM = xM.astype(int);

    #y = Yq[r, c];
    #x = Xq[r, c];

    yM = yM - rpM;
    xM = xM - cpM;

    y1M = 1.0 - yM;
    x1M = 1.0 - xM;

    common.DebugPrint("interp2(): yM = %s" % str(yM));
    common.DebugPrint("interp2(): xM = %s" % str(xM));

    common.DebugPrint("interp2(): y1M = %s" % str(y1M));
    common.DebugPrint("interp2(): x1M = %s" % str(x1M));

    # nan is stronger than any other number - can't mix with other numbers
    zeroRow = np.zeros( (1, V.shape[1]) ); #+ np.nan;
    zeroCol = np.zeros( (V.shape[0], 1) ); #+ np.nan;

    if False:
        V4 = np.zeros((V.shape[0], V.shape[1])); #+ np.nan;
        V4[:-1, :-1] = V[1:, 1:] * xM * yM;
    else:
        zeroCol1 = np.zeros((V.shape[0] - 1, 1)); #+ np.nan;
        #V4 = np.c_[V[1:, 1:] * xM[:-1, :-1] * yM[:-1, :-1], zeroCol1];
        #V4 = np.r_[V4, zeroRow];
        V4 = np.hstack( (V[1:, 1:] * xM[:-1, :-1] * yM[:-1, :-1], zeroCol1) );
        V4 = np.vstack( (V4, zeroRow) );

    """
    x = Xq[r, c];
    y = Yq[r, c];

    rp = int(y);
    cp = int(x);

    y -= rp;
    x -= cp;

    x1 = 1 - x;
    y1 = 1 - y;

    val = V[rp, cp] * x1 * y1 + \
            V[rp, cp + 1] * x * y1 + \
            V[rp + 1, cp] * x1 * y + \
            V[rp + 1, cp + 1] * x * y;
    """

    if False:
        common.DebugPrint("V[:,1:].shape = %s" % str(V[:,1:].shape));
        common.DebugPrint("xM[:,:-1].shape = %s" % str(xM[:,:-1].shape));
        common.DebugPrint("y1M[:,:-1].shape = %s" % str(y1M[:,:-1].shape));
        common.DebugPrint("nanCol.shape = %s" % str(nanCol.shape));

        common.DebugPrint("V[1:,:].shape = %s" % str(V[1:,:].shape));
        common.DebugPrint("x1M[:-1,:].shape = %s" % str(x1M[:-1,:].shape));
        common.DebugPrint("yM[:-1,:].shape = %s" % str(yM[:-1,:].shape));
        common.DebugPrint("nanRow.shape = %s" % str(nanRow.shape));

    res = V * x1M * y1M + \
            np.c_[V[:,1:] *  xM[:,:-1] * y1M[:,:-1], zeroCol] + \
            np.r_[V[1:,:] * x1M[:-1,:] *  yM[:-1,:], zeroRow] + \
            V4;

    #common.DebugPrint("Yq = %s" % str(Yq));
    ind = np.nonzero( np.logical_or(Yq < 0, Yq > numR - 1) );
    #common.DebugPrint("ind from Yq = %s" % str(ind));
    res[ind] = np.nan;

    #common.DebugPrint("Xq = %s" % str(Xq));
    ind = np.nonzero( np.logical_or(Xq < 0, Xq > numC - 1) );
    #common.DebugPrint("ind from Xq = %s" % str(ind));
    res[ind] = np.nan;

    return res;


"""
This implementation is for the sake of documenting the algorithm,
    but is VERY VERY slow - 1.55 secs / call for resolution of 320x240.
    (So we will need to implement it in C++ in OpenCV, preferably vectorized :) .)
"""
def interp2_nested_loops(V, Xq, Yq, interpolationMethod="linear"):
    assert interpolationMethod == "linear"; # Bilinear interpolation
    """
    From https://en.wikipedia.org/wiki/Bilinear_interpolation
        - unit square case
    If we choose a coordinate system in which the four points where
        f is known are (0, 0), (0, 1), (1, 0), and (1, 1), then the
        interpolation formula simplifies to
    f(x, y) = f(0, 0) (1-x) (1-y) + f(1, 0) x (1-y) + f(0, 1)(1-x)y + f(1, 1) xy

    Or equivalently, in matrix operations:
      f(x,y) = [1-x, x] [f(0,0) f(0,1); f(1,0) f(1,1)] [1-y; y]

    One could use the more complex, (yet more accurate I guess),
        barycentric Interpolation from
        http://classes.soe.ucsc.edu/cmps160/Fall10/resources/barycentricInterpolation.pdf
    """

    res = np.zeros( V.shape );

    numR, numC = V.shape;

    for r in range(numR):
        for c in range(numC):
            x = Xq[r, c];
            y = Yq[r, c];

            if (x < 0.0) or (y < 0.0) or (x >= numC) or (y >= numR):
                res[r, c] = np.nan;
                continue;

            # Need to check in which square unit (x, y) falls into
            rp = int(y);
            cp = int(x);

            # Adjust (x, y) relative to this particular unit square where it falls into
            y -= rp;
            x -= cp;

            if not((y <= 1) and (y >= 0)) or \
                        not((x <= 1) and (x >= 0)):
                common.DebugPrint("interp2(): r = %d, c = %d" % (r, c));
                common.DebugPrint("           rp = %d, cp = %d" % (rp, cp));
                common.DebugPrint("           x = %.3f, y = %.3f" % (x, y));
                common.DebugPrint("           Xq[r, c] = %.3f, Yq[r, c] = %.3f" % \
                                                (Xq[r, c], Yq[r, c]));
                #sys.stdout.flush();

            assert (x <= 1) and (x >= 0);
            assert (y <= 1) and (y >= 0);

            if (x == 0.0) and (y == 0.0):
                val = V[r, c];
            else:
                """
                common.DebugPrint("interp2(): r = %d, c = %d" % (r, c));
                common.DebugPrint("           rp = %d, cp = %d" % (rp, cp));
                common.DebugPrint("           x = %.3f, y = %.3f" % (x, y));
                """

                # First index of f is the col, 2nd index is row
                try:
                    """
                    f00 = V[rp, cp];
                    f01 = V[rp + 1, cp];
                    f10 = V[rp, cp + 1];
                    f11 = V[rp + 1, cp + 1];

                    # As said in https://en.wikipedia.org/wiki/Bilinear_interpolation
                    val = f00 * (1 - x) * (1 - y) + f10 * x * (1 - y) + \
                        f01 * (1 - x) * y + f11 * x * y;
                    """
                    x1 = 1 - x;
                    y1 = 1 - y;

                    val = V[rp, cp] * x1 * y1 + V[rp, cp + 1] * x * y1 + \
                        V[rp + 1, cp] * x1 * y + V[rp + 1, cp + 1] * x * y;
                except:
                    # If out of the array bounds, assign NaN
                    """
                    This portion of code can be executed even
                        if we checked for out-of-bounds of (x,y).
                    """
                    val = np.nan;

            res[r, c] = val;

    return res;


"""
This is a slightly optimized version of the interp2_nested_loops.
  1.135 secs / call for resolution of 320x240.
The gain in performance is around 27% .
"""
def interp2_1loop(V, Xq, Yq, interpolationMethod="linear"):
    assert interpolationMethod == "linear"; # Bilinear interpolation
    res = np.zeros( V.shape );

    numR, numC = V.shape;

    #for r in range(numR):
    #    for c in range(numC):
    numPixels = numR * numC;

    i = 0;
    r = 0;
    c = -1;

    while i < numPixels:
        i += 1;
        c += 1;

        if c == numC:
            c = 0;
            r += 1;

        #common.DebugPrint("interp2(): r = %d, c = %d, i=%d" % (r, c, i));

        x = Xq[r, c];
        y = Yq[r, c];

        if (x < 0.0) or (y < 0.0) or (x >= numC) or (y >= numR):
            res[r, c] = np.nan;
            continue;

        # Need to check in which square unit (x, y) falls into
        rp = int(y);
        cp = int(x);

        # Adjust (x, y) relative to this particular unit square where it falls into
        y -= rp;
        x -= cp;

        """
        if not((y <= 1) and (y >= 0)) or \
                    not((x <= 1) and (x >= 0)):
            common.DebugPrint("interp2(): r = %d, c = %d" % (r, c));
            common.DebugPrint("           rp = %d, cp = %d" % (rp, cp));
            common.DebugPrint("           x = %.3f, y = %.3f" % (x, y));
            common.DebugPrint("           Xq[r, c] = %.3f, Yq[r, c] = %.3f" % \
                                            (Xq[r, c], Yq[r, c]));
            #sys.stdout.flush();
        """

        # For efficiency reasons we do not do the asserts
        #assert (x <= 1) and (x >= 0);
        #assert (y <= 1) and (y >= 0);

        if (x == 0.0) and (y == 0.0):
            val = V[r, c];
        else:
            """
            common.DebugPrint("interp2(): r = %d, c = %d" % (r, c));
            common.DebugPrint("           rp = %d, cp = %d" % (rp, cp));
            common.DebugPrint("           x = %.3f, y = %.3f" % (x, y));
            """

            # First index of f is the col, 2nd index is row
            try:
                """
                f00 = V[rp, cp];
                f01 = V[rp + 1, cp];
                f10 = V[rp, cp + 1];
                f11 = V[rp + 1, cp + 1];

                # As said in https://en.wikipedia.org/wiki/Bilinear_interpolation
                val = f00 * (1 - x) * (1 - y) + f10 * x * (1 - y) + \
                    f01 * (1 - x) * y + f11 * x * y;
                """
                x1 = 1 - x;
                y1 = 1 - y;

                val = V[rp, cp] * x1 * y1 + V[rp, cp + 1] * x * y1 + \
                    V[rp + 1, cp] * x1 * y + V[rp + 1, cp + 1] * x * y;
            except:
                # If out of the array bounds, assign NaN
                """
                This portion of code can be executed even
                    if we checked for out-of-bounds of (x,y).
                """
                val = np.nan;

        res[r, c] = val;

    return res;


def interp2_scipy(V, Xq, Yq, interpolationMethod="linear"):
    assert interpolationMethod == "linear"; # Bilinear interpolation

    ###UNFORTUNATELY, THIS IMPLEMENTATION WHEN TRYING TO BE EFFICIENT CRASHES##

    # Unfortunately, scipy.interpolate.interp2d (from scipy v0.14 - latest Mar 2014)
    #   crashes in some cases, hence we don't really use this implementation yet

    # See http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html
    if False:
        #yVals = np.array([[[i] * V.shape[1]] for i in range(V.shape[0])]);
        yList = [];
        for i in range(V.shape[0]):
            yList += [i] * V.shape[1];
        #yVals = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]);
        xVals = np.array(range(V.shape[1]) * V.shape[0]);
    else:
        yVals = np.array(range(V.shape[0]));
        xVals = np.array(range(V.shape[1]));

    f = scipy.interpolate.interp2d(x=xVals, y=yVals, z=V, \
                                    kind="linear", copy=False); # Gives strange errors: "cubic");

    # We flatten the query matrices to 1D arrays:
    Xqf = np.ravel(Xq);
    Yqf = np.ravel(Yq);

    if False:
        # Copied from scipy\interpolate\fitpack.py
        Xqf = np.atleast_1d(Xqf);
        Yqf = np.atleast_1d(Yqf);
        Xqf,Yqf = map(np.atleast_1d,[Xqf,Yqf]);

    if False:
        common.DebugPrint("interp2(): xVals = %s" % str(xVals));
        common.DebugPrint("interp2(): yVals = %s" % str(yVals));

        common.DebugPrint("interp2(): Xqf = %s" % str(Xqf));
        common.DebugPrint("interp2(): Xqf.shape = %s" % str(Xqf.shape));
        common.DebugPrint("interp2(): len(Xqf.shape) = %s" % str(len(Xqf.shape)));
        common.DebugPrint("interp2(): Yqf = %s" % str(Yqf));
        common.DebugPrint("interp2(): Yqf.shape = %s" % str(Yqf.shape));
        common.DebugPrint("interp2(): len(Yqf.shape) = %s" % str(len(Yqf.shape)));

    #zNew = f(Xq, Yq); # ValueError: First two entries should be rank-1 arrays.

    if False:
        # It is more efficient like this - but the SciPy implementation (v0.11, at least ) has bugsTODO
        # Don't understand/see a reason why - I reported to scipy community on forum
        zNew = f(Xqf, Yqf); # Gives: ValueError: Invalid input data
    else:
        zNew = np.zeros( V.shape );

        numRows, numCols = V.shape;
        r =0; c = 0;
        for i in range(len(Xqf)):
            #print f(Xqf[i], Yqf[i]);
            #zNew[i / numRows, i % numRows] = f(Xqf[i], Yqf[i]);
            zNew[r, c] = f(Xqf[i], Yqf[i]);
            c += 1;
            if c == numCols:
                r += 1;
                c = 0;

    #zNew = f(Xqf, Yqf); # ValueError: Invalid input data
    #zNew = f(0.7, 0.3);

    return zNew;

def interp2_OpenCV(V, Xq, Yq, interpolationMethod="linear"):
    assert interpolationMethod == "linear"; # Bilinear interpolation

    ####UNFORTUNATELY, IMPLEMENTATION WITH OpenCV REMAP GIVES WRONG RESULTS####
    # BUT MAYBE USEFUL FOR OTHER PROCEDURES...

    """
    From https://stackoverflow.com/questions/19912234/cvremap-in-opencv-and-interp2-matlab

      <<In case you didn't find your answer yet, this is how you should use it.
          remap(f,f2,x2,y2,CV_INTER_CUBIC);
      The function remap supposes that you're working on exactly the grid where
          f is defined so no need to pass the x,y monotonic coordinates.
      I'm almost sure that the matrices cannot be CV_64F (double) so, take that
          into account.>>

      Alex: VERY GOOD explanation:
        <<interp2 is interpolator - if you get x,y and f values for some mesh
            it gives you the value f2 for x2 and y2.
        remap - wraps your mesh by moving x and y coordinates acording to the
            deformation maps.
        if you want interpolate regular mesh then use scaling (cv::resize for example).
        If data is scattered then you can use Delaunay triangulation and then
            barycentric interpolation as variant or inverse distance weighting.>>
    """

    #OpenCV Error: Assertion failed (ifunc != 0) in unknown function, file ..\..\..\src\opencv\modules\imgproc\src\imgwarp.cpp, line 2973
    # error: ..\..\..\src\opencv\modules\imgproc\src\imgwarp.cpp:2973: error: (-215) ifunc != 0

    V = V.astype(float);

    #Xq = cv.fromarray(Xq);
    #Yq = cv.fromarray(Yq);

    #Xq = Xq.astype(float);
    #Yq = Yq.astype(float);

    common.DebugPrint("cv2.INTER_LINEAR = %s" % str(cv2.INTER_LINEAR));

    # Inspired from https://stackoverflow.com/questions/12535715/set-type-for-fromarray-in-opencv-for-python
    r, c = Xq.shape[0], Xq.shape[1];
    Xq_32FC1 = cv.CreateMat(r, c, cv.CV_32FC1);
    cv.Convert(cv.fromarray(Xq), Xq_32FC1);

    r, c = Yq.shape[0], Yq.shape[1];
    Yq_32FC1 = cv.CreateMat(r, c, cv.CV_32FC1);
    cv.Convert(cv.fromarray(Yq), Yq_32FC1);

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("interp2_OpenCV(): Xq_32FC1 = %s" % str(ConvertCvMatToNPArray(Xq_32FC1)));
        common.DebugPrint("interp2_OpenCV(): Yq_32FC1 = %s" % str(ConvertCvMatToNPArray(Yq_32FC1)));

    """
    From http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html?highlight=remap#remap
    cv2.remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue ] ] ])
    """
    """
    Gives error:
      OpenCV Error: Assertion failed (((map1.type() == CV_32FC2 || map1.type() == CV_16SC2) && !map2.data) || (map1.type() == CV_32FC1
      && map2.type() == CV_32FC1)) in unknown function, file ..\..\..\src\opencv\modules\imgproc\src\imgwarp.cpp, line 2988
    dst = cv2.remap(src=V, map1=Xq, map2=Yq, interpolation=cv2.INTER_LINEAR);
    """
    # Gives error: TypeError: map1 is not a numpy array, neither a scalar
    #dst = cv2.remap(src=V, map1=Xq_32FC1, map2=Xq_32FC1, interpolation=cv2.INTER_LINEAR);

    # From http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html?highlight=remap#remap
    dst = cv.fromarray(V.copy());
    cv.Remap(src=cv.fromarray(V), dst=dst, \
                mapx=Xq_32FC1, mapy=Yq_32FC1, \
                flags=cv.CV_INTER_LINEAR); #CV_INTER_NN, CV_INTER_AREA, CV_INTER_CUBIC

    if False:
        dst = cv.fromarray(V.copy());

        """
        Gives error:
          OpenCV Error: Assertion failed (((map1.type() == CV_32FC2 || map1.type() == CV_16SC2) && !map2.data) || (map1.type() == CV_32FC1
          && map2.type() == CV_32FC1)) in unknown function, file ..\..\..\src\opencv\modules\imgproc\src\imgwarp.cpp, line 2988
        """

        # From http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html?highlight=remap#remap
        cv.Remap(src=cv.fromarray(V), dst=dst, \
                    mapx=cv.fromarray(Xq), mapy=cv.fromarray(Yq), \
                    flags=cv.CV_INTER_LINEAR); #flags=CV_INNER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))

    return dst;

###############################################################################
###############################################################################
###############################################################################
###############################################################################


"""
# From Matlab help (also http://www.mathworks.com/help/images/ref/imresize.html) :
  B = imresize(A, scale) returns image B that is scale times
  the size of A. The input image A can be a grayscale, RGB, or binary image. If scale is
  between 0 and 1.0, B is smaller than A.
  If scale is greater than 1.0, B is larger than A.

  B = imresize(A, [numrows numcols]) returns
  image B that has the number of rows and columns
  specified by [numrows numcols]. Either numrows or numcols may
  be NaN, in which case imresize computes
  the number of rows or columns automatically to preserve the image
  aspect ratio.

  Options:
    - 'nearest' Nearest-neighbor interpolation; the output pixel is assigned
      the value of the pixel that the point falls within. No other pixels
      are considered.

    - 'bicubic' Bicubic interpolation (the default); the output pixel
      value is a weighted average of pixels in the nearest 4-by-4 neighborhood


Example of standard imresize() in Matlab:
Trial>> img=ones(240, 320);
Trial>> img(92,192) = 0;
Trial>> res=imresize(img, [11, 11]) % this is default, bicubic interpolation
    res(3,7)=1.0001
    res(4,7)=0.9995
    res(5,7)=0.9987
    res(6,7)=1.0001
    res(5,8)=0.9999
    all other elems of res are 1.0000
What is surprising is that in this case, instead of let's say having
        res(4,7) = 0.99 and the rest 1, we have like above.
  This could indicate that because of antialiasing we have in the original image
    a 100x100 stretch to the neighboring 1s influenced due to the 0 at img(92,192).
  This is highly unlikely the case, therefore DOES imresize() scale down using
    "pyramids", i.e., a few intermediate scale-downs to reach the desired
    destination size?

Trial>> res=imresize(img, [11, 11], 'bilinear')
    res(4,7)=0.9996
    res(5,7)=0.9990
    res(5,8)=0.9999
    all other elems of res are 1.0000

Trial>> res=imresize(img, [11, 11], 'nearest')
    all elems of res are 1

"""
def imresize(A, scale=None, newSize=None, interpolationMethod=cv2.INTER_CUBIC): #!!!!TODO: should we put something different than
    assert (scale != None) or (newSize != None);
    assert not ((scale != None) and (newSize != None));

    common.DebugPrint("Entered imresize(A.shape=%s, scale=%s, newSize=%s)" % \
                            (str(A.shape), str(scale), str(newSize)));

    if scale != None:
        newSize = (int(A.shape[0] * scale), int(A.shape[1] * scale));

    # Matlab has [numrows numcols] and OpenCV has [numcols numrows] for size specification.
    newSize = (newSize[1], newSize[0]);

    """
    NOTE: the image returned by cv2.resize after INTER_CUBIC contains slightly
      different results compared to Matlab's imresize(, 'bicubic').

    # From http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#resize
        cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation ]]]]) -> dst

        interpolation - interpolation method:
        - INTER_NEAREST - a nearest-neighbor interpolation
        - INTER_LINEAR - a bilinear interpolation (used by default)
        - INTER_AREA - resampling using pixel area relation.
            It may be a preferred method for image decimation,
            as it gives moire'-free results. But when the image is zoomed,
            it is similar to the INTER_NEAREST method.
        - INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
        - INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
    """
    if True:
        """
        NOTE:
        From http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
            "Scaling is just resizing of the image.
            OpenCV comes with a function cv2.resize() for this purpose.
            The size of the image can be specified manually,
            or you can specify the scaling factor.
            Different interpolation methods are used.
            Preferable interpolation methods are cv2.INTER_AREA for shrinking and
            cv2.INTER_CUBIC (slow) & cv2.INTER_LINEAR for zooming.
            By default, interpolation method used is cv2.INTER_LINEAR for all
            resizing purposes."
        """
        if newSize[0] >= A.shape[0]:
            # We do zoom in of the image (make the image bigger)
            """
            common.DebugPrint("imresize(): issue - A.shape=%s, newSize=%s" % \
                                                    (str(A.shape), newSize));
            """
            #assert newSize[0] < A.shape[0];
            res = cv2.resize(A, newSize, interpolation=cv2.INTER_LINEAR);
        else:
            # We make the image smaller
            """
            This is a ~bit equivalent with Matlab's imresize w.r.t. antialiasing:
                - even for 1 pixel 0 surrounded only by 1s, when we reduce
                  the image, that 0 contributes a bit to the result of resize()
                  in only a pixel (not a square of pixels like in Matlab).
            See test_OpenCV.py for behavior of resize(..., INTER_AREA)
            """
            res = cv2.resize(A, newSize, interpolation=cv2.INTER_AREA);
    else:
        """
        NOTE: As said above INTER_CUBIC interp. method is slow and does not do
           anti-aliasing at all (while Matlab's imresize does antialiasing).
        """
        res = cv2.resize(A, newSize, interpolation=interpolationMethod);

    return res;


"""
From Matlab help (http://www.mathworks.com/help/matlab/ref/hist.html):
    "hist(data) creates a histogram bar plot of data.
    Elements in data are sorted into 10 equally spaced bins along the x-axis
        between the minimum and maximum values of data."

    "hist(data,xvalues) uses the values in vector xvalues to determine the bin
        intervals and sorts data into the number of bins
        determined by length(xvalues). To specify the bin
        centers, set xvalues equal to a vector of evenly
        spaced values. The first and last bins extend to cover the minimum
        and maximum values in data."

    "nelements = hist(___) returns a row vector, nelements, indicating the number of elements in each bin."
"""
"""
From https://stackoverflow.com/questions/18065951/why-does-numpy-histogram-python-leave-off-one-element-as-compared-to-hist-in-m:
  <<Note that in matlab's hist(x, vec), vec difines the bin-centers, while in
    matlab histc(x, vec) vec defines the bin-edges of the histogram.
    Numpy's histogram seems to work with bin-edges.
    Is this difference important to you?
    It should be easy to convert from one to the other, and you might have to
      add an extra Inf to the end of the bin-edges to get it to return the
      extra bin you want.
      More or less like this (untested):
      For sure it does not cover all the edge-cases that matlab's hist provides,
        but you get the idea.>>
"""
def hist(x, binCenters):
    #!!!!TODO: verify if it covers all the edge-cases that matlab's hist provides
    #print binCenters[:-1] + binCenters[1:]
    binEdges = np.r_[-np.Inf, 0.5 * (binCenters[:-1] + binCenters[1:]), np.Inf]
    counts, edges =  np.histogram(x, binEdges)

    return counts


def kron(A, B):
    """
    From Matlab help:
      kron
          Kronecker tensor product
      K = kron(A,B)
        example
      Description
        example
      K = kron(A,B)
        returns the Kronecker tensor product of matrices A and B.
      If A is an m-by-n matrix
      and B is a p-by-q matrix,
      then kron(A,B) is an m*p-by-n*q matrix
      formed by taking all possible products between the elements of A and
      the matrix B.

      %   KRON(X,Y) is the Kronecker tensor product of X and Y.
      %   The result is a large matrix formed by taking all possible
      %   products between the elements of X and those of Y. For
      %   example, if X is 2 by 3, then KRON(X,Y) is
      %
      %      [ X(1,1)*Y  X(1,2)*Y  X(1,3)*Y
      %        X(2,1)*Y  X(2,2)*Y  X(2,3)*Y ]
      %
      %   If either X or Y is sparse, only nonzero elements are multiplied
      %   in the computation, and the result is sparse.
    """

    # See the interesting Matlab implementation

    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.kron.html
    res = np.kron(A, B);

    """
    Note: it seems that np.kron might be failing to compute exactly what
      Matlab kron() computes.
        See https://stackoverflow.com/questions/17035767/kronecker-product-in-python-and-matlab.
    Following https://stackoverflow.com/questions/17035767/kronecker-product-in-python-and-matlab
        we can get the right result.??
    # See http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.kron.html
        res = scipy.sparse.kron(A, B);
    """

    return res;


"""
Matlab unique from http://www.mathworks.com/help/matlab/ref/unique.html:
C = unique(A) returns the same data as in A, but with no repetitions.
    If A is a numeric array, logical array, character array, categorical array, or a cell array of strings, then unique returns the unique values in A. The values of C are in sorted order.
    If A is a table, then unique returns the unique rows in A. The rows of table C are in sorted order.
example
C = unique(A,'rows') treats each row of A as a single entity and returns the unique rows of A. The rows of the array C are in sorted order.
The 'rows' option does not support cell arrays.
[C,ia,ic] = unique(A) also returns index vectors ia and ic.
    If A is a numeric array, logical array, character array, categorical array, or a cell array of strings, then C = A(ia) and A = C(ic).
    If A is a table, then C = A(ia,:) and A = C(ic,:).
[C,ia,ic] = unique(A,'rows') also returns index vectors ia and ic, such that C = A(ia,:) and A = C(ic,:).
"""
def unique(c2):
    # NOT good since it flattens the array (transforms multidim vector into 1D vector)
    #c2, c2i = np.unique(ar=c2, return_index=True) #Check if "rows", "first" is really required

    # c2 is a list of lists of 2 float elements. Ex: c2F = [[89.188, 33.111], [90.994, 250.105], ...]
    #common.DebugPrint("c2 = %s" % str(c2));

    c2l = c2.tolist();
    #common.DebugPrint("unique(): c2l=%s" % str(c2l));

    c2lSorted = [];
    for i, e in enumerate(c2l):
        #e1 = e; #.copy();
        e.append(i);
        c2lSorted.append(e);
    c2lSorted.sort();
    #common.DebugPrint("unique(): c2lSorted = %s" % str(c2lSorted));

    c2lSortedIndex = [];
    for e in c2lSorted:
        c2lSortedIndex.append(e[2]);
    #common.DebugPrint("unique(): c2lSortedIndex = %s" % str(c2lSortedIndex));

    #quit();

    """
    np.argsort() is a crappy function in the end...
    Unfortunately I don't understand the output of np.argsort() when a is
         bi/multi-dimensional... - and it's not my fault - see
         http://stackoverflow.com/questions/12496531/sort-numpy-float-array-column-by-column
         and http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    """
    if False:
        #c2i = []
        # Does NOT return c2i: c2 = np.sort(a=c2, axis=0, kind="quicksort")
        c2i = np.argsort(a=c2, axis=0, kind="quicksort");
        assert len(c2i) == len(c2);

	if common.MY_DEBUG_STDOUT:
            common.DebugPrint("unique(): c2i = %s" % str(c2i));
        c2i = c2i[:, 0]; # c2i returned is a list of lists of 2 elements: c2i[k][1] is the index of the kth element in c2i

    c2i = np.array(c2lSortedIndex);
    #common.DebugPrint("unique(): c2i = %s" % str(c2i));

    # This is not good since the last element is the position: c2 = np.array(c2lSorted);
    try:
        c2 = c2[c2i];
    except:
        #common.DebugPrint("c2 = %s" % str(c2))
        #common.DebugPrint("c2i = %s" % str(c2i))
        return np.array([]), np.array([]);
       #quit()

    #common.DebugPrint("unique(): c2 = %s" % str(c2));
    #common.DebugPrint("c2 (after indirect sorting) = %s" % str(c2));

    c2F = []; #np.array([])
    c2iF = [];

    #rowFirst = 0;
    #for row in range(c2i.size - 1):
    row = 0;
    #for row in range(c2i.size - 1):
    while row < c2i.size:
        #print "row = %d" % row;

        c2F.append(c2[row].tolist());
        c2iF.append(c2i[row]);
        #common.DebugPrint("row = %d" % row);

        if row + 1 < c2i.size:
            rowCrt = row + 1;
            while rowCrt < c2i.size:
                #common.DebugPrint("rowCrt = %d" % rowCrt);

                """
                # This comparison is NOT general - it assumes
                #            np.dims(c2) == (X, 2).
                if (c2[row][0] == c2[rowCrt][0]) and \
                                (c2[row][1] == c2[rowCrt][1]):
                """
                """
                 Test that the 2 rows are identical
                    (each pair of corresponding elements are equal)
                """
                if (c2[row] == c2[rowCrt]).all():
                    pass;
                else:
                    break;
                rowCrt += 1;
            row = rowCrt;
        else:
            row += 1;

    """
    #for row in range(c2i.size - 1):
    while row < c2i.size:
        #print "row = %d" % row
        if row == c2i.size - 1:
            c2F.append(c2[row].tolist());
            c2iF.append(c2i[row]);

        for rowCrt in range(row + 1, c2i.size):
            #print "rowCrt = %d" % rowCrt
            if (c2[row][0] == c2[rowCrt][0]) and \
                            (c2[row][1] == c2[rowCrt][1]):
                row += 1;
                pass;
            else:
                c2F.append(c2[row].tolist());
                c2iF.append(c2i[row]);
                #rowFirst += 1;
                break
        row += 1;
    """

    #print "unique(): c2F = %s" % (str(c2F));
    #print "unique(): c2iF = %s" % (str(c2iF));

    c2F = np.array(c2F);
    c2iF = np.array(c2iF);

    return c2F, c2iF;




def hamming(N):
    """
    Alex: replaced Matlab "Signal Processing Toolbox"'s hamming() with my own
        simple definition - inspired from
        http://www.mathworks.com/matlabcentral/newsreader/view_thread/102510
    """

    t = np.array( range(N) )
    b = np.zeros( (N) )

    if N == 0:
        return b;
    elif N == 1:
        b[0] = 1;
        return b;

    #print "hamming(): b.shape = %s" % str(b.shape);
    #print "hamming(): t.shape = %s" % str(t.shape);

    #b[t] = 0.54 - 0.46 * math.cos(2 * math.pi * (t - 1) / (N - 1));
    b[t] = 0.54 - 0.46 * np.cos(2 * math.pi * t / float(N - 1));

    return b;


"""
h is a vector... - see use below filter2(b, ...)

From Matlab help:
  Syntax
  Y = filter2(h,X)
  Y = filter2(h,X,shape)
  Description
  Y = filter2(h,X) filters
  the data in X with the two-dimensional FIR filter
  in the matrix h. It computes the result, Y,
  using two-dimensional correlation, and returns the central part of
  the correlation that is the same size as X.Y = filter2(h,X,shape) returns
  the part of Y specified by the shape parameter. shape is
  a string with one of these values:

  Given a matrix X and a two-dimensional FIR
  filter h, filter2 rotates
  your filter matrix 180 degrees to create a convolution kernel. It
  then calls conv2, the two-dimensional convolution
  function, to implement the filtering operation.filter2 uses conv2 to
  compute the full two-dimensional convolution of the FIR filter with
  the input matrix. By default, filter2 then extracts
  the central part of the convolution that is the same size as the input
  matrix, and returns this as the result. If the shape parameter
  specifies an alternate part of the convolution for the result, filter2 returns
  the appropriate part.
"""
# https://stackoverflow.com/questions/16278938/convert-matlab-to-opencv-in-python
def filter2(window, src):
    assert len(window.shape) == 1

    # In certain cases it's unusual that we have a window that is 1D and x is 2D
    #common.DebugPrint("filter2(): src.shape = %s" % str(src.shape))
    #common.DebugPrint("filter2(): window.shape = %s" % str(window.shape))

    #common.DebugPrint("filter2(): src = %s" % str(src))
    #common.DebugPrint("filter2(): window = %s" % str(window))

    # From http://docs.opencv.org/modules/imgproc/doc/filtering.html#cv.Filter2D
    #cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst
    #res = cv2.filter2D(src=window, ddepth=-1, kernel=x)
    #res = cv2.filter2D(src=x, ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT) #cv2.BORDER_CONSTANT) #cv2.BORDER_ISOLATED) #cv2.BORDER_TRANSPARENT)

    """
    # OpenCV's formula to compute filter2 is dst(x,y)= \sum...
        As we can see they consider the matrix being in column-major order, so we need to transpose the matrices
        Doing so we obtain the right values at the 4 borders.
    """
    src = src.T;
    #window = window.T

    # Note: CvPoint is (x, y)
    res = cv2.filter2D(src=src, ddepth=-1, kernel=window, anchor=(-1, -1), \
                        borderType=cv2.BORDER_ISOLATED); #BORDER_DEFAULT) #BORDER_CONSTANT) #cv2.BORDER_TRANSPARENT)

    res = res.T;


    if False:
        # Alex's implementation following http://docs.opencv.org/modules/imgproc/doc/filtering.html#filter2d
        dst = np.zeros( (src.shape[0], src.shape[1]), dtype=np.float64);
        for y in range(src.shape[1]):
            for x in range(src.shape[0]):
                for xp in range(window.shape[0]):
                    """
                    if (y >= src.shape[0]) or (x + xp - 1 >= src.shape[1]) or \
                                                (x + xp - 1 < 0):
                    """
                    if (y >= src.shape[1]) or (x + xp - 1 >= src.shape[0]) or \
                                                (x + xp - 1 < 0):
                        pass;
                    else:
                        #dst[y, x] += window[xp] * src[y, x + xp - 1];
                        dst[x, y] += window[xp] * src[x + xp - 1, y];
        res = dst;
        return res;

    """
    """
    if False:
        range1 = src.shape[0] - window.shape[0] + 1;

        #range2 = src.shape[1] - window.shape[1] + 1;
        range2 = src.shape[1] - window.shape[0] + 1;

        if range2 < 0:
            range2 = 1;

        res = np.zeros((range1, range2), dtype=np.float64);

        for i in range(range1):
            for j in range(range2):
                #common.DebugPrint("filter2(): j = %d" % j)
                #res[i, j] = np.sum(np.multiply(x[i: 11 + i, j: 11 + j], window))
                res[i, j] = np.sum( \
                        np.multiply(src[i: window.shape[0] + i, j: window.shape[0] + j], window));

    # From https://codereview.stackexchange.com/questions/31089/optimizing-numpy-code - optimized version using as_strided and sum instead of nested loops
    if False:
        # Gives exception: "ValueError: negative dimensions are not allowed"
        x1 = np.lib.stride_tricks.as_strided(x, \
                    ((src.shape[0] - 10) / 1, (src.shape[1] - 10) / 1, 11, 11), \
                    (src.strides[0] * 1, src.strides[1] * 1, \
                    src.strides[0], src.strides[1])) * window;
        res = x1.sum((2, 3));

    return res;



def gradient(img, spaceX=1, spaceY=1, spaceZ=1):
    assert (img.ndim == 2) or (img.ndim == 3);
    #if img.ndim == 3:
    #    assert spaceZ != None

    """
    From Matlab help:
      Description:
        FX = gradient(F), where F is
        a vector, returns the one-dimensional numerical gradient of F.
        Here FX corresponds to deltaF/deltax,
        the differences in x (horizontal) direction. [FX,FY] = gradient(F),
        where F is a matrix, returns the x and y components
        of the two-dimensional numerical gradient. FX corresponds
        to deltaF/deltax, the
        differences in x (horizontal) direction. FY corresponds
        to  deltaF/deltay, the
        differences in the y (vertical) direction. The
        spacing between points in each direction is assumed to be one.

        [FX,FY,FZ,...] = gradient(F),
        where F has N dimensions, returns
        the N components of the gradient of F.
        There are two ways to control the spacing between values in F:
            A single spacing value, h, specifies
        the spacing between points in every direction.
        N spacing values (h1,h2,...)
        specifies the spacing for each dimension of F.
        Scalar spacing parameters specify a constant spacing for each dimension.
        Vector parameters specify the coordinates of the values along corresponding
        dimensions of F. In this case, the length of the
        vector must match the size of the corresponding dimension.

      Note:
        The first output FX is always the gradient
        along the 2nd dimension of F, going across columns.
         The second output FY is always the gradient along
        the 1st dimension of F, going across rows.  For
        the third output FZ and the outputs that follow,
        the Nth output is the gradient along the Nth dimension of F.

        [...] = gradient(F,h1,h2,...) with N spacing
        parameters specifies the spacing for each dimension of F.
    """

    """
    The best documentation is at:
        http://answers.opencv.org/question/16422/matlab-gradient-to-c/
      Note: There they mention also

    #!!!!See also
        http://stackoverflow.com/questions/17977936/matlab-gradient-equivalent-in-opencv
        http://stackoverflow.com/questions/9964340/how-to-extract-numerical-gradient-matrix-from-the-image-matrix-in-opencv
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    """
    #print "img = %s" % str(img);

    """
    !!!!TODO: try to implement a ~different gradient using OpenCV's Sobel,
      or other parallelizable function - I think gradient() will be a hotspot
      of the application.
    """
    if False:
        # From http://answers.opencv.org/question/16422/matlab-gradient-to-c/
        """
        Result is VERY wrong compared to Matlab's x,y = gradient.
        """
        #sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=5);
        #sobelx = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0, ksize=5);
        #sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5);
        sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5);
        #sobely = cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=5);
        #sobely = cv2.Sobel(src=img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5);
        sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5);
        #sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5);

        print "sobelx = %s" % str(sobelx);
        print "sobely = %s" % str(sobely);

    if False:
        # Following example from http://answers.opencv.org/question/16422/matlab-gradient-to-c/ :
        grad_x = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0);

        #cv2.convertScaleAbs(src=grad_x, dst=abs_grad_x) #, alpha[, beta ]]])
        abs_grad_x = cv2.convertScaleAbs(src=grad_x); #, alpha[, beta ]]])

        grad_y = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1);

        abs_grad_y = cv2.convertScaleAbs(src=grad_y); #, alpha[, beta ]]])

        #cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype ]])
        grad = cv2.addWeighted(src1=abs_grad_x, alpha=0.5, src2=abs_grad_y, beta=0.5, gamma=0.0);

        print "grad = %s" % str(grad);



    # Inspired heavily from the C++ code available at
    #   http://answers.opencv.org/question/16422/matlab-gradient-to-c/:

    #/// Internal method to get numerical gradient for x components.
    #/// @param[in] mat Specify input matrix.
    #/// @param[in] spacing Specify input space.
    def gradientX(mat, spacing):
        grad = np.zeros( (mat.shape[0], mat.shape[1]), dtype=mat.dtype );

        #const int maxCols = mat.cols;
        #const int maxRows = mat.rows;
        maxRows, maxCols = mat.shape;

        #/* get gradients in each border */
        #/* first col */
        #Mat col = (-mat.col(0) + mat.col(1))/(float)spacing;
        col = (-mat[:, 0] + mat[:, 1]) / float(spacing);

        # Rect() is explained for example at http://docs.opencv.org/modules/core/doc/drawing_functions.html - basically we denote a rectange: col_ul, row_ul and then delta_col, delta_row
        #col.copyTo(grad(Rect(0,0,1,maxRows)));
        #grad[: maxRows, : 1] = col; # Gives exception: "ValueError: output operand requires a reduction, but reduction is not enabled"
        grad[0: maxRows, 0: 1] = col.reshape( (col.size, 1) ); #.copy();

        #/* last col */
        #col = (-mat.col(maxCols-2) + mat.col(maxCols-1))/(float)spacing;
        col = (-mat[:, maxCols - 2] + mat[:, maxCols - 1]) / float(spacing);

        #col.copyTo(grad(Rect(maxCols-1,0,1,maxRows)));
        #grad[0: maxRows, maxCols - 1: maxCols] = col; # Gives exception: "ValueError: output operand requires a reduction, but reduction is not enabled"
        grad[0: maxRows, maxCols - 1: maxCols] = col.reshape( (col.size, 1) ); #.copy();

        #/* centered elements */
        #Mat centeredMat = mat(Rect(0,0,maxCols-2,maxRows));
        centeredMat = mat[0: maxRows, 0: maxCols - 2];

        #Mat offsetMat = mat(Rect(2,0,maxCols-2,maxRows));
        offsetMat = mat[0:maxRows, 2: maxCols];

        #Mat resultCenteredMat = (-centeredMat + offsetMat)/(((float)spacing)*2.0);
        resultCenteredMat = (-centeredMat + offsetMat) / (float(spacing) * 2.0);

        #resultCenteredMat.copyTo(grad(Rect(1,0,maxCols-2, maxRows)));
        grad[0: maxRows, 1: maxCols - 1] = resultCenteredMat; #.copy();

        return grad;


    #/// Internal method to get numerical gradient for y components.
    #/// @param[in] mat Specify input matrix.
    #/// @param[in] spacing Specify input space.
    def gradientY(mat, spacing):
        #Mat grad = Mat::zeros(mat.cols,mat.rows,CV_32F);
        grad = np.zeros( (mat.shape[0], mat.shape[1]), dtype=mat.dtype );

        #const int maxCols = mat.cols;
        #const int maxRows = mat.rows;
        maxRows, maxCols = mat.shape;

        #/* get gradients in each border */
        #/* first row */
        #Mat row = (-mat.row(0) + mat.row(1))/(float)spacing;
        row = (-mat[0, :] + mat[1, :]) / float(spacing);

        #row.copyTo(grad(Rect(0,0,maxCols,1)));
        #grad[: maxCols, : 1] = row; # Gives exception: "ValueError: output operand requires a reduction, but reduction is not enabled"
        grad[0: 1, 0: maxCols] = row.reshape( (1, row.size) ); #.copy();

        #if False:
	if common.MY_DEBUG_STDOUT:
            common.DebugPrint("gradientY(): spacing = %s" % str(spacing));
            common.DebugPrint("grad[0: 1, 0: maxCols].shape = %s" % str(grad[0: 1, 0: maxCols].shape));
            common.DebugPrint("row.shape = %s" % str(row.shape));
            common.DebugPrint("row = %s" % str(row));

        #/* last row */
        #row = (-mat.row(maxRows-2) + mat.row(maxRows-1))/(float)spacing;
        row = (-mat[maxRows - 2, :] + mat[maxRows - 1, :]) / float(spacing);

        #row.copyTo(grad(Rect(0,maxRows-1,maxCols,1)));
        grad[maxRows - 1: maxRows, 0: maxCols] = row; #.copy();

        #/* centered elements */
        #Mat centeredMat = mat(Rect(0,0,maxCols,maxRows-2));
        centeredMat = mat[0: maxRows - 2, 0: maxCols];

        #Mat offsetMat = mat(Rect(0,2,maxCols,maxRows-2));
        offsetMat = mat[2: maxRows, 0: maxCols];

        #Mat resultCenteredMat = (-centeredMat + offsetMat)/(((float)spacing)*2.0);
        resultCenteredMat = (-centeredMat + offsetMat) / (float(spacing) * 2.0);

        #resultCenteredMat.copyTo(grad(Rect(0,1,maxCols, maxRows-2)));
        grad[1: maxRows - 1, 0: maxCols] = resultCenteredMat; #.copy();

        return grad;


    def gradientZ(mat, indexZ, spacing, grad):
        """
        Since we know we have a 3D array we work on planes (2d subelements),
            not on 1D elements (rows and columns).
        """
	if common.MY_DEBUG_STDOUT:
            common.DebugPrint("Entered gradientZ(indexZ=%d)" % indexZ);

        maxRows, maxCols, maxZs = mat.shape;

        # get gradients in each border
        # first Z plane
        plane = (-mat[:, :, 0] + mat[:, :, 1]) / float(spacing);

        #grad[0: maxRows, 0: maxCols, 0] = plane; #.reshape( (maxRows, maxCols) ); #plane.reshape( (mat.shape[0], mat.shape[1]) );
        grad[:, :, 0] = plane;

        #if False:
	if common.MY_DEBUG_STDOUT:
            common.DebugPrint("gradientY(): spacing = %s" % str(spacing));
            common.DebugPrint("grad[0: 1, 0: maxCols].shape = %s" % str(grad[0: 1, 0: maxCols].shape));
            common.DebugPrint("row.shape = %s" % str(row.shape));
            common.DebugPrint("row = %s" % str(row));

        # last Z plane
        plane = (-mat[:, :, maxZs - 2] + mat[:, :, maxZs - 1]) / float(spacing);

        #grad[0: maxRows, 0: maxCols, maxZs - 1] = plane;
        grad[:, :, maxZs - 1] = plane;

        # centered elements
        centeredMat = mat[:, :, 0: maxZs - 2];

        offsetMat = mat[:, :, 2: maxZs];

        resultCenteredMat = (-centeredMat + offsetMat) / (float(spacing) * 2.0);

        #grad[1: maxRows - 1, 0: maxCols] = resultCenteredMat;
        grad[:, :, 1: maxZs - 1] = resultCenteredMat;

        return grad;


    if img.ndim == 3:
        # The shape of the
        gradX = np.zeros( (img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype );
        gradY = np.zeros( (img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype );
        gradZ = np.zeros( (img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype );

        for i in range(img.shape[2]):
            gradX1 = gradientX(img[:, :, i], spaceX);
            gradX[:, :, i] = gradX1;

            gradY1 = gradientY(img[:, :, i], spaceY);
            gradY[:, :, i] = gradY1;

        gradientZ(img, i, spaceZ, gradZ);

        return (gradX, gradY, gradZ);


    #Mat gradX = gradientX(img,spaceX);
    gradX = gradientX(img, spaceX);
    #common.DebugPrint("gradient(): returned from gradientX");

    #Mat gradY = gradientY(img,spaceY);
    gradY = gradientY(img, spaceY);

    #pair<Mat,Mat> retValue(gradX,gradY);
    #return retValue;
    return (gradX, gradY);



def meshgrid(range1, range2):
    """
    From http://www.mathworks.com/help/matlab/ref/meshgrid.html:
    [X,Y] = meshgrid(xgv,ygv) replicates the grid vectors xgv and ygv
        to produce a full grid.
    This grid is represented by the output coordinate arrays X and Y.
    The output coordinate arrays X and Y contain copies of the grid
        vectors xgv and ygv respectively.
        The sizes of the output arrays are determined by the length of
            the grid vectors. For grid vectors xgv and ygv of length M and N
            respectively, X and Y will have N rows and M columns.

        Examples
        2-D Grid From Vectors

        Create a full grid from two monotonically increasing grid vectors:

        [X,Y] = meshgrid(1:3,10:14)
        X =
            1     2     3
            1     2     3
            1     2     3
            1     2     3
            1     2     3
        Y =
            10    10    10
            11    11    11
            12    12    12
            13    13    13
            14    14    14
    """
    #y, x = np.mgrid[range2, range1];
    #x, y = np.meshgrid(range(1, 4), range(10, 15));

    #assert len(range1) == len(range2);

    x, y = np.meshgrid(range1, range2); #NOT working: copy=False);

    return x, y;


def fspecial(type, p2, p3):
    # See
    # http://stackoverflow.com/questions/16278938/convert-matlab-to-opencv-in-python
    # and http://blog.csdn.net/sunxin7557701/article/details/17163263
    # and http://stackoverflow.com/questions/23471083/create-2d-log-kernel-in-opencv-like-fspecial-in-matlab

    """
    % Alex: the code is taken from the Matlab, (type fspecial)
    siz   = (p2-1)/2;
    std   = p3;

    % Alex: adapted this, since siz is scalar in our case
    %[x,y] = meshgrid(-siz(2):siz(2),-siz(1):siz(1));
    [x,y] = meshgrid(-siz:siz,-siz:siz);
    arg   = -(x.*x + y.*y)/(2*std*std);

    h     = exp(arg);
    h(h<eps*max(h(:))) = 0;
    sumh = sum(h(:));

    if sumh ~= 0,
       h  = h/sumh;
    end;
    """

    assert (type == "gaussian") or (type == "ga");
    siz   = int((p2 - 1) / 2);
    std   = p3;

    #% Alex: adapted this, since siz is scalar in our case
    x, y = meshgrid(range(-siz, siz + 1), range(-siz, siz + 1));
    arg  = -(x * x + y * y) / (2.0 * std * std);

    h     = np.exp(arg);
    h[h < eps() * h.max()] = 0;

    sumh = h.sum();

    #if sumh != 0:
    if abs(sumh) > 1.e-6:
       h  = h / sumh;

    return h;


def imfilter(A, H, options=None):
    # We could raise an exception if somebody uses this function:
    #assert False;
    #return A;

    """
    From http://nf.nci.org.au/facilities/software/Matlab/toolbox/images/imfilter.html

    imfilter
    Multidimensional image filtering

    Syntax

        B = imfilter(A,H)
        B = imfilter(A,H,option1,option2,...)

    Description
    B = imfilter(A,H) filters the multidimensional array A with the multidimensional filter H.
      The array, A, can be a nonsparse numeric array of any class and dimension.
      The result, B, has the same size and class as A.

    Each element of the output, B, is computed using double-precision floating point.
      If A is an integer array, then output elements that exceed the range of
        the integer type are truncated, and fractional values are rounded.
    B = imfilter(A,H,option1,option2,...) performs multidimensional filtering
        according to the specified options. Option arguments can have the
        following values.

    Boundary Options Option
        Description
    X
        Input array values outside the bounds of the array are implicitly
          assumed to have the value X. When no boundary option is specified,
            imfilter uses X = 0.

    'symmetric'
        Input array values outside the bounds of the array are computed by
            mirror-reflecting the array across the array border.
    'replicate'
        Input array values outside the bounds of the array are assumed to
            equal the nearest array border value.
    'circular'
        Input array values outside the bounds of the array are computed by
            implicitly assuming the input array is periodic.


    Output Size Options Option
        Description
    'same'
        The output array is the same size as the input array.
        This is the default behavior when no output size options are specified.
    'full'
        The output array is the full filtered result, and so is larger than the input array.
    """

    # From http://answers.opencv.org/question/8783/implement-imfiltermatlab-with-opencv/
    """
    Point anchor( 0 ,0 );
    double delta = 0;

    float data[2][5] = {{11,11,11,11,11},{11,11,11,11,11}};
    float kernel[2][2] = {{2,2},{2,2}};

    Mat src = Mat(2, 5, CV_32FC1, &data);
    Mat ker = Mat(2, 2, CV_32FC1, &kernel);
    Mat dst = Mat(src.size(), src.type());

    Ptr<FilterEngine> fe =  createLinearFilter(src.type(), ker.type(), ker, anchor,
            delta, BORDER_CONSTANT, BORDER_CONSTANT, Scalar(0));
    fe->apply(src, dst);
    cout << dst << endl;
    """

    """
    See
      https://stackoverflow.com/questions/18628373/imfilter-equivalent-in-opencv
      http://answers.opencv.org/question/20785/replicate-with-mask-in-opencv/
    <<The kernel is the "mask" that i want to move in the convolution and the "anchor" is the middle of this matrix. CvType.CV_32FC1 means that the values inside the result matrix could be also negative, and "BORDER_REPLICATE" just fill the edge with zeroes.>>
    (not relevant: https://stackoverflow.com/questions/18628373/imfilter-equivalent-in-opencv )
    """
    """
    # The Matlab code we try to emulate is:
    bx = imfilter(gray_image,mask, 'replicate');
    """

    """
    Point anchor(0, 0);
    float delta = 0.0;

    cv::filter2D(gray_img, bx, CV_32FC1, mask, anchor, delta, BORDER_REPLICATE);
    """

    # Note: CvPoint is (x, y)
    """
    From http://docs.opencv.org/modules/imgproc/doc/filtering.html#filter2d :
      Convolves an image with the kernel.
      Parameters:
        kernel - convolution kernel (or rather a correlation kernel),
                a single-channel floating point matrix;
                if you want to apply different kernels to different channels,
                split the image into separate color planes using split() and
                process them individually.
        anchor - anchor of the kernel that indicates the relative position of
                a filtered point within the kernel;
                the anchor should lie within the kernel;
                default value (-1,-1) means that the anchor is at the kernel
                center.
        delta - optional value added to the filtered pixels before storing
                them in dst.
        borderType - pixel extrapolation method
                            (see borderInterpolate() for details).
    """
    res = cv2.filter2D(src=A, ddepth=-1, kernel=H, anchor=(-1, -1), \
                        borderType=cv2.BORDER_REPLICATE); #BORDER_ISOLATED); #BORDER_DEFAULT) #BORDER_CONSTANT) #cv2.BORDER_TRANSPARENT)

    return res;


def bwlabel_and_find(BW):
    # NOTE: this is a similar, but not identical implementation
    assert False; #!!!!TODO: not implemented

    """
    From http://www.mathworks.com/help/images/ref/bwlabel.html
        Label connected components in 2-D binary image
        L = bwlabel(BW, n)
        [L, num] = bwlabel(BW, n)
        Description

        L = bwlabel(BW, n) returns a matrix L, of the same size as BW, containing labels for the connected objects in BW. The variable n can have a value of either 4 or 8, where 4 specifies 4-connected objects and 8 specifies 8-connected objects. If the argument is omitted, it defaults to 8.

        The elements of L are integer values greater than or equal to 0. The pixels labeled 0 are the background. The pixels labeled 1 make up one object; the pixels labeled 2 make up a second object; and so on.

        [L, num] = bwlabel(BW, n) returns in num the number of connected objects found in BW.

        The functions bwlabel, bwlabeln, and bwconncomp all compute connected components for binary images. bwconncomp replaces the use of bwlabel and bwlabeln. It uses significantly less memory and is sometimes faster than the other functions.

    Create a small binary image to use for this example.

    BW = logical ([1     1     1     0     0     0     0     0
                1     1     1     0     1     1     0     0
                1     1     1     0     1     1     0     0
                1     1     1     0     0     0     1     0
                1     1     1     0     0     0     1     0
                1     1     1     0     0     0     1     0
                1     1     1     0     0     1     1     0
                1     1     1     0     0     0     0     0]);

    Create the label matrix using 4-connected objects.

    L = bwlabel(BW,4)

    L =

        1     1     1     0     0     0     0     0
        1     1     1     0     2     2     0     0
        1     1     1     0     2     2     0     0
        1     1     1     0     0     0     3     0
        1     1     1     0     0     0     3     0
        1     1     1     0     0     0     3     0
        1     1     1     0     0     3     3     0
        1     1     1     0     0     0     0     0

    Use the find command to get the row and column coordinates of the object labeled "2".

    [r, c] = find(L==2);
    rc = [r c]

    rc =

        2     5
        3     5
        2     6
        3     6
    """

    """
    From http://stackoverflow.com/questions/20357337/opencv-alternative-for-matlabs-bwlabel
    This isn't exactly the same as bwlabel, but may be close enough.
    One possible alternative is to use findContours and/or drawContours, as explained in the docs.
    """

    # From http://stackoverflow.com/questions/12688524/connected-components-in-opencv
    pass;



class TestSuite(unittest.TestCase):
    def testUnique1(self):
        l = [[1, 2], [1, 3], [1, 2]];
        l = np.array(l);

        lF, liF = unique(l);
        #common.DebugPrint("lF = %s" % str(lF));
        #common.DebugPrint("liF = %s" % str(liF));

        res = np.array([[1, 2], [1, 3]]);
        aZero = res - lF;

        resi = np.array([0, 1]);
        aZeroi = resi - liF;

        # State our expectation
        #self.assertTrue(lF == res);
        self.assertTrue( (aZero == 0).all() and \
                        (aZeroi == 0).all())

    def testUnique2(self):
        l = [[1, 2], [1, 2]];
        l = np.array(l);

        lF, liF = unique(l);
        #common.DebugPrint("lF = %s" % str(lF));
        #common.DebugPrint("liF = %s" % str(liF));

        res = np.array([[1, 2]]);
        aZero = res - lF;

        resi = np.array([0]);
        aZeroi = resi - liF;

        # State our expectation
        #self.assertTrue(lF == res);
        self.assertTrue( (aZero == 0).all() and \
                        (aZeroi == 0).all())

    def testUnique3(self):
        l = [[1, 3], [1, 2], [0, 7], [5, 9], [1, 2], [100, 0], [5, 10]];
        l = np.array(l);

        lF, liF = unique(l);
        #common.DebugPrint("lF = %s" % str(lF));
        #common.DebugPrint("liF = %s" % str(liF));

        res = np.array([[[0, 7], [1, 2], [1, 3], [5, 9], [5, 10], [100, 0]]]);
        aZero = res - lF;

        resi = np.array([2, 1, 0, 3, 6, 5]);
        aZeroi = resi - liF;

        # State our expectation
        #self.assertTrue(lF == res);
        self.assertTrue( (aZero == 0).all() and \
                        (aZeroi == 0).all())
    #################################################


    def testHamming(self):
        # Test 1
        #print "hamming(11) = %s" % str(hamming(11))
        h11 = hamming(11);
        #h11 = [ 0.08        0.16785218  0.39785218  0.68214782  0.91214782  1.
        #       0.91214782  0.68214782  0.39785218  0.16785218  0.08      ]
        h11Good = np.array([0.0800, 0.1679, 0.3979, 0.6821, 0.9121, 1.0000, \
                            0.9121, 0.6821, 0.3979, 0.1679, 0.0800]);
        aZero = h11 - h11Good;
        #print "h11 = %s" % str(h11);
        #print "aZero = %s" % str(aZero);
        #assert (aZero < 1.0e-3).all();
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());

        h0 = hamming(0);
        #print "h0 = %s" % str(h0);
        self.assertTrue(h0.size == 0);

        h1 = hamming(1);
        #print "h1 = %s" % str(h1);
        h1Good = np.array([1.0]);
        aZero = h1 - h1Good;
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());

        h2 = hamming(2);
        #print "h2 = %s" % str(h2);
        h2Good = np.array([0.08, 0.08]);
        aZero = h2 - h2Good;
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());

        h100 = hamming(100);
        #print "h100 = %s" % str(h100);
        h100Good = np.array([ \
            0.0800, 0.0809, 0.0837, 0.0883, 0.0947, 0.1030, 0.1130, 0.1247, \
            0.1380, 0.1530, 0.1696, 0.1876, 0.2071, 0.2279, 0.2499, 0.2732, \
            0.2975, 0.3228, 0.3489, 0.3758, 0.4034, 0.4316, 0.4601, 0.4890, \
            0.5181, 0.5473, 0.5765, 0.6055, 0.6342, 0.6626, 0.6905, 0.7177, \
            0.7443, 0.7700, 0.7948, 0.8186, 0.8412, 0.8627, 0.8828, 0.9016, \
            0.9189, 0.9347, 0.9489, 0.9614, 0.9723, 0.9814, 0.9887, 0.9942, \
            0.9979, 0.9998, 0.9998, 0.9979, 0.9942, 0.9887, 0.9814, 0.9723, \
            0.9614, 0.9489, 0.9347, 0.9189, 0.9016, 0.8828, 0.8627, 0.8412, \
            0.8186, 0.7948, 0.7700, 0.7443, 0.7177, 0.6905, 0.6626, 0.6342, \
            0.6055, 0.5765, 0.5473, 0.5181, 0.4890, 0.4601, 0.4316, 0.4034, \
            0.3758, 0.3489, 0.3228, 0.2975, 0.2732, 0.2499, 0.2279, 0.2071, \
            0.1876, 0.1696, 0.1530, 0.1380, 0.1247, 0.1130, 0.1030, 0.0947, \
            0.0883, 0.0837, 0.0809, 0.0800]);
        aZero = h100 - h100Good;
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());

    def testHammingOld(self):
        # Test 1
        #print "hamming(11) = %s" % str(hamming(11))
        h11 = hamming(11);
        #h11 = [ 0.08        0.16785218  0.39785218  0.68214782  0.91214782  1.
        #       0.91214782  0.68214782  0.39785218  0.16785218  0.08      ]
        h11Good = np.array([0.0800, 0.1679, 0.3979, 0.6821, 0.9121, 1.0000, \
                            0.9121, 0.6821, 0.3979, 0.1679, 0.0800]);
        aZero = h11 - h11Good;
        #print "h11 = %s" % str(h11);
        #print "aZero = %s" % str(aZero);
        #assert (aZero < 1.0e-3).all();

        self.assertTrue((np.abs(aZero) < 1.0e-3).all());


    def testFilter2(self):
        #window = hamming(11)
        window = np.array( [1, 2, 4] );
        x = np.ones( (23, 8) );
        res = filter2(window, x);
        #print "filter2(window, x) = %s" % str(res);

        resGood = [[6,  7,  7,  7,  7,  7,  7,  3]] * 23;
        resGood = np.array(resGood);
        aZero = res - resGood;
        #assert (aZero == 0).all();

        self.assertTrue((aZero == 0).all());


    def testHist(self):
        print "Entered testHist()."
        howMany = [56, 38, 54, 28, 32, 40, 38, 32, 54, 62, 46, 42]

        RD_start = 2001
        RD_end = 2012

        l = []
        #for i in range(RD_start, RD_end + 1):
        for i in range(RD_end - RD_start + 1):
            l += [RD_start + i] * howMany[i]

        #common.DebugPrint("l = %s" % str(l))

        assert len(l) == 522
        l = np.array(l)

        rr = np.array(range(RD_start, RD_end + 1))
        n_d = [56, 38, 54, 28, 32, 40, 38, 32, 54, 62, 46, 42]

        # Test 1
        #print "hamming(11) = %s" % str(hamming(11))
        res = hist(l, rr);

        aZero = res - n_d
        #print "h11 = %s" % str(h11)
        #print "aZero = %s" % str(aZero)
        if False:
            assert (aZero == 0).all()

        # State our expectation
        self.assertTrue((aZero == 0).all())


    def testGradient(self):
        # Testing bidimensional gradient:

        """
        From http://answers.opencv.org/question/16422/matlab-gradient-to-c/
        A =
        1     3
        4     2

        [dx dy] = gradient(A, 4, 4)

        Output:

        dx =
        0.5000    0.5000
        -0.5000   -0.5000

        dy =
        0.7500   -0.2500
        0.7500   -0.2500
        """
        A = np.array([ \
                        [1.0,     3.0],
                        [4.0,     2.0]]);

        dx, dy = gradient(A, 4, 4);
        dxGood = np.array([ \
                        [0.5000,    0.5000],
                        [-0.5000,   -0.5000]]);
        dyGood = np.array([ \
                        [0.7500,   -0.2500],
                        [0.7500,   -0.2500]]);
        aZero = dx - dxGood;
        #common.DebugPrint("testGradient(): dx = %s" % str(dx));
        #common.DebugPrint("testGradient(): aZero = %s" % str(aZero));
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());
        aZero = dy - dyGood;
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());


        #common.DebugPrint("For A = %s we have the following gradients:" % str(A));
        #common.DebugPrint("dx = %s" % str(dx));
        #common.DebugPrint("dy = %s" % str(dy));


        dx, dy = gradient(A, 1, 1);
        """
        The Matlab results are:
            dx =
                 2     2
                -2    -2

            dy =
                 3    -1
                 3    -1
        """
        dxGood = np.array([ \
                        [2,    2],
                        [-2,   -2]]);
        dyGood = np.array([ \
                        [3,   -1],
                        [3,   -1]]);
        aZero = dx - dxGood;
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());
        aZero = dy - dyGood;
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());



        A = np.array([ \
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]);

        dx, dy = gradient(A, 4, 4);

        """
        The Matlab results are:
            dx =
                0.2500    0.2500    0.2500
                0.2500    0.2500    0.2500
                0.2500    0.2500    0.2500

            dy =
                0.7500    0.7500    0.7500
                0.7500    0.7500    0.7500
                0.7500    0.7500    0.7500
        """
        dxGood = np.array([ \
                        [0.2500,   0.2500,   0.2500],
                        [0.2500,   0.2500,   0.2500],
                        [0.2500,   0.2500,   0.2500]]);
        dyGood = np.array([ \
                        [0.7500,   0.7500,   0.7500],
                        [0.7500,   0.7500,   0.7500],
                        [0.7500,   0.7500,   0.7500]]);
        aZero = dx - dxGood;
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());
        aZero = dy - dyGood;
        self.assertTrue((np.abs(aZero) < 1.0e-3).all());

        #common.DebugPrint("For A = %s we have the following gradients:" % str(A));
        #common.DebugPrint("dx = %s" % str(dx));
        #common.DebugPrint("dy = %s" % str(dy));

        # Testing threedimensional gradient (for subframe sequences):
        A3d = np.zeros( (3, 3, 4) );
        for i in range(A3d.shape[2]):
            A3d[:, :, i] = A + (i + 1) * np.eye(A3d.shape[0]);

        #dx, dy, dz = gradient(A3d, 4, 4, 4);
        dx, dy, dz = gradient(A3d, 1, 1, 1);
        common.DebugPrint("For A3d = %s we have the following gradients:" % Repr3DMatrix(A3d)); #str(A3d));
        #common.DebugPrint("dx = %s" % str(dx));
        common.DebugPrint("dx = %s" % Repr3DMatrix(dx));
        #common.DebugPrint("dy = %s" % str(dy));
        common.DebugPrint("dy = %s" % Repr3DMatrix(dy));
        #common.DebugPrint("dz = %s" % str(dz));
        common.DebugPrint("dz = %s" % Repr3DMatrix(dz));

        """
        The following Matlab program:
            A1 = [1 2 3; 4 5 6; 7 8 9];
            %A = zeros(3,3,3);
            %A(:,:,1) = A1;
            %A(:,:,2) = A1;
            %A(:,:,3) = A1;
            %[vx, vy, vt] = gradient(A);

            A = zeros(3,3,4);
            A(:,:,1) = A1 + 1*eye(3);
            A(:,:,2) = A1 + 2*eye(3);
            A(:,:,3) = A1 + 3*eye(3);
            A(:,:,4) = A1 + 4*eye(3);

            Trial>> [vx, vy, vt] = gradient(A)

            vx(:,:,1) =
                     0    0.5000    1.0000
                2.0000    1.0000         0
                1.0000    1.5000    2.0000

            vx(:,:,2) =
                -1     0     1
                 3     1    -1
                 1     2     3

            vx(:,:,3) =
               -2.0000   -0.5000    1.0000
                4.0000    1.0000   -2.0000
                1.0000    2.5000    4.0000

            vx(:,:,4) =
                -3    -1     1
                 5     1    -3
                 1     3     5

            vy(:,:,1) =
                2.0000    4.0000    3.0000
                2.5000    3.0000    3.5000
                3.0000    2.0000    4.0000

            vy(:,:,2) =
                 1     5     3
                 2     3     4
                 3     1     5

            vy(:,:,3) =
                     0    6.0000    3.0000
                1.5000    3.0000    4.5000
                3.0000         0    6.0000

            vy(:,:,4) =
                -1     7     3
                 1     3     5
                 3    -1     7

            vt(:,:,1) =
                 1     0     0
                 0     1     0
                 0     0     1

            vt(:,:,2) =
                 1     0     0
                 0     1     0
                 0     0     1

            vt(:,:,3) =
                 1     0     0
                 0     1     0
                 0     0     1

            vt(:,:,4) =
                 1     0     0
                 0     1     0
                 0     0     1
        """


    def testMeshgrid(self):
        """
        From ...

        [X,Y] = meshgrid(1:3,10:14)
        X =
            1     2     3
            1     2     3
            1     2     3
            1     2     3
            1     2     3
        Y =
            10    10    10
            11    11    11
            12    12    12
            13    13    13
            14    14    14
        """
        resX, resY = meshgrid( range(1, 3 + 1) , range(10, 14 + 1));
        goodX = np.array( [ \
            [1,    2,    3],
            [1,    2,    3],
            [1,    2,    3],
            [1,    2,    3],
            [1,    2,    3]]);

        goodY = np.array( [ \
            [10,  10,   10],
            [11,  11,   11],
            [12,  12,   12],
            [13,  13,   13],
            [14,  14,   14]]);

        aZero = resX - goodX;
        self.assertTrue((aZero == 0).all());

        aZero = resY - goodY;
        self.assertTrue((aZero == 0).all());


    def testFspecial(self):
        # Tests are generated first with Matlab
        si = 2.4;
        #%p2 = max(1,fix(6*si)+1);
        p2 = 15.0;
        p3 = si;

        res = fspecial("gaussian", p2, p3);
        common.DebugPrint("testFspecial(): res.shape = %s" % str(res.shape));
        common.DebugPrint("testFspecial(): res = %s" % str(res));

        #Result from Matlab:
        resGood = np.array([ \
            [0.0000,   0.0000,   0.0000,   0.0001,   0.0002,   0.0003,   0.0004,   0.0004,
                0.0004,   0.0003,   0.0002,   0.0001,   0.0000,   0.0000,   0.0000],
            [0.0000,   0.0001,   0.0001,   0.0003,   0.0006,   0.0009,   0.0011,   0.0012,
                0.0011,   0.0009,   0.0006,   0.0003,   0.0001,   0.0001,   0.0000],
            [0.0000,   0.0001,   0.0004,   0.0008,   0.0014,   0.0022,   0.0029,   0.0032,
                0.0029,   0.0022,   0.0014,   0.0008,   0.0004,   0.0001,   0.0000],
            [0.0001,   0.0003,   0.0008,   0.0017,   0.0032,   0.0049,   0.0063,   0.0069,
                0.0063,   0.0049,   0.0032,   0.0017,   0.0008,   0.0003,   0.0001],
            [0.0002,   0.0006,   0.0014,   0.0032,   0.0058,   0.0090,   0.0116,   0.0127,
                0.0116,   0.0090,   0.0058,   0.0032,   0.0014,   0.0006,   0.0002],
            [0.0003,   0.0009,   0.0022,   0.0049,   0.0090,   0.0138,   0.0180,   0.0196,
                0.0180,   0.0138,   0.0090,   0.0049,   0.0022,   0.0009,   0.0003],
            [0.0004,   0.0011,   0.0029,   0.0063,   0.0116,   0.0180,   0.0233,   0.0254,
                0.0233,   0.0180,   0.0116,   0.0063,   0.0029,   0.0011,   0.0004],
            [0.0004,   0.0012,   0.0032,   0.0069,   0.0127,   0.0196,   0.0254,   0.0277,
                0.0254,   0.0196,   0.0127,   0.0069,   0.0032,   0.0012,   0.0004],
            [0.0004,   0.0011,   0.0029,   0.0063,   0.0116,   0.0180,   0.0233,   0.0254,
                0.0233,   0.0180,   0.0116,   0.0063,   0.0029,   0.0011,   0.0004],
            [0.0003,   0.0009,   0.0022,   0.0049,   0.0090,   0.0138,   0.0180,   0.0196,
                0.0180,   0.0138,   0.0090,   0.0049,   0.0022,   0.0009,   0.0003],
            [0.0002,   0.0006,   0.0014,   0.0032,   0.0058,   0.0090,   0.0116,   0.0127,
                0.0116,   0.0090,   0.0058,   0.0032,   0.0014,   0.0006,   0.0002],
            [0.0001,   0.0003,   0.0008,   0.0017,   0.0032,   0.0049,   0.0063,   0.0069,
                0.0063,   0.0049,   0.0032,   0.0017,   0.0008,   0.0003,   0.0001],
            [0.0000,   0.0001,   0.0004,   0.0008,   0.0014,   0.0022,   0.0029,   0.0032,
                0.0029,   0.0022,   0.0014,   0.0008,   0.0004,   0.0001,   0.0000],
            [0.0000,   0.0001,   0.0001,   0.0003,   0.0006,   0.0009,   0.0011,   0.0012,
                0.0011,   0.0009,   0.0006,   0.0003,   0.0001,   0.0001,   0.0000],
            [0.0000,   0.0000,   0.0000,   0.0001,   0.0002,   0.0003,   0.0004,   0.0004,
                0.0004,   0.0003,   0.0002,   0.0001,   0.0000,   0.0000,   0.0000]]);


        aZero = np.abs(res - resGood);
        common.DebugPrint("aZero = %s" % str(aZero));
        self.assertTrue((aZero < 1e-4).all());


    def testImfilter(self):
        # Not tested, since not implemented

        """
        A = np.array([ \
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]);

        res = imfilter(A, np.array([6])); #imfilter(A, 6, 1);
        #common.DebugPrint("res from imfilter() = %s" % str(res));

        # This is the result from Matlab
        resGood = np.array([ \
                        [11,   12,   11],
                        [11,   12,   11],
                        [11,   12,   11]]);
        """

        """
        To test well imfilter use a filter, as given in ecc_homo_spacetime.imgaussian():
            H = np.exp(-(x ** 2 / (2.0 * pow(sigma, 2))));
            H = H / H[:].sum();
            Hx = H.reshape( (H.size, 1) );
            res = imfilter(I, Hx, "same", "replicate");
        """
        A = np.array( [ \
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]);
        res = imfilter(np.ones((4, 4)), A, 'replicate');
        resGood = 45 * np.ones((4, 4));
        aZero = np.abs(res - resGood);
        #common.DebugPrint("aZero = %s" % str(aZero));
        self.assertTrue((aZero < 1e-4).all());


        B = np.array( [ \
                        [1, 2],
                        [3, 4]]);

        res = imfilter(np.ones((4, 4)), B, 'replicate');
        #common.DebugPrint("testImfilter(): res = %s" % str(res));

        resGood = np.array([ \
                        [10,   10,   10,   10],
                        [10,   10,   10,   10],
                        [10,   10,   10,   10],
                        [10,   10,   10,   10]]);

        aZero = np.abs(res - resGood);
        #common.DebugPrint("aZero = %s" % str(aZero));
        self.assertTrue((aZero < 1e-4).all());


        C = np.array( [ \
                        [1, 2],
                        [2, 1]]);

        res = imfilter(np.ones((5, 5)), C, 'replicate');
        #common.DebugPrint("testImfilter(): res = %s" % str(res));

        resGood = np.array([ \
                        [6,   6,   6,   6,   6],
                        [6,   6,   6,   6,   6],
                        [6,   6,   6,   6,   6],
                        [6,   6,   6,   6,   6],
                        [6,   6,   6,   6,   6]]);
        aZero = np.abs(res - resGood);
        #common.DebugPrint("aZero = %s" % str(aZero));
        self.assertTrue((aZero < 1e-4).all());


    def testInterp2(self):
        """
        V = np.array( [ \
                        [ 2,  8,  6],
                        [ 8, 10, 12],
                        [14, 16, 18]]);
        """

        if False:
            V = np.array( [ \
                        [ 1,  2,  3],
                        [ 4,  5,  6],
                        [ 7,  8,  9]]);

        V = np.array( [ \
                        [ 1,  5,  3],
                        [ 4,  7,  6],
                        [ 7,  10,  9]]);

        V = V.astype(np.float64);

        #if False:
        if True:
            # Testing out of bound
            Xq = np.array( [ \
                        [ 1.7,  2.7,  3.7],
                        [ 1.4,  2.5,  3],
                        [ 1,  2,  3]]);
        else:
            # Testing element coordinates not "in its square" - elem [0,2]
            Xq = np.array( [ \
                        [ 1.7,  2.7,  2.9],
                        [ 1.4,  2.5,  3],
                        [ 1,  2,  3]]);

        Yq = np.array( [ \
                        [ 1.3,  1,  1],
                        [ 2,  2,  2],
                        [ 3,  3,  3]]);

        Xq -= 1; # In Python we start from index 0
        Yq -= 1; # In Python we start from index 0

        itype = "linear";

        # interp2_nested_loops() is deemed to be correct

        Xq = Xq.astype(np.float64);
        Yq = Yq.astype(np.float64);

        #res = interp2_nested_loops(V, Xq, Yq, itype);
        res = interp2(V, Xq, Yq, itype);

        #if False:
        if True:
            if False:
                # For the (nonfunctioning) implementation with cv.remap()
                common.DebugPrint("interp2() returned %s" % str(ConvertCvMatToNPArray(res)));
            else:
                common.DebugPrint("interp2() returned %s" % str(res));

        goodRes = np.array( [ \
                            [  4.49,   3.6,  np.nan],
                            [  5.2 ,   6.5,    6.  ],
                            [  7.  ,   10.,    9.  ]]);

        """
        #common.DebugPrint("interp2() returned %s" % str(dir(res)));
        for r in range(res.rows):
            for c in range(res.cols):
                # NOT GOOD: common.DebugPrint("%.5f " % res[r][c][0]),
                #common.DebugPrint("%.5f " % res.__getitem__((r, c))),
                common.DebugPrint("%.5f " % res[r, c]),
                pass
            print
        #common.DebugPrint("interp2() returned %s" % str(res.tostring())); # Prints binary data...
        """

        aZero = res - goodRes;

        #common.DebugPrint("interp2() returned %s" % str(aZero));

        self.assertTrue(np.isnan(res[0, 2]) and np.isnan(goodRes[0, 2]));
        aZero[0, 2] = 0.0;
        self.assertTrue((np.abs(aZero) < 1.e-5).all());

        res = interp2(V, Xq, Yq, itype);

        """
        This is one (~small) test that my BAD vectorized interp2()
          implementation failed.
        """
        V = np.array([ \
                        [ 138., 131., 123., 118., 121., 130., 142., 150., 142., 142.],
                        [ 109.,  75.,  43.,  45.,  80., 115., 128., 126., 142., 142.],
                        [ 136., 142., 146., 146., 142., 139., 142., 145., 142., 142.],
                        [ 130., 135., 140., 143., 142., 140., 140., 140., 142., 142.],
                        [ 133., 124., 117., 122., 136., 147., 150., 147., 142., 142.],
                        [ 130., 144., 157., 154., 142., 132., 132., 138., 142., 142.],
                        [ 137., 143., 147., 143., 135., 132., 138., 146., 142., 142.],
                        [ 142., 144., 144., 137., 130., 131., 140., 150., 142., 142.],
                        [ 143., 143., 143., 143., 143., 143., 143., 143., 143., 143.],
                        [ 143., 143., 143., 143., 143., 143., 143., 143., 143., 143.]]);
        Xq = np.array([ \
                    [ 0.6977, 1.6883, 2.679,  3.6698, 4.6606, 5.6515, 6.6424, 7.6334, 8.6245, 9.6156],
                    [ 0.6951, 1.6857, 2.6764, 3.6672, 4.658,  5.6488, 6.6398, 7.6308, 8.6218, 9.6129],
                    [ 0.6925, 1.6831, 2.6738, 3.6645, 4.6553, 5.6462, 6.6371, 7.6281, 8.6191, 9.6102],
                    [ 0.6899, 1.6805, 2.6712, 3.6619, 4.6527, 5.6436, 6.6345, 7.6254, 8.6165, 9.6076],
                    [ 0.6872, 1.6779, 2.6685, 3.6593, 4.6501, 5.6409, 6.6318, 7.6228, 8.6138, 9.6049],
                    [ 0.6846, 1.6752, 2.6659, 3.6566, 4.6474, 5.6383, 6.6292, 7.6201, 8.6111, 9.6022],
                    [ 0.682,  1.6726, 2.6633, 3.654,  4.6448, 5.6356, 6.6265, 7.6175, 8.6085, 9.5995],
                    [ 0.6794, 1.67,   2.6607, 3.6514, 4.6422, 5.633,  6.6239, 7.6148, 8.6058, 9.5969],
                    [ 0.6768, 1.6674, 2.6581, 3.6488, 4.6395, 5.6303, 6.6212, 7.6121, 8.6031, 9.5942],
                    [ 0.6742, 1.6648, 2.6554, 3.6461, 4.6369, 5.6277, 6.6186, 7.6095, 8.6005, 9.5915]]);
        Yq = np.array([ \
                    [ 0.0144, 0.0115, 0.0086, 0.0057, 0.0027,-0.0002,-0.0031,-0.006, -0.0089,-0.0119],
                    [ 1.0134, 1.0105, 1.0076, 1.0048, 1.0019, 0.999,  0.9961, 0.9932, 0.9903, 0.9874],
                    [ 2.0124, 2.0096, 2.0067, 2.0038, 2.001,  1.9981, 1.9953, 1.9924, 1.9895, 1.9867],
                    [ 3.0114, 3.0086, 3.0057, 3.0029, 3.0001, 2.9972, 2.9944, 2.9916, 2.9888, 2.9859],
                    [ 4.0104, 4.0076, 4.0048, 4.002,  3.9992, 3.9964, 3.9936, 3.9908, 3.988,  3.9852],
                    [ 5.0093, 5.0065, 5.0038, 5.001,  4.9982, 4.9955, 4.9927, 4.9899, 4.9871, 4.9844],
                    [ 6.0082, 6.0055, 6.0028, 6.,     5.9973, 5.9945, 5.9918, 5.9891, 5.9863, 5.9836],
                    [ 7.0072, 7.0045, 7.0017, 6.999,  6.9963, 6.9936, 6.9909, 6.9882, 6.9855, 6.9827],
                    [ 8.0061, 8.0034, 8.0007, 7.998,  7.9953, 7.9927, 7.99,   7.9873, 7.9846, 7.9819],
                    [ 9.005,  9.0023, 8.9997, 8.997,  8.9943, 8.9917, 8.989,  8.9864, 8.9837, 8.9811]]);

        if False:
            # Measuring the performance of interp2().
            t1 = float(cv2.getTickCount());
            for i in range(10000):
                interp2(V, Xq, Yq, itype);
            t2 = float(cv2.getTickCount());
            myTime = (t2 - t1) / cv2.getTickFrequency();
            common.DebugPrint("testInterp2(): interp2() " \
                                "took %.6f [sec]" % myTime);

        # IMPORTANT test
        if True:
            #V1 = np.empty();
            V = imresize(V, scale=100);
            Xq = imresize(Xq, scale=100);
            Yq = imresize(Yq, scale=100);
            res = interp2(V, Xq, Yq, itype);
            #common.DebugPrint("interp2() returned %s" % str(res));

            resGood = interp2_vectorized(V, Xq, Yq, itype);
            #common.DebugPrint("interp2_vectorized() returned %s" % str(res));
            self.assertTrue(CompareMatricesWithNanElements(res, resGood));

        if False: # These are just performance tests (compare times of running unit-tests)
           V = np.zeros( (1920, 1080) );
           Xq = np.zeros( (1920, 1080) );
           Yq = np.zeros( (1920, 1080) );
           res = interp2(V, Xq, Yq, itype);
           res = interp2(V, Xq, Yq, itype);
           res = interp2_vectorized(V, Xq, Yq, itype);
           res = interp2_vectorized(V, Xq, Yq, itype);


    def testKron(self):
        # Taken from the Matlab help:
        A = np.eye(4);
        B = np.array([[1, -1], [-1, 1]]);

        res = kron(A, B);

        resGood = np.array( [ \
           [ 1,   -1,    0,    0,    0,    0,    0,    0],
           [-1,    1,    0,    0,    0,    0,    0,    0],
           [ 0,    0,    1,   -1,    0,    0,    0,    0],
           [ 0,    0,   -1,    1,    0,    0,    0,    0],
           [ 0,    0,    0,    0,    1,   -1,    0,    0],
           [ 0,    0,    0,    0,   -1,    1,    0,    0],
           [ 0,    0,    0,    0,    0,    0,    1,   -1],
           [ 0,    0,    0,    0,    0,    0,   -1,    1]]);

        aZero = res - resGood;
        self.assertTrue((aZero == 0).all());


        A = np.array( [[1, 2, 3], [4, 5, 6]] );
        #A = np.r_[np.array([1, 2, 3]), np.array([4, 5, 6])];

        #A = np.ravel(A);
        B = np.ones( (2, 2) );

        #common.DebugPrint("testKron(): B = %s[END]" % str(B));
        #common.DebugPrint("testKron(): B.shape = %s[END]" % str(B.shape));

        res = np.kron(A, B);
        resGood = np.array( [ \
             [1,    1,    2,    2,    3,    3],
             [1,    1,    2,    2,    3,    3],
             [4,    4,    5,    5,    6,    6],
             [4,    4,    5,    5,    6,    6]] );

        #common.DebugPrint("testKron(): A = %s" % str(A));
        #common.DebugPrint("testKron(): res = %s" % str(res));
        aZero = res - resGood;
        self.assertTrue((aZero == 0).all());


    def testSub2ind(self):
        # Taken from the Matlab help of sub2ind
        A = np.empty((3, 4, 2));
        r = np.array([3, 2, 3, 1, 2]);
        c = np.array([3, 4, 1, 3, 4]);
        d3 = np.array([2, 1, 2, 2, 1]);

        r -= 1;
        c -= 1;
        d3 -= 1;

        res = sub2ind(matrixSize=A.shape, rowSub=r, colSub=c, dim3Sub=d3);

        # Taken from the Matlab help of sub2ind
        resGood = np.array([21, 11, 15, 19, 11]);
        resGood -= 1;

        #common.DebugPrint("testSub2ind(): res = %s" % str(res));
        aZero = res - resGood;
        self.assertTrue((aZero == 0).all());


    def testOrdfilt2(self):
        A = np.array([ \
                        [ 1,  2,  3,  4],
                        [ 5,  6,  7,  8],
                        [ 9, 10, 11, 12],
                        [13, 14, 15, 16]]);

        A = A.astype(np.float64);

        #res = ordfilt2_4_nested_loops(A, 9, np.ones((3, 3)));
        res = ordfilt2(A, 9, np.ones((3, 3)));
        common.DebugPrint("testOrdfilt2(): res = %s" % str(res));

        resGood = np.array([ \
                        [ 6,    7,    8,    8],
                        [10,   11,   12,   12],
                        [14,   15,   16,   16],
                        [14,   15,   16,   16]]);

        aZero = res - resGood;
        #self.assertTrue(np.abs(aZero == 0).all());
        self.assertTrue(np.abs(aZero < 1.0e-4).all());

        """
        >> A = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
        >> ordfilt2(A, 0, ones(3))

        ans =

          1.0e-307 *

            0.4450    0.4450    0.4450    0.4450
            0.4450    0.4450    0.4450    0.4450
            0.4450    0.4450    0.4450    0.4450
            0.4450    0.4450    0.4450    0.4450

        >> ordfilt2(A, 1, ones(3))

        ans =

             0     0     0     0
             0     1     2     0
             0     5     6     0
             0     0     0     0

        >> ordfilt2(A, 2, ones(3))

        ans =

             0     0     0     0
             0     2     3     0
             0     6     7     0
             0     0     0     0

        >> ordfilt2(A, 3, ones(3))

        ans =

             0     0     0     0
             0     3     4     0
             0     7     8     0
             0     0     0     0

        >> ordfilt2(A, 4, ones(3))

        ans =

             0     1     2     0
             1     5     6     3
             5     9    10     7
             0     9    10     0

        >> ordfilt2(A, 5, ones(3))

        ans =

             0     2     3     0
             2     6     7     4
             6    10    11     8
             0    10    11     0

        >> ordfilt2(A, 6, ones(3))

        ans =

             1     3     4     3
             5     7     8     7
             9    11    12    11
             9    11    12    11

        >> ordfilt2(A, 7, ones(3))

        ans =

             2     5     6     4
             6     9    10     8
            10    13    14    12
            10    13    14    12

        >> ordfilt2(A, 8, ones(3))

        ans =

             5     6     7     7
             9    10    11    11
            13    14    15    15
            13    14    15    15

        >> ordfilt2(A, 9, ones(3))

        ans =

             6     7     8     8
            10    11    12    12
            14    15    16    16
            14    15    16    16

        >> ordfilt2(A, 10, ones(3))

        ans =

             0     0     0     0
             0     0     0     0
             0     0     0     0
             0     0     0     0
        """



if __name__ == '__main__':
    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(precision=4, suppress=True, \
                        threshold=1000000, linewidth=5000);
    unittest.main();

