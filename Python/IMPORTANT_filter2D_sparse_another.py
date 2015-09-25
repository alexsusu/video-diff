imfilter(A, H, options=None):
    reduces to:
      return cv2.filter2D(src=A, ddepth=-1, kernel=H, anchor=(-1, -1), \
                        borderType=cv2.BORDER_REPLICATE);

    http://docs.opencv.org/modules/imgproc/doc/filtering.html#filter2d
        \texttt{dst} (x,y) = \sum _{ \stackrel{0\leq x' < \texttt{kernel.cols},}{0\leq y' < \texttt{kernel.rows}} } \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )


    // The standard way of computing filter2D
    ay = rowsK / 2;
    ax = colsK / 2;

    /*
    //BAD
    for (ys = 0; ys < rowsSrc; ys++) {
        for (xs = 0; xs < colsSrc; xs++) {
            for (y = 0; y < rowsK; y++) {
                yf = ys + y - ay;
                for (x = 0; x < colsK; x++) {
                    xf = xs + x - ax;
                    if (yf, xf) in bounds {
                        res += src[yf][xf];
                    }
                }
            }
            dst[ys][xs] = res;
        }
    }
    */
    for (yd = 0; yd < rowsSrc; yd++) {
        for (xd = 0; xd < colsSrc; xd++) {
            // we compute the value for the dst point (yd, xd)
            res = 0;
            for (yk = 0; yk < rowsK; yk++) {
                ys = yd + yk - ay;
                for (xk = 0; xk < colsK; xk++) {
                    xs = xd + xk - ax;
                    if (ys, xs) in bounds {
                        res += src[ys][xs];
                    }
                }
            }
            dst[yd][xd] = res;
        }
    }




    // The inverse way of computing filter2D
    ay = rowsK / 2;
    ax = colsK / 2;

    dst = zeros matrix;

    for (y = 0; y < rowsK; y++)
        for (x = 0; x < colsK; x++)
            res[rowsK][colsK] = 0;

    for (ys = 0; ys < rowsSrc; ys++) {
        for (xs = 0; xs < colsSrc; xs++) {
            // For each source matrix point we add its contribution to the dst matrix
            for (yk_start = -rowsK; yk_start <= 0; yk_start++) {
                // yd is the center of the kernel starting at yk_start
                yd = ys + yk_start + ay;
                for (xk_start = 0; xk_start < colsK; xk_start++) {
                    xd = xs + xk_start + ax;
                    if (yd, xd) in bounds {
                        dst[yd][xd] += src[ys][xs];
                    }
                }
            }
        }
    }

    



cv::INTER_LINEAR - does it do interpolation or chooses nearest?
    following https://github.com/Itseez/opencv/blob/03fc3d1ceb867cbd6882e2a2809a196582d0efc1/modules/imgproc/src/imgwarp.cpp
         class WarpPerspectiveInvoker :
            ...
            virtual void operator() (const Range& range) const
            ...

                if( interpolation == INTER_NEAREST )


    warpPerspective - ce foloseste in spate? vezi implementare Matlab
        https://github.com/Itseez/opencv/blob/5efad375e0b1eae0563e40d02030b4b39309d48a/modules/imgproc/src/imgwarp.cpp
        **
        if False:
            #!!!!TODO_PROFOUND: understand why we get poorer results with warpPerspective than with interp_space_time - see /home/asusu/drone-diff_test_against_EV_videos/Videos/output (*_good_new/new0.png) VS *_good.png
            wout = cv2.warpPerspective(src=refFrame, M=warp_p, \
                                dsize=(refFrame.shape[1], refFrame.shape[0]));
        else:
            wout = interp_space_time(volumeA, img_index, warp_p, t, int_type, \
                                                        nx, ny, pixel_select); : does uv = np.dot(M, xy), np.r_, np.ravel, Matlab.interp2

    // From https://github.com/Itseez/opencv/blob/03fc3d1ceb867cbd6882e2a2809a196582d0efc1/modules/imgproc/src/imgwarp.cpp
    void cv::warpPerspective( InputArray _src, OutputArray _dst, InputArray _M0,
    Size dsize, int flags, int borderType, const Scalar& borderValue )
    {
        ...
        if( !(flags & WARP_INVERSE_MAP) )
            invert(matM, matM);
        Range range(0, dst.rows);
        WarpPerspectiveInvoker invoker(src, dst, M, interpolation, borderType, borderValue);
        parallel_for_(range, invoker, dst.total()/(double)(1<<16));
    }

If using OpenCV's ECC we need to use warpPerspective and ...

sparse ECC_homo_spacetime.py needs:
    !!!!Matlab.interp2, imresize, imfilter, mean2, gradient (uses numpy) - should we use LIL or is it ok with CSC/CSR?
        It appears that all these functions (check gradient!!!!) are optimally? computed using a CSC/CSR representation, going backwards - we compute the contribution of each source point to the destination matrix. Doing so we perform a standard iteration over elements, in the order they are stored.
    imgaussian - basically reduces to imfilter
        #function I=imgaussian(I,sigma,siz)
        if sigma > 0:
            #% Make 1D Gaussian kernel
            #x=-ceil(siz/2):ceil(siz/2);
            x = range(-math.ceil(siz / 2.0), math.ceil(siz / 2.0) + 1);

            #H = exp(-(x.^2/(2*sigma^2)));
            H = np.exp(-(x ** 2 / (2.0 * pow(sigma, 2))));

            #H = H/sum(H(:));
            H = H / H[:].sum();

            #% Filter each dimension with the 1D Gaussian kernels\
            #if(ndims(I)==1)
            #if len(I.shape) == 1:
            if I.ndim == 1:
                assert False;
                #I=imfilter(I,H, 'same' ,'replicate');
                I = imfilter(I, H, "same", "replicate");
            #elseif(ndims(I)==2)
            elif I.ndim == 2:
                #Hx=reshape(H,[length(H) 1]);
                Hx = H.reshape((H.size, 1), order="F");

                #Hy=reshape(H,[1 length(H)]);
                Hy = H.reshape((1, H.size), order="F");

                #I=imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
                I = imfilter( \
                        imfilter(I, Hx, "same", "replicate"), \
                        Hy, "same", "replicate");
            #elseif(ndims(I)==3)
            elif I.ndim == 3:
                assert False;
                """
                if(size(I,3)<4) % Detect if 3D or color image
                    Hx=reshape(H,[length(H) 1]);
                    Hy=reshape(H,[1 length(H)]);
                    for k=1:size(I,3)
                        I(:,:,k)=imfilter(imfilter(I(:,:,k),Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
                    end
                else
                    Hx=reshape(H,[length(H) 1 1]);
                    Hy=reshape(H,[1 length(H) 1]);
                    Hz=reshape(H,[1 1 length(H)]);
                    I=imfilter(imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate'),Hz, 'same' ,'replicate');
                """
            else:
                if common.MY_DEBUG_STDOUT:
                    common.DebugPrint( \
                        "imgaussian:input: unsupported input dimension");
                assert False;

        return I;
    - interp_space_time() - might not be feasible as sparse
        - similar to warpPerspective:
          // from http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#warpperspective
            \texttt{dst} (x, y) = \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} , \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )

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
         - another way to compute this is to use a DOK representation of the src sparse matrix. BUT the biggest disadvantage is that we would have to query each point of the dense dst matrix (in the end dst could become sparse, but clearly this standard way of computing warpPerspective() is at least as inefficient as the complete dense method)
            - less relevant: otherwise, the next best is to use CSC representation and do binary search over the elements of a column
