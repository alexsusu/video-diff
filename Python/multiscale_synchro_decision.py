import common
import config
import Matlab

import math
import sys

import numpy as np
from numpy import linalg as npla

import cv2


CAUSAL_DO_NOT_SMOOTH = True;


#function index = randIndex(maxIndex,len)
def randIndex(maxIndex, aLen):
    #%INDEX = RANDINDEX(MAXINDEX,LEN)
    #%   randomly, non-repeatedly select LEN integers from 1:MAXINDEX

    if aLen > maxIndex:
        index = np.array([])
        return

    index = np.zeros( (1, aLen) );

    #available = 1:maxIndex
    available = range(1, maxIndex);

    """
    From Matlab help:
      r = rand(n) returns an n-by-n matrix
        containing pseudorandom values drawn from the standard uniform distribution
        on the open interval (0,1).
      r = rand(m,n) or r = rand([m,n]) returns an m-by-n matrix.
    """
    #rs = ceil(rand(1,len).*(maxIndex:-1:maxIndex-len+1));
    rs = np.ceil(np.random.rand(1,len) * \
            range(maxIndex, maxIndex - aLen + 1 + -1, -1));

    for p in range(1, len + 1):
        while rs[p - 1] == 0:
            #rs(p) = ceil(rand(1)*(maxIndex-p+1));
            rs[p - 1] = np.ceil(rand(1)*(maxIndex-p+1));

        #index(p) = available(rs(p));
        index[p - 1] = available[rs[p - 1]];

        #available(rs(p)) = [];
        available[rs[p - 1]] = np.array([]);

    return index


# It seems ransac_line is NEVER called, since nop == 0 in causal()
#function [ alpha, beta ] = ransac_line( pts,iterNum,thDist,thInlrRatio )
def ransac_line(pts, iterNum, thDist, thInlrRatio):
    #%RANSAC Use RANdom SAmple Consensus to fit a line
    #%   RESCOEF = RANSAC(PTS,ITERNUM,THDIST,THINLRRATIO) PTS is 2*n matrix including
    #%   n points, ITERNUM is the number of iteration, THDIST is the inlier
    #%   distance threshold and ROUND(THINLRRATIO*SIZE(PTS,2)) is the inlier number threshold. The final
    #%   fitted line is y = alpha*x+beta.
    #%   Yan Ke @ THUEE, xjed09@gmail.com
    #
    #%   modified by georgios evangelidis

    #!!!!TODO: not finished implementing it since it's not used
    assert False;

    sampleNum = 2;

    ptNum = pts.shape[1];
    thInlr = np.round(thInlrRatio * ptNum);

    inlrNum = np.zeros( (1, iterNum) )
    theta1 = np.zeros( (1,iterNum) )
    rho1 = np.zeros( (1, iterNum) )

    for p in range(1, iterNum + 1):
        #% 1. fit using 2 random points
        sampleIdx = randIndex(ptNum, sampleNum);

        #ptSample = pts(:,sampleIdx);
        ptSample = pts[:, sampleIdx - 1];

        #d = ptSample(:,2)-ptSample(:,1);
        d = ptSample[:, 1] - ptSample[:, 1];

        #d=d/norm(d); #% direction vector of the line
        d = d / npla.norm(d); #% direction vector of the line

        #% 2. count the inliers, if more than thInlr, refit; else iterate
        #n = [-d(2),d(1)]; #% unit normal vector of the line
        n = np.c_[-d[1], d[0]]; #% unit normal vector of the line

        #dist1 = n*(pts-repmat(ptSample(:,1),1,ptNum));
        dist1 = n * (pts - repmat(ptSample[:, 0], 1, ptNum)); #!!!!TODO: check more

        inlier1 = find(abs(dist1) < thDist);

        #inlrNum(p) = length(inlier1);
        inlrNum[p - 1] = len(inlier1);

        #if length(inlier1) < thInlr, continue; end
        if len(inlier1) < thInlr:
            continue

        #ev = princomp(pts(:,inlier1)');
        ev = princomp(pts[:, inlier1].T);

        #d1 = ev(:,1);
        d1 = ev[:, 0];

        #theta1(p) = -atan2(d1(2),d1(1)); #% save the coefs
        theta1[p - 1] = - math.atan2(d1[1], d1[0]); #% save the coefs

        #rho1(p) = [-d1(2),d1(1)]*mean(pts(:,inlier1),2);
        rho1[p - 1] = [-d1(2),d1(1)] * pts[:, inlier1].mean(2);

    #% 3. choose the coef with the most inliers
    #[~,idx] = max(inlrNum);
    idx = argmax(inlrNum);

    theta = theta1[idx];
    rho = rho1[idx];

    alpha = -sin(theta) / cos(theta);
    beta  = rho / cos(theta);

    return alpha, beta


"""
About costs:
   - Evangelidis' causal() uses a
      mean over Vote space and weighted sum w.r.t. the scales
   - my causal() uses a simple sumation
   -
   - Evangelidis' dp3() starts from
      sum after scale of Vote space
     but also uses in the update phase of the memoization table the
          weights over the different scales.
"""
def ComputeCost(crossref, V, fileName="crossref.txt"):
    # V[r][q] = votes of ref frame r for query frame q

    print("ComputeCost(): V.shape = %s" % str(V.shape));
    print("ComputeCost(): crossref.shape = %s" % str(crossref.shape));

    if False:
        for q in range(67, 71):
            for r in range(V.shape[0]):
                print("  V[%d, %d] = %.7f" % (r + config.initFrame[1], q, V[r, q]));
            print;

    """
    print("ComputeCost(): crossref and V =");
    cost = 0.0;
    myText2 = "";
    for i in range(crossref.shape[0]):
        assert crossref[i][0] == i;
        cost += V[crossref[i][1]][i];
        print("[%d %d] %.7f" % (i, crossref[i][1], V[crossref[i][1]][i]));

    print("ComputeCost(): cost computed is %.7f" % cost);
    """

    #!!!!TODO TODO: print also a synchronization error (look at TPAMI 2013 Evangelidis)

    #crossref2 = crossref.copy();
    #crossref2[:, 1] += config.initFrame[1];

    numBack = 0;
    totalStep = 0;
    penaltyCost = 0;
    myMin = crossref[0][1]; #1000000;
    myMax = crossref[0][1]; #-1;
    for i in range(1, crossref.shape[0]):
        if myMin > crossref[i][1]:
            myMin = crossref[i][1];
        if myMax < crossref[i][1]:
            myMax = crossref[i][1];

        totalStep += abs(crossref[i][1] - crossref[i - 1][1]);
        penaltyCost += abs(crossref[i][1] - crossref[i - 1][1]); #!!!!TODO: check also if we stay too long in the same ref frame and penalize if more than 10-20 same value in a row

        if crossref[i][1] < crossref[i - 1][1]:
            numBack += 1;
    absAvgStep = totalStep / (crossref.shape[0] - 1);
    avgStep = (crossref[crossref.shape[0] - 1][1] - crossref[0][1]) / (crossref.shape[0] - 1);

    cost = 0.0;
    myText2 = "ComputeCost(): crossref and V =\n";
    for q in range(crossref.shape[0]):
        assert crossref[q][0] == q;
        try:
            cost += V[crossref[q][1]][q];
            myText2 += "[%d %d] %.7f; " % \
                        (q, crossref[q][1] + config.initFrame[1], \
                        V[crossref[q][1]][q]);
            #for r in range(int(crossref[q][1]) + config.initFrame[1] - 5, \
            #               int(crossref[q][1]) + config.initFrame[1] + 5):
            for r in range(int(crossref[q][1]) - 5, int(crossref[q][1]) + 5):
                if r < 0:
                    continue;
                if r >= V.shape[0]:
                    break;
                myText2 += "%.7f " % V[r, q];
        except:
            common.DebugPrintErrorTrace();

        """
        We print the first to nth order statistics - e.g., the first 5 biggest
          vote values.
        I got inspired from
          https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
         (see also
          https://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array)
        """
        myArr = V[:, q].copy();
        myArrIndices = myArr.argsort()[-5:][::-1];
        myText2 += "; max ind = %s" % str(myArrIndices + config.initFrame[1]);
        myText2 += "; max vals = %s" % str(myArr[myArrIndices]);
        myText2 += "\n";

    myText2 += "\n\ncost computed is %.7f\n" % cost;
    myText2 += "penalty is %.7f\n" % penaltyCost;
    myText2 += "reference frames are in the interval [%d, %d]\n" % \
                (myMin + config.initFrame[1], myMax + config.initFrame[1]);
    myText2 += "absolute avg step computed is %.7f\n" % absAvgStep;
    myText2 += "  avg step computed is %.7f\n" % avgStep;
    myText2 += "Number of times going back (numBack) is %d" % numBack;

    #!!!!TODO TODO: print also a synchronization error (look at TPAMI 2013 Evangelidis)

    #myText = "crossref = %s" % crossref2;
    #fOutput = open("crossref.txt", "wt");
    fOutput = open(fileName, "wt");
    fOutput.write(myText2);
    fOutput.close();



def causal_Alex(Vspace, numFramesQ, numFramesR):
    nos = Vspace.shape[2];
    # We transform nan in 0 in Vspace
    for i in range(1, nos + 1):
        #V_temp=Vspace(:,:,i);
        V_temp = Vspace[:, :, i - 1];

        #V_temp(isnan(V_temp))=0;
        V_temp[np.isnan(V_temp)] = 0;

        #Vspace(:,:,i)=V_temp;
        Vspace[:, :, i - 1] = V_temp;

    #!!!!TODO: we should use a weighted sum w.r.t. the scales, just like in causal()
    V = Vspace.sum(2);

    crossref = np.zeros( (numFramesQ, 2) );

    for iFor in range(numFramesQ):
        crossref[iFor, 0] = iFor;
        b = V[:, iFor].argmax();
        #a = V[:, iFor][b];
        crossref[iFor, 1] = b;

    print("causal_Alex(): crossref = %s" % str(crossref));

    ComputeCost(crossref, V, "crossref_causal_Alex.txt");
    print("causal_Alex(): END");

    #!!!!TODO: write crossref_causal_Alex.txt
    return crossref;


#function cross=multiscale_synchro_decision(Vspace, H, q_path, r_path, BOV_flag, cropflag, const_type, nop)
#def causal(Vspace, H, q_path, r_path, BOV_flag, cropflag, const_type, nop=0):
def causal(Vspace, H, numFramesQ, numFramesR, BOV_flag, cropflag, const_type, nop=0):
    #%cons_type : 1 means that spatial constraint applies after voting
    #%            2 means that spatial constraint applies before voting -> NOT
    #%            USED IN PAPER

    # causal() is the local/greedy optimization solution

    #% nargin>7 means that we do the local smoothing with RANSAC (see the paper)

    """
    common.DebugPrint("causal(): At entrance Vspace=%s,\nH=%s, \n" \
        "numFramesQ=%s, numFramesR=%s, BOV_flag=%d, cropflag=%d, const_type=%d, " \
        "nop=%d" % \
        (str(Vspace), str(H), numFramesQ, numFramesR, BOV_flag, cropflag, \
            const_type, nop));
    """
    print("causal(): At entrance \n" \
        "    numFramesQ=%s, numFramesR=%s, BOV_flag=%d, cropflag=%d, const_type=%d, " \
        "    nop=%d" % \
        (numFramesQ, numFramesR, BOV_flag, cropflag, \
            const_type, nop));

    # Normally BOV_flag=0, cropflag=0, const_type=1,nop=0

    if True:
        print("causal(): Vspace.shape = %s" % str(Vspace.shape));
        print("causal(): H.shape = %s" % str(H.shape));

        for i in range(Vspace.shape[2]):
            common.DebugPrint("causal(): Vspace[:, :, %d] = \n%s" % (i, str(Vspace[:, :, i])));
            """
            common.DebugPrint("causal(): Vspace[:, :, %d] = " % (i));
            for r in range(Vspace.shape[0]):
                for c in range(Vspace.shape[1]):
                    common.DebugPrint("%.4f " % Vspace[r, c, i]),
                print;
            """
        print

        for i in range(H.shape[2]):
            common.DebugPrint("causal(): H[:, :, %d] = \n%s" % (i, str(H[:, :, i])));
            """
            common.DebugPrint("causal(): H[:, :, %d] = " % (i));
            for r in range(H.shape[0]):
                for c in range(H.shape[1]):
                    common.DebugPrint("%.4f " % H[r, c, i]),
                print;
            """

    #nos=size(Vspace,3); %3D matrix
    nos = Vspace.shape[2]; #%3D matrix

    #% same with the multi_scale_harris function
    sigma_0 = 1.2;

    #n=0:nos-1;
    n = range(0, nos);

    #sigma_D=sqrt(1.8).^n*sigma_0; % We should see sqrt(1.8) returning a vector of nos elements
    #NOT GOOD in Python: sigma_D = math.sqrt(1.8)**n * sigma_0;
    sq18 = np.ones( (1, nos) ) * math.sqrt(1.8);
    sigma_D = sq18**n * sigma_0;

    #w = sigma_D;
    w = sigma_D[0];

    #common.DebugPrint("causal(): w = %s" % str(w));
    #common.DebugPrint("causal(): w.shape = %s" % str(w.shape));

    w = w / w.sum();
    #common.DebugPrint("causal(): w = %s" % str(w));

    # Alex: normally NOT executed
    if BOV_flag == 1:
        #w=fliplr(w);
        w = w[:, ::-1] # We flip on the vertical (left becomes right)

    print("causal(): w = %s" % str(w));

    # Alex: normally NOT executed
    if (const_type == 1) and (BOV_flag == 1):
        assert False; # Normally this code does NOT get executed
        #VV=zeros(size(Vspace,1),size(Vspace,2));
        VV = np.zeros( (Vspace.shape[0], Vspace.shape[1]) )

        for j in range(1, nos + 1):
            #VV=VV+Vspace(:,:,j);
            VV = VV + Vspace[:, :, j - 1];

        #[X,Y]=sort(VV,'descend');
        X, Y = sort(VV, "descend");

        #%enable top-N list
        #%N = 50;
        #%N = 100;
        N = 300;

        for s in range(1, nos + 1):
            for i in range(1, Vspace.shape[1] + 1):
                #y=Y(:,i);
                y = Y[:, i - 1];

                #votes=Vspace(:,i,s);
                votes = Vspace[:, i - 1, s - 1];

                #h=H(:,i,s);
                h = H[:, i - 1, s - 1];

                #votes(y(N+1:end))=0;
                votes[y[N :]] = 0;

                #h(y(N+1:end))=0;
                h[y[N:]] = 0;

                #Vspace(:,i,s)=votes;
                Vspace[:, i - 1, s - 1] = votes;

                #H(:,i,s)=h;
                H[:, i - 1, s - 1] = h;

    #clear VV
    VV = None;

    #QD=dir([q_path 'multiharlocs*.mat']);
    #RD=dir([r_path 'multiharlocs*.mat']);
    #QD = [None] * numFramesQ;
    #RD = [None] * numFramesR;

    #cross=zeros(length(QD),2);
    #crossref = np.zeros( (len(QD), 2) );
    crossref = np.zeros( (numFramesQ, 2) );
    #common.DebugPrint("causal(): crossref = %s" % str(crossref));


    #% str=['load ' r_path 'multiscale_nod'];
    #% eval(str)

    #%replace nan votes with zeros
    #%sum(sum(isnan(Vspace(:,:,1))))

    # We transform nan in 0 in Vspace
    """
    We substitute i - 1 with i, since array indexing starts from 1 in Matlab
        and 0 in Python.
    """
    #for i in range(1, nos + 1):
    for i in range(nos):
        #V_temp=Vspace(:,:,i);
        #V_temp = Vspace[:, :, i - 1];
        V_temp = Vspace[:, :, i];

        #V_temp(isnan(V_temp))=0;
        V_temp[np.isnan(V_temp)] = 0;

        #Vspace(:,:,i)=V_temp;
        #Vspace[:, :, i - 1] = V_temp;
        Vspace[:, :, i] = V_temp;

    # Alex: I personally find this idea of using filter2() VERY BAD - the results in Vspace should already be VERY good

    if CAUSAL_DO_NOT_SMOOTH == False:
        #% this filtering of votes favors smoother results
        #%b=ones(11,1);
        #%b=b/prod(size(b));
        b = Matlab.hamming(11); #% USED

        #if False:
        """
        We substitute i - 1 with i, since array indexing starts from 1 in Matlab
            and 0 in Python.
        """
        #for i in range(1, nos + 1):
        for i in range(nos):
            #Vspace(:,:,i)=filter2(b,Vspace(:,:,i));
            """
            From the Matlab help:
              Y = filter2(h,X) filters
              the data in X with the two-dimensional FIR filter
              in the matrix h. It computes the result, Y,
              using two-dimensional correlation, and returns the central part of
              the correlation that is the same size as X.
            """
            #Vspace[:, :, i - 1] = Matlab.filter2(b, Vspace[:, :, i - 1]);
            Vspace[:, :, i] = Matlab.filter2(b, Vspace[:, :, i]);
            #% use imfilter if filter2 takes time #!!!!TODO: do the optimization Evangelidis says here

    """
    From Matlab help:
      M = mean(A,dim) returns
      the mean values for elements along the dimension of A specified
      by scalar dim. For matrices, mean(A,2) is
      a column vector containing the mean value of each row.
    """

    #V=mean(Vspace,3); % this might help more instead of starting from zero votes
    V = Vspace.mean(2) # this might help more instead of starting from zero votes

    #common.DebugPrint("causal(): V.shape = %s" % str(V.shape));

    """
    We substitute i - 1 with i, since array indexing starts from 1 in Matlab
        and 0 in Python.
    """
    #for i in range(1, nos + 1):
    for i in range(nos):
        if cropflag == 0:
            if (const_type == 1) and (BOV_flag == 1):
                #V=V+w(i)*H(:,:,i)+Vspace(:,:,i);
                #V += w[i - 1] * H[:, :, i - 1] + Vspace[:, :, i - 1]; #!!!!TODO: think well *
                V += w[i] * H[:, :, i] + Vspace[:, :, i]; #!!!!TODO: think well *
            else:
                # Alex: we are normally in this case, since cropflag == 0, const_type == 1, BOV_flag == 0
                #V=V+w(i)*Vspace(:,:,i);
                #V += w[i - 1] * Vspace[:, :, i - 1]; #!!!!TODO: think well *   Exception "ValueError: operands could not be broadcast together with shapes (5) (23,8)"
                V += w[i] * Vspace[:, :, i]; #!!!!TODO: think well *   Exception "ValueError: operands could not be broadcast together with shapes (5) (23,8)"
        else:
            #V=V+w(i)*Vspace(:,:,i)+H(:,:,i);
            #V += w[i - 1] * Vspace[:, :, i - 1] + H[:, :, i - 1]; #!!!!TODO: think well *
            V += w[i] * Vspace[:, :, i] + H[:, :, i]; #!!!!TODO: think well *

    #common.DebugPrint("causal(): Vspace.shape = %s" % str(Vspace.shape));
    common.DebugPrint("causal(): V.shape = %s" % str(V.shape));
    common.DebugPrint("causal(): V (the matrix used to choose the max-voting reference frame) = %s" % str(V));
    #common.DebugPrint("causal(): len(QD) = %d" % len(QD));

    #for iFor in range(1, len(QD) + 1):
    """
    We substitute iFor -1 with iFor since arrays start with 0 in Python,
        not with 1 like in Matlab
    """
    #for iFor in range(1, numFramesQ + 1):
    for iFor in range(numFramesQ):
        #cross(i,1)=str2num(QD(i).name(end-9:end-4));
        crossref[iFor, 0] = iFor; #TODO - think well

        #[a,b]=max(V(:,i));
        b = V[:, iFor].argmax();
        a = V[:, iFor][b];

        #cross(i,2)=str2num(RD(b).name(end-9:end-4));
        crossref[iFor, 1] = b; #!!!!TODO - think well


    # We normally do NOT execute the following code, since nop == 0

    #if nargin>7
    if nop != 0:
        #XX = cross(:,1)';
        XX = crossref[:, 0].T;

        #YY = cross(:,2)';
        YY = crossref[:, 1].T;

        YY_new = YY;

        if mod(nop, 2) == 0:
            #error('Number of points (parameter NOP) must be odd number');
            #common.DebugPrint("Number of points (parameter NOP) must be odd number");
            quit();

        # miso is used only as array index and in range expressions --> it can be an integer
        miso = (nop - 1) / 2;

        if nop < 7:
            #common.DebugPrint("Choose more than 5 points for local fitting");
            pass

        #for i = miso+1:length(XX)-miso
        for iFor in range(miso + 1, len(XX) - miso + 1):
            #xx = XX(i-miso:i+miso);
            xx = XX[iFor - miso - 1 : iFor + miso];

            #yy = YY(i-miso:i+miso);
            yy = YY[iFor - miso - 1 : iFor + miso];

            if nop < 9:
                """
                iterNum is used only in range expressions and
                        array dimensions --> it can be an integer.
                """
                iterNum = nop * (nop - 1) / 2;
            else:
                iterNum = 10;

            thDist = 2;
            thInlrRatio = 0.6;

            """
            From Matlab help:
              RANSAC Use RANdom SAmple Consensus to fit a line
                RESCOEF = RANSAC(PTS,ITERNUM,THDIST,THINLRRATIO) PTS is 2*n matrix including
                n points, ITERNUM is the number of iteration, THDIST is the inlier
                distance threshold and ROUND(THINLRRATIO*SIZE(PTS,2)) is the inlier number threshold. The final
                fitted line is y = alpha*x+beta.
                Yan Ke @ THUEE, xjed09@gmail.com
            """
            #[ alpha, beta ] = ransac_line( [xx;yy],iterNum,thDist,thInlrRatio );
            alpha, beta = ransac_line( np.r_[xx, yy], iterNum, \
                                      thDist, thInlrRatio );

            if alpha != 0:
                yy_new = alpha * xx + beta; #!!!!TODO: think well *
                YY_new[iFor - 1] = yy_new[miso+1 - 1];

        #cross(:,2) = YY_new;
        crossref[:, 2] = YY_new;

    crossref = crossref.astype(int);

    print("causal(): crossref = %s" % str(crossref));

    ComputeCost(crossref, V, "crossref_causal.txt");

    if False: #True:
        #print("Am here1111");
        #causal_Alex(Vspace, numFramesQ, numFramesR);

        #print("Am here2222");
        #dp3(Vspace, numFramesR, numFramesQ, BOV_flag);

        #print("Am here3333");
        #dp_Alex(Vspace, numFramesR, numFramesQ, BOV_flag);
        dp_Alex(Vspace, numFramesR, numFramesQ, BOV_flag, PREV_REF=5, NEXT_REF=-1);

    return crossref;


#function [y,x,D,Tback,cross] = dp3(Vspace, r_path, q_path, BOV_flag)
def dp3(Vspace, numFramesR, numFramesQ, BOV_flag):
    #% Dynamic programming for a maximum-vote path in vote-space
    #% 2010, Georgios Evangelidis <georgios.evangelidis@iais.fraunhofer.de>

    print("Entered dp3(): Running dynamic programming...");
    common.DebugPrint("dp3(): Vspace = %s" % str(Vspace));

    #tic

    #QD=dir([q_path 'multiharlocs*.mat']);
    #RD=dir([r_path 'multiharlocs*.mat']);

    #cross=zeros(length(QD),2);
    #crossref = np.zeros( (len(q_path), 2) );
    crossref = np.zeros( (numFramesQ, 2) );

    #sigma_0=1.2;
    sigma_0 = 1.2;

    #[r,c,d] = size(Vspace);
    r, c, d = Vspace.shape;

    #n=[1:d]; %scale levels
    n = np.array(range(1, d + 1)); #%scale levels

    #sigma_I=sqrt(1.8).^n*sigma_0;
    sigma_I = math.sqrt(1.8)**n * sigma_0;

    w = sigma_I;

    #%w=w*0+1;
    #w=w/sum(w);
    w =  w / float(w.sum());

    if BOV_flag == 1:
        #w=fliplr(w);
        w = w[:, ::-1];

    #% Initialization
    #D = zeros(r+1, c+1);
    D = np.zeros( (r + 1, c + 1) );

    #D(1,:) = NaN;
    D[0, :] = np.nan;

    #D(:,1) = NaN;
    D[:, 0] = np.nan;

    #D(1,1) = 0;
    D[0, 0] = 0;

    #for i=1:d
    """
    We substitute i - 1 with i since arrays start with 0 in Python,
        not with 1 like in Matlab.
    """
    #for i in range(1, d + 1):
    for i in range(d):
        #V_temp=Vspace(:,:,i);
        V_temp = Vspace[:, :, i];

        #V_temp(isnan(V_temp))=0;
        V_temp[np.isnan(V_temp)] = 0; # !!!!TODO: check OK

        #Vspace(:,:,i)=V_temp;
        Vspace[:, :, i] = V_temp;

    #VV=zeros(r,c);
    VV = np.zeros( (r, c) );

    #for j=1:d
    """
    We substitute j - 1 with j since arrays start with 0 in Python,
        not with 1 like in Matlab.
    """
    #for j in range(1, d + 1):
    for j in range(d):
        #VV=VV+Vspace(:,:,j);
        VV = VV + Vspace[:, :, j];

    #D(2:end,2:end)=VV;
    D[1:, 1:] = VV;

    #NEW_DP3_ALEX = True;
    NEW_DP3_ALEX = False;

    if NEW_DP3_ALEX:
        # Alex: added cost
        cost = np.zeros( (r + 1, c + 1) );

    #% for traceback
    #Tback = zeros(r+1,c+1);
    Tback = np.zeros( (r + 1, c + 1) );

    if True:
        common.DebugPrint("dp3(): printing locally optimum solutions:");
        # Alex: trying out to find a better solution than dp3() !!!!TODO : more
        # This solution is basically the one returned by causal() IF we do NOT apply Matlab.filter2() on Vspace
        for j in range(1, c + 1):
            maxCol = 0.0;
            maxPos = -1;
            for i in range(1, r + 1):
                assert D[i, j] >= 0.0;
                if maxCol < D[i, j]:
                    maxCol = D[i, j];
                    maxPos = i; # So for query frame j we have a candidate matching ref frame i
                    common.DebugPrint("dp3():     for query frame %d - candidate frame %d" % (j - 1, maxPos));
            common.DebugPrint("dp3(): for query frame %d we found matching ref frame %d" % (j - 1, maxPos));


    # !!!!TODO: make i =0.., j=0.. and substitute i-1 with i, i-2 with i-1, etc
    #for i = 1:r;
    for i in range(1, r + 1):
        #for j = 1:c
        for j in range(1, c + 1):
            #if (i>1) && (j>1)
            if (i > 1) and (j > 1):
                #dd1 = w(1)*Vspace(i-1,max(1,j-2),1);
                dd1 = w[0] * Vspace[i - 2, max(0, j - 3), 0];

                #dd2 = w(1)*Vspace(max(1,i-2),j-1,1);
                dd2 = w[0] * Vspace[max(0, i - 3), j - 2, 0];

                #dd3 = w(1)*Vspace(i-1,j-1,1);
                dd3 = w[0] * Vspace[i - 2, j - 2, 0];

                #dd4 = w(1)*Vspace(i-1,j,1);
                dd4 = w[0] * Vspace[i - 2, j - 1, 0];

                #dd5 = w(1)*Vspace(i,j-1,1);
                dd5 = w[0] * Vspace[i - 1, j - 2, 0];

                if d > 1:
                    #for sc = 2:d
                    for sc in range(2, d + 1):
                        #dd1 = max(dd1, w(sc)*Vspace(i-1,max(1,j-2),sc));
                        dd1 = max(dd1, w[sc - 1] * \
                                        Vspace[i - 2, max(0, j - 3), sc - 1]);

                        #dd2 = max(dd2, w(sc)*Vspace(max(1,i-2),j-1,sc));
                        dd2 = max(dd2, w[sc - 1] * \
                                        Vspace[max(0, i - 3), j - 2, sc - 1]);

                        #dd3 = max(dd3, w(sc)*Vspace(i-1,j-1,sc));
                        dd3 = max(dd3, w[sc - 1] * Vspace[i - 2, j - 2, sc - 1]);

                        #dd4 = max(dd4, w(sc)*Vspace(i-1,j,sc));
                        dd4 = max(dd4, w[sc - 1] * Vspace[i - 2, j - 1, sc - 1]);

                        #dd5 = max(dd5, w(sc)*Vspace(i,j-1,sc));
                        dd5 = max(dd5, w[sc - 1] * Vspace[i - 1, j - 2, sc - 1]);

                #D(i,j-1)=D(i,j-1)+dd1;
                D[i - 1, j - 2] += dd1;

                #D(i-1,j)=D(i-1,j)+dd2;
                D[i - 2, j - 1] += dd2;

                #D(i,j)=D(i,j)+ dd3;
                D[i - 1, j - 1] += dd3;

                #D(i,j+1)=D(i,j+1)+ dd4;
                D[i - 1, j] += dd4;

                #D(i+1,j)=D(i+1,j)+ dd5;
                D[i, j - 1] += dd5;


                #%             % instead of above loop, use the following five lines when scales=6
                #%             D(i,j-1)=D(i,j-1)+max([w(1)*Vspace(i-1,max(1,j-2),1),w(2)*Vspace(i-1,max(1,j-2),2),w(3)*Vspace(i-1,max(1,j-2),3),w(4)*Vspace(i-1,max(1,j-2),4),w(5)*Vspace(i-1,max(1,j-2),5),w(6)*Vspace(i-1,max(1,j-2),6)]);
                #%             D(i-1,j)=D(i-1,j)+max([w(1)*Vspace(max(1,i-2),j-1,1),w(2)*Vspace(max(1,i-2),j-1,2),w(3)*Vspace(max(1,i-2),j-1,3),w(4)*Vspace(max(1,i-2),j-1,4),w(5)*Vspace(max(1,i-2),j-1,5),w(6)*Vspace(max(1,i-2),j-1,6)]);
                #%             D(i,j)=D(i,j)+max([w(1)*Vspace(i-1,j-1,1),w(2)*Vspace(i-1,j-1,2),w(3)*Vspace(i-1,j-1,3),w(4)*Vspace(i-1,j-1,4),w(5)*Vspace(i-1,j-1,5),w(6)*Vspace(i-1,j-1,6)]);
                #%             D(i,j+1)=D(i,j+1)+max([w(1)*Vspace(i-1,j,1),w(2)*Vspace(i-1,j,2),w(3)*Vspace(i-1,j,3),w(4)*Vspace(i-1,j,4),w(5)*Vspace(i-1,j,5),w(6)*Vspace(i-1,j,6)]);
                #%             D(i+1,j)=D(i+1,j)+max([w(1)*Vspace(i,j-1,1),w(2)*Vspace(i,j-1,2),w(3)*Vspace(i,j-1,3),w(4)*Vspace(i,j-1,4),w(5)*Vspace(i,j-1,5),w(6)*Vspace(i,j-1,6)]);


                #% [dmax, tb] = max([D(i, j), D(i-1, j), D(i, j-1), D(i,j+1),D(i+1,j)]);
                #[dmax, tb] = max([D(i, j)+1/sqrt(2), D(i-1, j)+1/sqrt(5), D(i, j-1)+1/sqrt(5), D(i,j+1)+1, D(i+1,j)+1]);
                dmax, tb = Matlab.max(np.array([ \
                                D[i - 1, j - 1] + 1.0 / math.sqrt(2.0), \
                                D[i - 2, j - 1] + 1.0 / math.sqrt(5.0), \
                                D[i - 1, j - 2] + 1.0 / math.sqrt(5.0), \
                                D[i - 1, j] + 1,
                                D[i, j - 1] + 1]));
            else:
                #[dmax, tb] = max([D(i,j), D(i,j+1), D(i+1,j)]);
                dmax, tb = Matlab.max( \
                        np.array([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]));
                common.DebugPrint("dp3(): dmax = %s" % str(dmax));
                common.DebugPrint("dp3(): tb = %s" % str(tb));

            #D(i+1,j+1) = D(i+1,j+1)+dmax;
            if NEW_DP3_ALEX:
                cost[i, j] = 0; #!!!!TODO: think more
            else:
                D[i, j] += dmax; #!!!!TODO: for me it's weird he adds dmax here...

            #Tback(i+1,j+1) = tb;
            Tback[i, j] = tb;

    common.DebugPrint("dp3(): D.shape = %s" % str(D.shape));
    common.DebugPrint("dp3(): D = %s" % str(D));

    common.DebugPrint("dp3(): Tback.shape = %s" % str(Tback.shape));
    common.DebugPrint("dp3(): Tback = %s" % str(Tback));

    #% Traceback
    i = r + 1;
    j = c + 1;
    y = i - 1;
    x = j - 1;

    #while i > 2 & j > 2
    while (i > 2) and (j > 2):
        #tb = Tback(i,j);
        tb = Tback[i - 1, j - 1] + 1; # In Matlab, max returns indices from 1..

        if tb == 1:
            i -= 1;
            j -= 1;
        elif tb == 2:
            i -= 2;
            j -= 1;
        elif tb == 3:
            i -= 1;
            j -= 2;
        elif tb == 4:
            i -= 1;
            j = j;
        elif tb == 5:
            j -= 1;
            i = i;
        else:
            #error;
            assert False;

        #y = [i,y];
        #NOT GOOD: y = np.c_[i, y];
        y = np.hstack([i - 1, y]);

        #x = [j,x];
        #NOT GOOD: x = np.c_[j, x];
        x = np.hstack([j - 1, x]);

    common.DebugPrint("dp3(): before D.shape = %s" % str(D.shape));
    common.DebugPrint("dp3(): before D = %s" % str(D));

    #% Strip off the edges of the D matrix before returning
    #D = D(2:(r+1),2:(c+1));
    D = D[1: (r + 1), 1: (c + 1)];

    common.DebugPrint("dp3(): D.shape = %s" % str(D.shape));
    common.DebugPrint("dp3(): D = %s" % str(D));

    #RD_start=str2num(RD(1).name(end-9:end-4));
    RD_start = 1;

    common.DebugPrint("dp3(): Vspace.shape = %s" % str(Vspace.shape));

    #!!!!TODO: understand well what is x,y and why computes p
    #for i=1:size(Vspace,2):
    for i in range(0, Vspace.shape[1]):
        #cross(i,1)=str2num(QD(i).name(end-9:end-4));
        crossref[i, 0] = i; #i; #!!!!TODO: think if OK

        #p=find(x==i);
        p = np.nonzero(x == i);
        p = p[0];

        common.DebugPrint("dp3(): x.shape = %s" % str(x.shape));
        common.DebugPrint("dp3(): x = %s" % str(x));
        common.DebugPrint("dp3(): y.shape = %s" % str(y.shape));
        common.DebugPrint("dp3(): y = %s" % str(y));
        common.DebugPrint("dp3(): i = %s" % str(i));
        common.DebugPrint("dp3(): p = %s" % str(p));

        #if isempty(p)
        if p.size == 0:
            #% Alex: Vali Codreanu said to change from temp=0; to temp=3;
            #temp=3;
            temp = 0;

            common.DebugPrint("dp3(): temp = %s" % str(temp));

            crossref[i, 1] = 0 + RD_start - 1;
        else:
            #temp=y(p);
            temp = y[p];

            common.DebugPrint("dp3(): temp = %s" % str(temp));

            #cross(i,2)=temp(end)+RD_start-1;
            if temp.size == 1:
                # If temp has only 1 element:
                crossref[i, 1] = temp + RD_start - 1;
            else:
                crossref[i, 1] = temp[-1] + RD_start - 1;
            #assert temp.size == 1;

        #common.DebugPrint("dp3(): temp = %s" % str(temp));

        """
        #cross(i,2)=temp(end)+RD_start-1;
        if True:
            crossref[i, 1] = temp[-1] + RD_start - 1;
        else:
            # BAD IDEA!!!!TODO
            crossref[i, 1] = temp + RD_start - 1;
        """

    #printf("dp3(): Done in blabla secs\n");
    common.DebugPrint("dp3(): crossref = %s" % str(crossref));

    ComputeCost(crossref, VV, "crossref_dp3.txt");

    #return [y,x,D,Tback,cross]
    return y, x, D, Tback, crossref;



def dp_Alex(Vspace, numFramesR, numFramesQ, BOV_flag, PREV_REF=5, NEXT_REF=0):
    #PREV_REF = 1000; #20; #0; #3; #2
    #NEXT_REF = -1; #3; #2

    """
    Vspace is a matrix with shape (numFramesR, numFramesQ).
    See multiscale_quad_retrieval.py for definition:
        Votes_space = np.zeros( (len(RD), len(QD)) );
    """

    t1 = float(cv2.getTickCount());

    common.DebugPrint("Entered dp_Alex(): Running dynamic programming...");

    #causal(Vspace, H=None, numFramesQ, numFramesR, BOV_flag, crop_flag, const_type, nop=0);
    #causal(Vspace, None, numFramesQ, numFramesR, BOV_flag, 0, 1, 0);

    if True:
        r, c, d = Vspace.shape;

        # We substitute all NaN's of Vspace
        for i in range(d):
            V_temp = Vspace[:, :, i];
            V_temp[np.isnan(V_temp)] = 0;
            Vspace[:, :, i] = V_temp;

        VV = np.zeros( (r, c) );
        for j in range(d):
            VV += Vspace[:, :, j];

        # Checking that VV has positive elements
        #print("VV < 0.0 = %s" % str(np.nonzero(VV < 0.0)));
        assert np.nonzero(VV < 0.0)[0].size == 0;
        #assert np.nonzero(VV < 0.0)[1].size == 0;
    else:
        # For testing purposes only
        numFramesR = 10;
        numFramesQ = 10;
        VV = np.zeros((numFramesR, numFramesQ));
        r, c = VV.shape;

        VV[0, 0] = 10;
        VV[3, 0] = 10;
        VV[5, 3] = 20;

    PRINT_MATRICES = True;
    #PRINT_MATRICES = False;

    #print("VV[:100, :100] = %s" % str(VV[:100, :100]));
    if PRINT_MATRICES:
        print("dp_Alex(): r = %d, c = %d" % (r, c));
        print("dp_Alex(): VV = \n%s" % str(VV));
        sys.stdout.flush();

    D = np.zeros( (r, c) );
    Tback = np.zeros( (r, c) );

    for ref in range(r):
        D[ref, 0] = VV[ref, 0];
        Tback[ref, 0] = -1;

    #PREV_REF = 1000; #20; #0; #3; #2
    #NEXT_REF = -1; #3; #2

    for qry in range(1, c):
        for ref in range(r):
            # We enumerate a few reference frames to find the one with highest votes
            lb = ref - PREV_REF;
            ub = ref + NEXT_REF;

            if lb < 0:
                lb = 0;
            if lb >= r:
                lb = r - 1;

            if ub < 0:
                ub = 1; #0;
            if ub >= r:
                ub = r - 1;

            #print("lb=%d, ub=%d" % (lb, ub));

            maxPos = lb;
            for i in range(lb + 1, ub + 1):
                """
                We use <= --> we break ties by going forward in the
                    reference video (incrementing the reference frame for
                    the next query frame).
                """
                if D[maxPos, qry - 1] <= D[i, qry - 1]:
                    maxPos = i;
            # maxPos is the maximum vote reference frame for query frame qry

            #print("qry=%d, ref=%d: maxPos = %d" % (qry, ref, maxPos));

            D[ref, qry] += D[maxPos, qry - 1] + VV[ref, qry];
            Tback[ref, qry] = maxPos;

            """
            if (qry > 1) and (qry < c - 1):

            elif (qry == 0):

            elif (qry == c - 1):
            """

    if PRINT_MATRICES:
        print("D = \n%s" % str(D));
        print("Tback = \n%s" % str(Tback));

    crossref = np.zeros( (numFramesQ, 2) );

    # Find max-cost path (the critical path) for the last query frame:
    maxPos = 0;
    for ref in range(1, r):
        """
        We use <= --> we break ties by going forward in the
            reference video (incrementing the reference frame for
            the next query frame) - debatable if this is a good idea!!!!TODO.
        """
        if D[maxPos, c - 1] <= D[ref, c - 1]:
            maxPos = ref;
    print("maxPos = %d" % maxPos);

    print("dp_Alex(): cost critical path = %s" % str(D[maxPos, c - 1]));

    posRef = maxPos;
    for qry in range(c - 1, 0-1, -1):
        crossref[qry][0] = qry;
        crossref[qry][1] = posRef;

        print("qry=%d, posRef=%d" % (qry, posRef));
        posRef = Tback[posRef, qry];

    #!!!!time took
    #common.DebugPrint("dp_Alex(): crossref = %s" % str(crossref));
    print("dp_Alex(): crossref = %s" % str(crossref));

    #ComputeCost(crossref, VV);
    ComputeCost(crossref, VV, "crossref_dp_Alex.txt");
    #!!!!TODO: assert cost computed is = D[maxPos,...]

    t2 = float(cv2.getTickCount());
    myTime = (t2 - t1) / cv2.getTickFrequency();
    #common.DebugPrint("dp_Alex() took %.6f [sec]" % myTime);
    print("dp_Alex() took %.6f [sec]" % myTime);

    #if True:
    if False:
        #print("Am here1111");
        causal_Alex(Vspace, numFramesQ, numFramesR);

        #print("Am here2222");
        dp3Orig(Vspace, numFramesR, numFramesQ, BOV_flag);

        #def causal(Vspace, H, numFramesQ, numFramesR, BOV_flag, cropflag, const_type, nop=0):
        H = np.zeros( (10000, 10000, 5), dtype=np.int8 );
        causal(Vspace, H, numFramesQ, numFramesR, BOV_flag=0, cropflag=0, const_type=1, nop=0);

    y = None;
    x = None;
    #quit();
    return y, x, D, Tback, crossref;


dp3Orig = dp3;
dp3 = dp_Alex;


#dp3(None, 0, 0, 0);
#quit();


import unittest

#print "Am here"

class TestSuite(unittest.TestCase):
    #def testRansac_line(self):


    def testCausal(self):
        # This is a test case from the videos from Evangelidis, with frameStep = 200
        #Vspace = np.zeros( (92, 31, 5) );
        Vspace = np.zeros( (12, 4, 5) );
        Vspace[:, :, 0] = np.array([ [0,        0,        0,        0],
                                     [0,        0,   1.1646,        0],
                                     [0,   1.1646,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0],
                                     [0,        0,        0,        0]]);
        # The rest of Vspace is completely 0 :)

        #H = np.ones( (92, 31, 5) );
        H = np.ones( (12, 4, 5) );

        numFramesQ = 4;
        numFramesR = 12;
        BOV_flag = 0;
        cropflag = 0;
        const_type = 1;

        res = causal(Vspace, H, numFramesQ, numFramesR, BOV_flag, \
                                    cropflag, const_type);
        print "testCausal(): res from causal() = %s" % str(res);

        resGood = np.array([[ 0, 2],
                            [ 1, 2],
                            [ 2, 1],
                            [ 3, 1]]);

        #assert res == resGood # Complains about it

        aZero = res - resGood;
        #print np.nonzero(res - resGood);
        #assert (aZero == 0).all();

        self.assertTrue((aZero == 0).all());

        res = dp3(Vspace, numFramesR, numFramesQ, BOV_flag);
        #TODO: test result of dp3()

if __name__ == '__main__':
    # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(threshold=1000000, linewidth=5000);

    unittest.main();

