import numpy as np

import common


#function ordvot=spatial_consistency(space_xy,qcen, M, st_threshold, cropflag)
def spatial_consistency(space_xy, qcen, M, st_threshold, cropflag):
    ordvot = np.zeros((M, 1))

    for iii in range(1, M + 1):
        #uu=space_xy(:,(iii-1)*2+1:(iii-1)*2+2);
        uu = space_xy[:, (iii-1) * 2 : (iii-1) * 2 + 1];

        #pn=~isnan(uu(:,1));
        pn = np.logical_not(np.isnan(uu[:, 0]));

        if isempty(pn):
            continue

        #vv=(uu(pn,:)-qcen(pn,:));
        vv = (uu[pn, :] - qcen[pn, :]);

        if cropflag == 1:
            #inliers=(sign(vv(:,2))==-1)&(abs(vv(:,1))<10);
            inliers = (sign(vv[:, 1]) == -1) and (abs(vv[:, 0]) < 10);

            #outliers = (sign(vv(:,2))==1)|(abs(vv(:,1))>10)
            outliers = (sign(vv[:, 1]) == 1) or (abs(vv[:, 0]) > 10);

            bifla = sum(inliers); #/(sum(outliers)+sum(inliers))

            if isnan(bifla):
                ordvot[iii - 1] = 0;
            elif isinf(bifla):
                ordvot[iii - 1] = inliers.sum();
            else:
                ordvot[iii - 1] = bifla;
        else:
            #inliers=sqrt(sum(vv.^2,2))<st_threshold;
            inliers = sqrt((vv**2).sum(1)) < st_threshold;

            #outliers = sqrt(sum(vv.^2,2)) > st_threshold
            outliers = sqrt((vv**2).sum(1)) > st_threshold;

            bifla = inliers.sum() / (outliers.sum() + inliers.sum());

            if isnan(bifla):
                ordvot[iii - 1] = 0;
            elif isinf(bifla):
                ordvot[iii - 1] = inliers.sum();
            else:
                ordvot[iii - 1] = bifla;

    if common.MY_DEBUG_STDOUT:
        common.DebugPrint("ordvot = %s" % str(ordvot));

    return ordvot

