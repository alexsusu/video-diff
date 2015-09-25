import numpy as np

import common
import synchro_script
import SpatialAlignment



def AlignVideos(captureQ, captureR):
    print("Entered VideoAlignmentEvangelidis.AlignVideos().");

    TESTING = False

    if TESTING:
        crossref = [];
        for i in range(numFramesQ):
            crossref.append([i, i]);
            #crossref
        crossref = np.array(crossref);
    else:
        crossref = synchro_script.TemporalAlignment(captureQ, captureR);

    print("VideoAlignmentEvangelidis.AlignVideos(): crossref = %s" % str(crossref));

    SpatialAlignment.SpatialAlignmentEvangelidis(crossref, captureQ, captureR);

