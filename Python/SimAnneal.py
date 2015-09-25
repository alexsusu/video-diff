# From https://stackoverflow.com/questions/19757551/basics-of-simulated-annealing-in-python

import random
import math

#import ReadVideo
import MatchFrames

import cv2
import common
import config


LIMIT = 100000




capture2 = None
#lenA = None


# The ReadVideo module is responsible to reset this dictionary for a new processing
Acache = {}

"""
Unfortunately can't (easily?) create in Python a lazy list that can be
    accessed randomly (and sequentially) - generator (with yield) allows
    accessing elements sequentially
  So, we define function GetA(aCounter2) instead of accessing A[aCounter2], etc
"""
def GetA(aCounter2):
    global capture2
    # We get the frame at each access and convert it to RGB - we don't store the result in memory

    # We memoize the value corresponding to aCounter2 in Acache[aCounter2], in the idea that Simulated Annealing might come back to this value
    if aCounter2 in Acache:
        return Acache[aCounter2]

    # If we try to seek to a frame out-of-bounds frame it gets to the last one
    capture2.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, aCounter2)

    frame2 = capture2.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    if False:
        print "Alex: frame2 = %d" % frame2

    #aCounter2 = int(frame2) #0
    assert aCounter2 == int(frame2) #0

    ret2, img2 = capture2.read()
    if (ret2 == False) or (GetALength() <= aCounter2):
        common.DebugPrint(
            "Get(): REALLY BAD error: aCounter2 = %d, ret2 = %s" % \
            (aCounter2, str(ret2)))
        quit()
        #break

    if False:
        # Note: img1 and img2 are RGB-images. I need to convert them to grayscale (as cv2.imread(fn1, 0) )
        gray2 = ConvertImgToGrayscale(img2)

    res = MatchFrames.Main_img2(img2, aCounter2)
    #print "GetA(): res = %s" % (str(res))

    # We memoize this result - The ReadVideo module is responsible to reset this dictionary for a new processing
    Acache[aCounter2] = -res[1]

    # We return the number of matches
    return -res[1]



def GetALength():
    """
    global frameCount2
    print "frameCount2 = %s" % str(frameCount2)
    return frameCount2
    """
    return LIMIT







def update_temperature(T, k):
    return T - 0.001

def get_neighbors(i, L):
    assert L > 1 and i >= 0 and i < L
    if i == 0:
        return [1]
    elif i == L - 1:
        return [L - 2]
    else:
        return [i - 1, i + 1]

def make_move(x, T):
    # nhbs = get_neighbors(x, len(A))
    # nhb = nhbs[random.choice(range(0, len(nhbs)))]
    nhb = random.choice(xrange(0, GetALength())) # choose from all points

    delta = GetA(nhb) - GetA(x)

    if delta < 0:
        return nhb
    else:
        p = math.exp(-delta / T)
        return nhb if random.random() < p else x


def simulated_annealing():
    L = GetALength()

    common.DebugPrint("simulated_annealing(): L = %s" % str(L))

    #L = len(A)
    x0 = random.choice(xrange(0, L))
    T = 1.
    k = 1

    x = x0
    x_best = x0

    # Alex:
    #while T > 1e-3:
    #while T > 1e-1: # 901 iterations
    #while T > 1e-2: # 991 iterations
    #while T > 5e-1: # 501 iterations
    #while T > 6e-1: # 401 iterations
    #while T > 7e-1: # 301 iterations
    #while T > 8e-1: # 201 iterations

    #while T > 9.5e-1: # 51 iterations - suboptimal (401, 228) instead of (400, 377)
    #while T > 9.7e-1: # 31 iterations - suboptimal (402, 198) instead of (400, 377)

    """
    In a run obtained suboptimal (411, 78) instead of (400, 377), worse than 51 iterations
    In another run obtained optimal (400, 377) result.
    For our example from test.py this one converges nice 50% of the times.
    """
    while T > 9e-1: # 101 iterations:
        x = make_move(x, T)
        if GetA(x) < GetA(x_best):
            x_best = x
        T = update_temperature(T, k)
        k += 1

    common.DebugPrint("iterations: %d" % k)
    return x, x_best, x0


def isminima_local(p, A):
    return all(A[p] < A[i] for i in get_neighbors(p, len(A)))

def func(x):
    return math.sin((2 * math.pi / LIMIT) * x) + 0.001 * random.random()

def initialize(L):
    return map(func, xrange(0, L))



def main():
    global A

    # This is for comparison reasons only
    #if True:
    if False:
        # Computing local minima
        local_minima = []
        for i in xrange(0, LIMIT):
            if(isminima_local(i, A)):
                local_minima.append([i, A[i]])

        # Computing global minumum through linear search
        x = 0
        y = A[x]
        for xi, yi in enumerate(A):
            if yi < y:
                x = xi
                y = yi
        global_minumum = x

        print "number of local minima: %d" % (len(local_minima))
        print "global minimum @%d = %0.3f" % (global_minumum, A[global_minumum])

    x, x_best, x0 = simulated_annealing()
    print "Solution is @%d = %0.3f" % (x, GetA(x))
    print "Best solution is @%d = %0.3f" % (x_best, GetA(x_best))
    print "Start solution is @%d = %0.3f" % (x0, GetA(x0))

    return (x_best, GetA(x_best))



if __name__ == '__main__':
    A = initialize(LIMIT)
    main()
