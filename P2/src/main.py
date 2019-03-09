import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from math import exp,ceil
import random

# In:
#   filename: File path
#   flagcolor: cv2 color flag [0 Grayscale, 1 Color]
# Out:
#   Image matrix (H,W,C[BGR])
def leeimagen(filename, is_float = True):
    if is_float:
        #M = plt.imread(filename).astype(np.float)
        M = cv2.imread(filename,1)
        return M[..., ::-1]/255.
    else:
        return cv2.imread(filename,1)

# In:
#   im: Image matrix (H,W,C[BGR])
#   window_name: Window's title
#   wait: call cv2.waitKey()
def pintaI(im,window_name="window",wait=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name,cv2.pyrUp(im))
    if wait:
        cv2.waitKey(0)

# In:
#   vim: Image matrix (H,W,C[BGR]) iterable [list, tuple, np.array...]
def pintaMI(vim):
    for i,im in enumerate(vim):
        pintaI(im, window_name= "Image{}".format(i+1), wait=False)
    cv2.waitKey(0)

# In:
#   vim: Image matrix (H,W,C[BGR]) iterable [list, tuple, np.array...]
#   labels: [String] List of labels
#   cols: Number of columns
def pintaMISingleMPL(vim,labels=None,title="",cols=3):
    # Transform every image to RGB
    # vim2 = np.array([
    #     cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    #     if len(im.shape) == 2
    #     else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     for im in vim])
    res = []
    for im in vim:
        if im.dtype != np.uint8:
            im = (im*255).clip(0,255).astype(np.uint8)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        res.append(im)
    vim = np.array(res)

    # Calculate ncols and nrows
    import math
    cols = min(cols,len(vim))
    rows = math.ceil(len(vim)/cols)

    plt.suptitle(title)
    # Add subplot for image in vim
    for i,im in enumerate(vim):
        plt.subplot(rows,cols,i+1)
        plt.imshow(im)
        # Remove ticks
        plt.xticks([])
        plt.yticks([])
        # Add label if needed
        if labels is not None:
            plt.title(labels[i])

    plt.show()


# Código para Ejercicio 1.A

def applyGaussianBlur(img,ksize,sigma):
    if type(ksize) is not tuple:
        ksize = (ksize,ksize)
    return cv2.GaussianBlur(img,ksize,sigma)

def applyGaussianBlurKernel(img,ksize,sigma):
    k = cv2.getGaussianKernel(ksize,sigma)
    out = cv2.sepFilter2D(img,-1,k,k)
    return out

# Código para Ejercicio 1.B

def applyDeriv(img,ksize,der=1):
    # Obtenemos las dos componentes separadas (horizontal,
    # y vertical) y las aplicamos

    # Definimos normalize=True porque aplicamos la
    # derivada a imágenes en coma flotante. Referencia:
    # https://docs.opencv.org/3.4.3/d4/d86/group__imgproc__filter.html#ga6d6c23f7bd3f5836c31cfae994fc4aea
    kx,ky = cv2.getDerivKernels(der,der,ksize,normalize=True)

    # el segundo valor (ddepth) lo definimos como -1
    # para que OpenCV devuelve la imagen en el mismo
    # tipo de dato que la mandamos. Referencia:
    # https://docs.opencv.org/3.4.3/d4/d86/group__imgproc__filter.html#ga910e29ff7d7b105057d1625a4bf6318d
    out = cv2.sepFilter2D(img,-1,kx,ky)
    return out

def getMirrorValue(i,max,border):
    i = i-border
    if i < 0:
        i = (i)*-1
    elif i >= max:
        i = 2*max - i - 2
    return i

def getMirrorBorder(M,border):
    origHeight = M.shape[0]
    origWidth = M.shape[1]
    M2 = np.zeros((origHeight+2*border,origWidth+2*border))
    #Center Image
    #M2[border:border+origWidth,border:border+origHeight] = M
    for r in range(origHeight+2*border):
        for c in range(origWidth+2*border):
            orir = getMirrorValue(r,origHeight,border)
            oric = getMirrorValue(c,origWidth,border)
            M2[r,c] = M[orir,oric]
    return M2

def getConstantBorder(M,border,val):
    origHeight = M.shape[0]
    origWidth = M.shape[1]
    M2 = np.full((origHeight + 2 * border, origWidth + 2 * border),val, dtype=M.dtype)
    M2[border:origHeight+border,border:origWidth+border] = M
    return M2

BORDER_MIRROR = cv2.BORDER_REFLECT_101
BORDER_CONSTANT = cv2.BORDER_CONSTANT
def applySepConv(Mfull,k,border,border_val=0.):
    kx,ky = k
    assert ((kx.size%2 == 1) and (ky.size%2 == 1))

    #Get border image
    borderWidth = kx.size // 2

    res = []
    for M in cv2.split(Mfull):
        if border == BORDER_MIRROR:
            M2 = getMirrorBorder(M,borderWidth)
        else:
            M2 = getConstantBorder(M,borderWidth,border_val)

        #Apply kx
        M3 = np.zeros((M2.shape[0],M.shape[1]))
        for c in range(M.shape[1]):
            M3[:,c] = np.matmul(M2[:,c:c+2*borderWidth+1],kx.reshape((-1,1))).reshape((-1))

        M4 = np.zeros(M.shape)
        for r in range(M.shape[0]):
            M4[r,:] = np.matmul(ky.reshape((1,-1)),M3[r:r+2*borderWidth+1,:]).reshape((-1))
        res.append(M4)
    return cv2.merge(res)





def visualizeM(img):
    min = np.min(img)
    max = np.max(img)
    img = (img-min)/(max-min)*255
    return img.astype(np.uint8)

def halfProgression(x,n):
    total = 0
    for i in range(n):
        total += x
        x = ((x+1)//2)
    return int(total)



def gaussianPyramid(img,n,borderType=cv2.BORDER_DEFAULT,borderConstant=0.):
    M = np.zeros((halfProgression(img.shape[0],n),img.shape[1]))
    beg = 0
    for i in range(n):
        M[beg:beg+img.shape[0],:img.shape[1]] = img
        beg += img.shape[0]
        if i != n-1:
            if (borderType == cv2.BORDER_CONSTANT):
                img = getConstantBorder(img, 2, borderConstant)
                img = cv2.pyrDown(img,borderType=cv2.BORDER_DEFAULT)
                img = img[1:-1,1:-1]
            else:
                img = cv2.pyrDown(img,borderType=borderType)
    return M



# def laplacianPyramid(img,n,borderType=cv2.BORDER_DEFAULT,borderConstant=0.):
#     M = np.zeros((halfProgression(img.shape[0], n+1), img.shape[1]*2))
#     offset = img.shape[1]
#     beg = 0
#     difs = []
#     for i in range(n+1):
#         M[beg:beg + img.shape[0], :img.shape[1]] = img
#         if (i != n):
#             beg += img.shape[0]
#             if (borderType == cv2.BORDER_CONSTANT):
#                 img2 = getConstantBorder(img,2,borderConstant)
#                 img2 = cv2.pyrDown(img2,borderType=cv2.BORDER_DEFAULT)
#                 img2 = img2[1:-1,1:-1]
#             else:
#                 img2 = cv2.pyrDown(img,borderType=borderType)
#             difs.append(img-cv2.pyrUp(img2,dstsize=(img.shape[1],img.shape[0])))
#             img = img2
#
#     beg = 0
#     for i in range(n+1):
#         M[beg:beg+img.shape[0],offset:offset+img.shape[1]] = img
#         if (i != n):
#             beg += img.shape[0]
#             dif = difs.pop()
#             img2 = cv2.pyrUp(img,dstsize=(dif.shape[1],dif.shape[0]))
#             img2 += dif
#             img = img2
#     # for im in difs:
#     #     M[beg:beg + im.shape[0], :im.shape[1]] = im
#     #     beg += im.shape[0]
#     return M


def getAmp(img):
    return np.max(img) - np.min(img)

def getSweetSpot(img,val=0.75):
    amp = getAmp(img)
    sigma = 1
    imgb = cv2.GaussianBlur(img, (0,0),sigma)
    while(getAmp(imgb)/amp > val):
        sigma += 1
        imgb = cv2.GaussianBlur(img, (0,0), sigma)
    return imgb, sigma


def highFreq(img,sigma):
    ksize = ocvKsize(sigma)
    kx = cv2.getGaussianKernel(ksize, sigma)
    img2 = cv2.sepFilter2D(img,-1,kx,kx)
    img = img - img2
    return img

def cLowFreq(img,ksize,s):
    k = gaussianMaskV(s,2)
    return applySepConv(img,(k,k),cv2.BORDER_DEFAULT)

def cHighFreq(img,sigma):
    k = gaussianMaskV(sigma,2)
    img2 = applySepConv(img,(k,k),cv2.BORDER_DEFAULT)
    img = img - img2
    return img


def hybridImg(img1,img2,s1,s2,lF=cv2.GaussianBlur,hF=highFreq):
    #bici-moto: 5-5
    #plane-bird: 5-11
    #cat-dog: 9-25
    #mari-eins: 7-7
    #fish-subma:7-15

    imgl = lF(img1,(0,0),s1)
    imgh = hF(img2,s2)
    imgf = np.clip(imgl + imgh,0,1)

    M = customGaussianPyramid(imgf,5)
    if len(img1.shape) == 3:
        M2 = np.zeros((img1.shape[0] * 2, img1.shape[1] * 2,3), dtype=np.uint8)
    else:
        M2 = np.zeros((img1.shape[0] * 2, img1.shape[1] * 2), dtype=np.uint8)
    M2[:M.shape[0],:M.shape[1]] = visualizeM(M)

    M2[:imgh.shape[0], img1.shape[1]:img1.shape[1] + imgh.shape[1]] = visualizeM(imgh)
    M2[imgh.shape[0]:imgh.shape[0]+imgl.shape[0], img1.shape[1]:img1.shape[1] + imgl.shape[1]] = visualizeM(imgl)

    return M2

def gauss(x,sigma):
    return exp(-0.5*((x**2)/(sigma**2)))

#from OpenCV sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
def ocvKsize(sigma):
    ksize = ceil((2 * sigma - 1) / 0.3)  # (~99%)
    ksize += (ksize + 1) % 2
    return ksize

def gaussianMaskV(sigma,formulae=0):
    #Option 1: first odd after 3*sigma
    if formulae == 0:
        ksize = ceil(3*sigma)
        ksize += (ksize + 1)%2 #(~86%)

    #Option 2: 2*(2*sigma)+1
    elif formulae == 1:
        ksize = 4*sigma+1 #(~95%)

    #Option 3: from OpenCV sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    else:
        ksize = ocvKsize(sigma)

    v = np.arange(-(ksize//2),ksize//2+1,1)
    v = np.vectorize(gauss)(v,sigma)
    v = v/np.sum(v)
    return v

def customGaussianPyramid(img,n,borderType=BORDER_MIRROR,borderConstant=0.):
    beg = 0
    k = gaussianMaskV(1.1)
    kx,ky=k,k

    is3C = len(img.shape) == 3
    assert (not is3C or img.shape[2] == 3)

    if is3C:
        M = np.zeros((halfProgression(img.shape[0], n), img.shape[1],3))
    else:
        M = np.zeros((halfProgression(img.shape[0],n),img.shape[1]))

    for i in range(n):
        M[beg:beg+img.shape[0],:img.shape[1]] = img
        beg += img.shape[0]
        if i != n-1:
            img = applySepConv(img,(kx,ky),borderType,borderConstant)
            img = img[::2,::2]
    return M

def getPyrGaussianKenel():
    return np.array([1.,4.,6.,4.,1.])/16.
    #return np.array([0.067593,	0.244702,	0.37541,	0.244702,	0.067593])

def cpyrDown(imgFull,borderType=cv2.BORDER_DEFAULT,borderConstant=0.):

    assert borderType in [cv2.BORDER_DEFAULT, cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT]

    res = []
    for img in cv2.split(imgFull):
        #k = gaussianMaskV(1,2)
        k = getPyrGaussianKenel()
        img = applySepConv(img,(k,k),borderType,borderConstant)
        img = img[::2,::2]
        res.append(img)

    return cv2.merge(res)

def cpyrUp(imgFull,dstsize=None, borderType=cv2.BORDER_DEFAULT, borderConstant=0.):
    if dstsize is None:
        dstsize = tuple(np.array(imgFull.shape[:2])*2)
    else:
        dstsize = (dstsize[1],dstsize[0])
    assert ((imgFull.shape[0]*2) % dstsize[0] < 2) and ((imgFull.shape[1]*2) % dstsize[1] < 2)
    res = []
    for img in cv2.split(imgFull):
        dst = np.zeros(dstsize)
        dst[::2,::2] = img
        #k = 2*gaussianMaskV(1.08, 2)
        k = 2*getPyrGaussianKenel()
        dst = applySepConv(dst, (k, k), borderType, borderConstant)

        res.append(dst)
    return cv2.merge(res)




def laplacianPyramid(imgFull,n,borderType=cv2.BORDER_DEFAULT,borderConstant=0.,pyrDown=cv2.pyrDown, pyrUp = cv2.pyrUp,):
    res = []
    for img in cv2.split(imgFull):
        M = np.zeros((halfProgression(img.shape[0], n+1), img.shape[1]*2))
        offset = img.shape[1]
        beg = 0
        difs = []
        for i in range(n+1):
            M[beg:beg + img.shape[0], :img.shape[1]] = img
            if (i != n):
                beg += img.shape[0]
                if (borderType == cv2.BORDER_CONSTANT):
                    img2 = getConstantBorder(img,2,borderConstant)
                    img2 = pyrDown(img2,borderType=cv2.BORDER_DEFAULT)
                    img2 = img2[1:-1,1:-1]
                else:
                    img2 = pyrDown(img,borderType=borderType)
                difs.append(img-pyrUp(img2,dstsize=(img.shape[1],img.shape[0])))
                img = img2

        beg = 0
        for i in range(n+1):
            M[beg:beg+img.shape[0],offset:offset+img.shape[1]] = img
            if (i != n):
                beg += img.shape[0]
                dif = difs.pop()
                img2 = pyrUp(img,dstsize=(dif.shape[1],dif.shape[0]))
                img2 += dif
                img = img2
        # for im in difs:
        #     M[beg:beg + im.shape[0], :im.shape[1]] = im
        #     beg += im.shape[0]
        res.append(M)
    return cv2.merge(res)

def pintaSepKernel(kx,ky=None):
    if ky is None:
        ky = kx.reshape((1,-1))
    kx = kx.reshape((-1,1))
    pintaI(visualizeM(kx*ky))

def mirrorVector(v,b):
    res = np.zeros(v.size+2*b)
    for i in range(res.size):
        p = i - b
        if p >= v.size:
            p = v.size - (p+2)
        elif p < 0:
            p = np.abs(p)
        res[i] = v[p]
    return res

def conv1D(v,k):
    assert (k.size % 2 == 1)
    mv = mirrorVector(v, k.size//2)
    ksize = k.size
    res = np.zeros(v.size)
    k = k.reshape((-1,1))
    for i in range(v.size):
        res[i] = np.matmul(mv[i:i+ksize].reshape((1,-1)),k)
    return res

def conv2D(mfull,k):
    res = []
    kx,ky = k
    for m in cv2.split(mfull):
        mt = np.zeros(m.shape)
        for i,row in enumerate(m):
            mt[i] = conv1D(row,kx)
        for i,col in enumerate(np.transpose(mt)):
            mt[:,i] = conv1D(col,ky)
        res.append(mt)
    return cv2.merge(res)


def wait():
    input('Press Enter:')

######################################
#
# Práctica 2 Code
#
######################################

import copy

#Generate n leves pyramid, the first level
# is the scaled up img(2x size)
def getGaussianPyramidList(img, levels = 4):
    assert(levels > 2)

    # res = [cv2.pyrUp(img),img]
    # for i in range(levels-2):
    #     res.append(cv2.pyrDown(res[i+1]))
    res = [img]
    for i in range(levels-1):
        res.append(cv2.pyrDown(res[i]))
    return res

def getCustomSIFT(img,octaves=4,blurs=5,k=np.sqrt(2)):

    #Blur values
    bvals = np.zeros(blurs)
    #bvals[0] = 1/k
    bvals[0] = 1
    for j in range(blurs-1):
        bvals[j+1] = bvals[j]*k

    #Octave-Scale Matrix
    gPyr = getGaussianPyramidList(img, levels=octaves)
    m = []
    for im in gPyr:
        l = []
        for s in bvals:
            l.append(cv2.GaussianBlur(im,(0,0),s))
        m.append(l)

    pintaI(m[1][4])

    #LoG (Difference of Gaussian)
    mL = []
    for i in range(len(m)):
        r = []
        for j in range(len(m[i])-1):
            r.append(m[i][j]-m[i][j+1])
        mL.append(r)
    pintaI(visualizeM(mL[2][2]))

    #Local Maxima/Minima
    mM = []
    #for scale
    for i in range(len(mL)):
        l = []
        #for blur
        for j in range(len(mL[i])-2):
            l.append(getMinMaxMap(mL[i][j-1], mL[i][j], mL[i][j+1]))
        mM.append(l)

    #Taylor filter
    #...
    pintaMI((mM[0][0].astype(np.float), visualizeM(mL[0][1])))



def getMinMaxMapLayer(im0,im1,im2):
    block = np.stack((im0,im1,im2))
    res = np.zeros(im1.shape, dtype=np.bool)
    maxHeight = im1.shape[0]
    maxWidth = im1.shape[1]
    for i in range(maxHeight):
        for j in range(maxWidth):
            print(i,j)
            isMin = True
            isMax = True
            for k in range(max(0,i-1),min(maxHeight,i+2)):
                for l in range(max(0,j-1),min(maxWidth,j+2)):
                    for d in range(0,3):
                        if not (i == k and j == l and d == 1 ):
                            isMin = isMin and (block[d,k,l] > block[1,i,j])
                            isMax = isMax and (block[d,k,l] < block[1,i,j])
                        if not (isMax or isMin):
                            break
                    if not (isMax or isMin):
                        break
                if not (isMax or isMin):
                    break
            res[i,j] = isMin or isMax
    return res

def getMinMaxMap(im0,im1,im2):
    res = np.zeros(im1.shape,dtype=np.bool)
    #BGR
    if(len(im1.shape) == 3):
        for i in range(3):
            res[...,i] = getMinMaxMapLayer(im0[...,i],im1[...,i],im2[...,i])
    #B&W
    else:
        res = getMinMaxMapLayer(im0,im1,im2)
    return res

##############
#
#   Ej 1 functions
#
##############


def unpackSIFTOctave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave & 0xFF
    layer = (_octave >> 8) & 0xFF
    if octave >= 128:
        octave |= -128
    if octave >= 0:
        scale = float(1/(1 << octave))
    else:
        scale = float(1 << -octave)
    return octave, layer, scale

def countSIFTKeypoints(kpt,octaves=10,layers=4):
    counter = np.zeros((octaves+1,layers+1))
    counter[0,1:] = np.arange(0,layers)
    counter[1:, 0] = np.arange(-1,octaves-1)

    counter[0,0] = np.NAN

    for k in kpt:
        oct, lay, sc = unpackSIFTOctave(k)
        counter[oct+2,lay] += 1
    counter = counter[np.max(counter[:, 1:],1) > 0]
    return counter

def countSURFKeypoints(kpt,octaves=10):
    counter = np.zeros((octaves,2),dtype=np.int)
    counter[:, 0] = np.arange(0,octaves)

    for k in kpt:
        counter[k.octave,1] += 1

    counter = counter[counter[:,1] > 0]
    return counter

def transformKeypoints(kpt):
    for k in kpt:
        k.octave = unpackSIFTOctave(k)[0]
    return kpt

def getSIFT(img,contrast_threshold=0.08,edge_threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create(contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)
    kp = sift.detect(gray)
    # https://docs.opencv.org/3.4.3/d0/d13/classcv_1_1Feature2D.html#ab3cce8d56f4fc5e1d530b5931e1e8dc0
    # Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation)
    kp, ds = sift.compute(img,kp)
    return kp, ds

def getSURF(img,hessianThreshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=hessianThreshold)
    kp = surf.detect(gray,None)
    kp, ds = surf.compute(img,kp)
    return kp, ds

def pintaPoints(img,points):
    pass

def getColors(octaves,layers):
    # Calculate diferent colors
    colors = np.zeros((octaves, layers, 3), dtype=np.uint8) + 255
    hues = np.repeat(np.arange(0, 180, 180 / (octaves), dtype=np.uint8).reshape((-1, 1)), layers, 1)
    lums = np.repeat(np.arange(64, 256, (256 - 64) / layers, dtype=np.uint8).reshape((-1, 1)), octaves, 1).transpose()
    lums = lums * (255/lums.max())
    colors[:, :, 0] = hues
    colors[:, :, 2] = lums
    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)
    #pintaI(colors)
    return colors

def rescaleColors(colors,r,c):
    res = np.zeros((r,c,3))
    multirow = len(colors.shape) == 3
    for i in range(r):
        for j in range(c):
            ni = int(i/r*colors.shape[0])
            nj = int(j/c*colors.shape[1])
            if multirow:
                res[i,j] = colors[ni,nj]
            else:
                res[i,j] = colors[ni]
    return res

def appendColors(img,colors):
    if(len(colors.shape) == 3):
        octaves, layers = colors.shape[:2]
    else:
        octaves, layers = colors.shape[0],1
    k = 15
    colorsbig = rescaleColors(colors, k * octaves, k * layers)
    img2 = np.zeros((img.shape[0], img.shape[1] + colorsbig.shape[1], img.shape[2]), dtype=np.uint8)
    img2[:, :img.shape[1]] = img
    img2[:colorsbig.shape[0], img.shape[1]:] = colorsbig
    return img2

def getTrans(pt,r,a):
    a = a*0.01745329252
    px,py=pt
    p = np.array([r,0]).reshape((2,1))
    M = np.array([[np.cos(a), np.sin(a)*-1],[np.sin(a),np.cos(a)]])
    ptt = np.matmul(M,p)
    return tuple(map(int,np.array(pt)+ptt.reshape(-1)))

def pintaSiftPoints(img,kp,octaves,layers,getSize=(lambda k: int(k.size/2)),doAppend = True):
    img_ori = img.copy()
    colors = getColors(octaves,layers)
    for k in kp:
        pxy = tuple(map(int,np.round(k.pt)))
        oc,ly,sc = unpackSIFTOctave(k)
        c = list(map(int, colors[oc+1,ly]))
        img = cv2.circle(img.copy(),pxy,getSize(k),c,lineType=cv2.LINE_AA)
        img = cv2.line(img.copy(),pxy,getTrans(pxy,getSize(k),k.angle),c,lineType=cv2.LINE_AA)
    if(doAppend):
        return appendColors(img,colors)
    else:
        return img
    #pintaI(appendColors(img,colors))
    #pintaMI([appendColors(img,colors),cv2.drawKeypoints(img_ori,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)])

def pintaSurfPoints(img,kp,octaves,getSize=(lambda k: int(k.size/2)),doAppend=True):
    colors = getColors(octaves, 1).reshape((-1,3))
    for k in kp:
        pxy = tuple(map(int, np.round(k.pt)))
        oc = k.octave
        c = list(map(int, colors[oc]))
        #img = cv2.circle(img.copy(), pxy, getSize(k), c)
        img = cv2.circle(img.copy(), pxy, getSize(k), c, lineType=cv2.LINE_AA)
        img = cv2.line(img.copy(), pxy, getTrans(pxy, getSize(k), k.angle), c, lineType=cv2.LINE_AA)
    if (doAppend):
        return appendColors(img,colors)
    else:
        return img
    #pintaI(appendColors(img, colors))


def ej1(img, pinta=True):
    siftkp, siftDs = getSIFT(img,0.08,10)
    surfkp, surfDs = getSURF(img,300)

    print("SIFT count")
    siftCount = countSIFTKeypoints(siftkp)
    print(siftCount)
    print("SURF count")
    surfCount = countSURFKeypoints(surfkp)
    print(surfCount)

    if pinta:
        pintaI(pintaSiftPoints(img,siftkp,siftCount.shape[0]-1,siftCount.shape[1]-1))

    SAMPLE_SIZE = 300
    #Bests
    sortedKp = sorted(surfkp,key=(lambda k: k.response),reverse=True)
    if pinta:
        pintaI(pintaSurfPoints(img,random.sample(surfkp,SAMPLE_SIZE),surfCount.shape[0]))
        #pintaI(pintaSurfPoints(img,sortedKp[:SAMPLE_SIZE],surfCount.shape[0]))
    #Worsts
    # sortedKp = sorted(surfkp, key=(lambda k: k.response), reverse=False)
    # pintaI(pintaSurfPoints(img, random.sample(sortedKp[:300], SAMPLE_SIZE), surfCount.shape[0]))

    return siftkp, siftDs, surfDs, surfkp

##############
#
#   Ej 2 functions
#
##############

def getMedianAngle(kp1,kp2,matches,desp):
    angles = np.zeros(len(matches))
    for i,m in enumerate(matches):
        i1, i2 = m.queryIdx, m.trainIdx
        k1, k2 = kp1[i1], kp2[i2]
        p1, p2 = tuple(map(int, np.round(k1.pt))), tuple(map(int, np.round(k2.pt) + [desp, 0]))

        angles[i] = np.arctan2(p1[0] - p2[0], p1[1] - p2[1])
    return np.median(angles)

def pintaLines(img1,img2,kp1,kp2,matches,use_angle=True, randomColor=False):
    imgd = np.zeros((np.max((img1.shape[0], img2.shape[0])), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    desp = img1.shape[1]
    imgd[:img1.shape[0], :img1.shape[1]] = img1
    imgd[:img2.shape[0], desp:] = img2


    a = getMedianAngle(kp1,kp2,matches,desp)
    for m in matches:
        i1, i2 = m.queryIdx, m.trainIdx
        k1, k2 = kp1[i1], kp2[i2]
        p1, p2 = tuple(map(int, np.round(k1.pt))), tuple(map(int, np.round(k2.pt)+[desp,0]))

        ma = np.arctan2(p1[0]-p2[0],p1[1]-p2[1])
        if (use_angle and np.abs(ma-a) > 0.05):
            #cv2.line(img.copy(), pxy, getTrans(pxy, getSize(k), k.angle), c, lineType=cv2.LINE_AA)
            imgd = cv2.line(imgd.copy(), p1, p2, (0, 0, 255), lineType=cv2.LINE_AA)
        else:
            if not randomColor:
                imgd = cv2.line(imgd.copy(), p1, p2, (255, 0, 0), lineType=cv2.LINE_AA)
            else:
                imgd = cv2.line(imgd.copy(), p1, p2, tuple(map(int,np.random.randint(0,255,3))), lineType=cv2.LINE_AA)

    return imgd

def getPointsFromMatch(kp1,kp2,match):
    l1, l2 = [],[]
    for m in match:
        l1.append(kp1[m.queryIdx].pt)
        l2.append(kp2[m.trainIdx].pt)
    return l1,l2

def ej2(img1,img2,siftKp1, siftKp2, siftDs1, siftDs2, pinta=True):
    #Bruteforce + Crosscheck
    bf_cc = cv2.BFMatcher.create(crossCheck=True)
    matches = bf_cc.match(siftDs1,siftDs2)
    print("#Matches BF+CC: {}".format(len(matches)))
    sm = sorted(matches,key=(lambda m: m.distance))
    bf_rand = pintaLines(img1, img2, siftKp1, siftKp2, random.sample(matches,100))
    bf_best = pintaLines(img1, img2, siftKp1, siftKp2, sm[:100])

    if pinta:
        pintaMISingleMPL([bf_rand,bf_best],["Random","Best"],title="Bruteforce + CC"+" "*72,cols=1)

    #Lowe-Average-2NN
    bf_2nn = cv2.BFMatcher.create(crossCheck=False)
    matches = bf_2nn.knnMatch(siftDs1,siftDs2,2)
    lowes = list(map(lambda x: x[0],(filter(lambda x: x[0].distance < 0.7*  x[1].distance,matches))))
    if pinta:
        pintaI(pintaLines(img1, img2, siftKp1, siftKp2, random.sample(lowes, 100), randomColor=False))
        pintaI(pintaLines(img1, img2, siftKp1, siftKp2, random.sample(lowes, 100), randomColor=True))
        print("#Matches Lowe's: {}".format(len(lowes)))

    return getPointsFromMatch(siftKp1,siftKp2,lowes)

def getTransNeeded(dimxy,lh):
    minx = np.inf
    maxx = - np.inf
    miny = np.inf
    maxy = - np.inf
    for h in lh:
        ps = np.array([[0,0,1],[0,dimxy[1],1],[dimxy[0],0,1],[dimxy[0],dimxy[1],1]]).transpose()
        ps = h.dot(ps)
        ps = ps/ps[2,:]
        min = np.min(ps,1)
        max = np.max(ps,1)

        minx = np.min((minx,min[0]))
        maxx = np.max((maxx,max[0]))

        miny = np.min((miny,min[1]))
        maxy = np.max((maxy,max[1]))

    return minx, maxx, miny, maxy

def getPoint(p1,h):
    a = h.dot(np.array([p1[0],p1[1],1]).reshape((-1,1)))
    a = a/a[2]
    return a[:2]

def ransacHomografy(p1l,p2l,reThreshold=3,maxIters=2000,confidence=0.85,everyStep=False):
    #Define minD
    min_dist = np.Inf
    min_h = np.zeros((3,3))
    min_mask = np.zeros(p1l.shape[0], dtype=np.bool)

    p1e = np.hstack((p1l, np.zeros((p1l.shape[0],1))+1)).transpose()
    p2e = np.hstack((p2l, np.zeros((p2l.shape[0], 1))+1)).transpose()
    for _ in range(maxIters):
        #sample points
        idxs = np.random.choice(p1l.shape[0],4,replace=False)
        pp1 = p1l[idxs,:]
        pp2 = p2l[idxs,:]

        #calc homography
        ps = []
        for i in range(4):
            p1 = [pp1[i][0], pp1[i][1]]
            p2 = [pp2[i][0], pp2[i][1]]

            a2 = [0, 0, 0, -p1[0], -p1[1], -1,
                  p2[1] * p1[0], p2[1] * p1[1], p2[1]]
            a1 = [-p1[0], -p1[1], -1, 0, 0, 0,
                  p2[0] * p1[0], p2[0] * p1[1], p2[0]]

            ps.append(a1)
            ps.append(a2)

        mPts = np.array(ps)

        # svd composition
        u, s, v = np.linalg.svd(mPts)

        # Linealmente dependiente
        if (np.abs(s)< 1e-32).sum() > 0:
            continue

        # get h and normalize
        h = v[8,:].reshape((3, 3))
        h = (1 / h[2,2]) * h

        #calc distance
        p1p = h.dot(p1e)
        if (p1p[2,:] == 0).sum() > 0:
            continue
        p1p = p1p / p1p[2,:]
        dist = np.sqrt(np.power(p1p-p2e,2).sum(0))

        #Select Inliners
        msk = dist<reThreshold
        conf = (msk).sum()/p1l.shape[0]

        #print(conf)

        if(confidence <= conf):
            if(everyStep):
                #Calculate new H with inliners
                new_p1 = p1e[:,msk]
                new_p2 = p2e[:,msk]

                def evalH(h):
                    nonlocal new_p1
                    nonlocal new_p2
                    r = h.reshape((3,3)).dot(new_p1)
                    r = r / r[2, :]
                    return [np.sqrt(np.power(r - new_p2, 2).sum(0)).sum()]+([0]*8)
                #Levenberg-Marquardt
                new_h = root(evalH,h.reshape(-1),method='lm')
                #update if fitter
                dist = new_h.fun[0]
                new_h = new_h.x.reshape((3,3))
            else:
                # update if fitter
                dist = dist.sum()
                new_h = h
            # update if fitter
            if (dist < min_dist):
                min_dist = dist
                min_h = new_h
                min_mask = msk
    if not everyStep:
        # Calculate new H with inliners
        new_p1 = p1e[:, min_mask]
        new_p2 = p2e[:, min_mask]

        def evalH(h):
            nonlocal new_p1
            nonlocal new_p2
            r = h.reshape((3, 3)).dot(new_p1)
            r = r / r[2, :]
            return [np.sqrt(np.power(r - new_p2, 2).sum(0)).sum()] + ([0] * 8)

        # Levenberg-Marquardt
        min_h = root(evalH, min_h.reshape(-1), method='lm').x.reshape((3,3))
    return min_h, min_mask



def getHomografies(imgs, fH=(lambda p1,p2: cv2.findHomography(p1,p2,cv2.RANSAC,1)), pointsAlg=getSIFT):
    sifts = [pointsAlg(img) for img in imgs]
    hs = [np.eye(3)]
    for i in range(len(imgs)-1):
        bf_2nn = cv2.BFMatcher.create(crossCheck=False)
        matches = bf_2nn.knnMatch(sifts[i][1], sifts[i+1][1], 2)
        lowes = list(map(lambda x: x[0], (filter(lambda x: x[0].distance < 0.7 * x[1].distance, matches))))
        p1,p2 = getPointsFromMatch(sifts[i][0],sifts[i+1][0],lowes)
        #h,mask = cv2.findHomography(np.array(p2),np.array(p1),cv2.RANSAC,1)
        #h,mask = ransacHomografy(np.array(p2),np.array(p1))
        h,mask = fH(np.array(p2),np.array(p1))
        hs.append(hs[i].dot(h))
    return hs


def ej3(imgs):
    hs = getHomografies(imgs)
    minx, maxx, miny, maxy = getTransNeeded((imgs[0].shape[1], imgs[0].shape[0]), hs)
    #im = np.zeros((int(np.ceil(maxy-miny)),int(np.ceil(maxx-minx)),3),dtype=np.uint8)
    im = np.zeros((int(maxy - miny), int(maxx - minx), 3), dtype=np.uint8)
    ht = np.eye(3)
    ht[0:2,2] = [-minx,-miny]

    for img,h in zip(imgs,hs):
        cv2.warpPerspective(img,ht.dot(h),(im.shape[1],im.shape[0]),dst=im,borderMode=cv2.BORDER_TRANSPARENT)

    #pintaI(im)
    return im

def b3(imgs):
    hs = getHomografies(imgs,fH=(lambda p1,p2: ransacHomografy(p1,p2,reThreshold=1,confidence=0.9)))
    minx, maxx, miny, maxy = getTransNeeded((imgs[0].shape[1], imgs[0].shape[0]), hs)
    im = np.zeros((int(np.ceil(maxy - miny)), int(np.ceil(maxx - minx)), 3), dtype=np.uint8)
    ht = np.eye(3)
    ht[0:2, 2] = [-minx, -miny]

    for img, h in zip(imgs, hs):
        cv2.warpPerspective(img, ht.dot(h), (im.shape[1], im.shape[0]), dst=im, borderMode=cv2.BORDER_TRANSPARENT)

    # pintaI(im)
    return im



def ej4(imgs):
    hs = getHomografies(imgs)
    minx, maxx, miny, maxy = getTransNeeded((imgs[0].shape[1], imgs[0].shape[0]), hs)
    im = np.zeros((int(np.ceil(maxy - miny)), int(np.ceil(maxx - minx)), 3), dtype=np.uint8)
    ht = np.eye(3)
    ht[0:2, 2] = [-minx, -miny]

    ims = []
    for img, h in zip(imgs, hs):
        ims.append(cv2.warpPerspective(img, ht.dot(h), (im.shape[1], im.shape[0])))

    imstack = np.stack(ims,axis=-1)

    #Mean
    # im = np.true_divide(imstack.sum(-1),(imstack!=0).sum(-1))
    # im = np.nan_to_num(im,copy=False).astype(np.uint8)
    #Max
    im = np.max(imstack,axis=-1)
    #pintaI(im)
    return im.astype(np.uint8)

# def b1p(img):
#     P_LEVELS = 5
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     p = [cv2.pyrUp(gray),gray]
#     for i in range(P_LEVELS-2):
#         p.append(cv2.pyrDown(p[i+1]))
#     dst = cv2.cornerHarris(p[1].astype(np.float32), 2, 3, 0.04)
#     dst = cv2.dilate(dst, None)
#     #dst = cv2.dilate(cv2.cornerHarris(p[1].astype(np.float32),2,3,0.04),None)
#
#     ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
#     dst = np.uint8(dst)
#
#     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
#
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#     corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
#
#     res = np.hstack((centroids, corners))
#     res = np.int0(res)
#     for r in res:
#         img = cv2.line(img,(r[0],r[1]),(r[2],r[3]),[0,0,255])
#     img[res[:, 1], res[:, 0]] = [0, 0, 255]
#     img[res[:, 3], res[:, 2]] = [0, 255, 0]
#
#     # msk = dst > 0.01 * dst.max()
#     # img[msk] = np.random.randint(0,255,(msk.sum(),3)).astype(np.uint8)
#     pintaI(img)

def anmsPoints(img,max_points=500):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float64)
    #Get Pyramid
    P_LEVELS = 5
    p = [cv2.pyrUp(gray),gray]
    #p = [gray]
    for i in range(P_LEVELS - 2):
        p.append(cv2.pyrDown(p[i+1]))

    dk = np.array([-1,0,1])
    gk = cv2.getGaussianKernel(5,1)

    kplist = []
    for octave,l in enumerate(p):
        octave = octave-1
        gl = cv2.GaussianBlur(l,(0,0),1)

        dx = cv2.Sobel(gl,-1,1,0)
        dy = cv2.Sobel(gl,-1,0,1)

        #Integration scale
        integ_s = 1.5

        ixx = cv2.GaussianBlur(dx**2,(0,0),integ_s)
        ixy = cv2.GaussianBlur(dy*dx,(0,0),integ_s)
        iyy = cv2.GaussianBlur(dy**2,(0,0),integ_s)

        dt = (ixx*iyy)-(ixy**2)
        tr = (ixx+iyy)

        #Get score matrix
        s = np.nan_to_num(dt/tr)
        ss = np.zeros(s.shape)
        threshold = 10

        #Get local maxima 3x3
        soffset = np.stack((
            s[0:-2, 0:-2],
            s[1:-1, 0:-2],
            s[2:, 0:-2],
            s[0:-2, 1:-1],
            s[1:-1, 1:-1],
            s[2:, 1:-1],
            s[0:-2, 2:],
            s[1:-1, 2:],
            s[2:, 2:]
        ),axis=-1)

        ss2 = np.max(soffset,axis=2)
        mskv = np.equal(ss2,s[1:-1,1:-1])
        ss[1:-1,1:-1][mskv] = s[1:-1,1:-1][mskv]
        #pintaI(visualizeM(ss))

        #Get subpixel
        #http://vision-cdc.csiro.au/changs/doc/sun02ivc.pdf
        #http://www.jmest.org/wp-content/uploads/JMESTN42351547.pdf
        points = []
        for i,j in zip(*np.where(ss>threshold)):
            A = (s[i-1,j-1] - 2*s[i,j-1] + s[i+1,j-1] + s[i-1,j] - 2*s[i,j] + s[i+1,j] +s[i-1,j+1] - 2*s[i,j+1] +s[i+1,j+1])/6
            B = (s[i-1,j-1] - s[i+1,j-1] - s[i-1,j+1] + s[i+1,j+1])/4
            C = (s[i-1,j-1] + s[i,j-1] + s[i+1,j-1] - 2*s[i-1,j] - 2*s[i,j] - 2*s[i+1,j] + s[i-1,j+1] + s[i,j+1] + s[i+1,j+1])/6
            D = (-s[i-1,j-1] + s[i+1,j-1] - s[i-1,j] + s[i+1,j] - s[i-1,j+1] + s[i+1,j+1])/6
            E = (-s[i-1,j-1] - s[i,j-1] - s[i+1,j-1] + s[i-1,j+1] + s[i,j+1] + s[i+1,j+1])/6
            F = (-s[i,j] + 2*s[i,j] - s[i,j] + 2*s[i,j] + 5*s[i,j] + 2*s[i,j] - s[i,j] + 2*s[i,j] - s[i,j])/9

            ox = (B*E - 2*C*D)/(4*A*C - B**2)
            oy = (B*D - 2*A*E)/(4*A*C - B**2)
            new_v = A*ox**2 + B*ox*oy + C*oy**2 + D*ox + E*oy + F

            x=j
            y=i

            to_add = [x,y,ss[i,j],x+ox,y+oy, new_v]
            points.append(to_add)
            #print(to_add)

        #Get angle
        gu = cv2.GaussianBlur(l,(0,0),4.5)
        dx = cv2.Sobel(gu,-1,1,0)
        dy = cv2.Sobel(gu,-1,0,1)

        mg = np.sqrt(dx**2+dy**2)
        a1 = np.zeros(mg.shape)
        a2 = a1.copy()

        msk = mg != 0
        #a1[msk] = np.arccos(dx[msk]/mg[msk])
        a2[msk] = np.arcsin(dy[msk]/mg[msk])

        #Convert to keypoints
        nmsk = np.zeros(l.shape)
        deg_factor = 180/np.pi
        for [x,y,v,nx,ny,nv] in points:
            kp = cv2.KeyPoint(x=nx*(2**octave), y=ny*(2**octave), _angle=a2[y,x]*deg_factor,_size=(2**octave)*3, _response=nv, _octave=octave)
            kplist.append(kp)

    #Adaptative Non-maximal supresion
    #Triangular matrix but this way is faster
    pts = np.array([[p.response,p.pt[0],p.pt[1]] for p in kplist],dtype=np.float32)
    vs = pts[:,0]
    xs = np.repeat(pts[:,1].reshape(1,-1),pts.shape[0],axis=0)
    ys = np.repeat(pts[:,2].reshape(1,-1),pts.shape[0],axis=0)

    xs2 = pts[:, 1]
    ys2 = pts[:, 2]
    #LENTO
    #Get distances
    # Without root
    print("Processing distances...")
    dis = ((xs - xs.transpose()) ** 2) + ((ys - ys.transpose()) ** 2)
    rs = np.zeros(vs.shape)
    max_vs = vs.max()
    for i in range(len(kplist)):
        gt = dis[i,:][0.9*vs > vs[i]]
        if (gt.size == 0):
            if(vs[i] == max_vs):
                #print("#", i, vs[i])
                rs[i] = np.inf
            #else ignore
        else:
            idxs = np.where(dis[i, :] == gt.min())[0]
            rs[i] = dis[i,idxs[0]]

    bst_kplist = [x for _,x in sorted(zip(rs,kplist),key=lambda x: x[0], reverse=True)]

    # # Pinta puntos
    # imgp = pintaSurfPoints(img, bst_kplist[:500], octaves=P_LEVELS)
    # pintaI(imgp)

    # # Pinta correspondencia response / rs
    # rs = np.sqrt(rs)
    # normalized = (rs / rs[rs != np.inf].max())
    # normalized[normalized == np.inf] = 1
    # normalized = np.repeat(normalized.reshape((1,-1)),100,axis=0)
    #
    # vss = np.repeat(vs.reshape(1,-1),100,axis=0)
    #
    # pintaI(np.vstack((visualizeM(normalized),visualizeM(vss))))

    return bst_kplist[:np.min((len(bst_kplist),max_points))]
    #return sorted(kplist,key=(lambda x: x.response),reverse=True)[:500]

def discreteHaarWaveletTransform(x):
    N = len(x)
    output = np.zeros(N)

    length = N >> 1
    sq2 = np.sqrt(2)
    while True:
        for i in range(0,length):
            summ = (x[i * 2] + x[i * 2 + 1])/sq2
            difference = (x[i * 2] - x[i * 2 + 1])/sq2
            output[i] = summ
            output[length + i] = difference

        if length == 1:
            return output

        #Swap arrays to do next iteration
        #System.arraycopy(output, 0, x, 0, length << 1)
        x = output[:length << 1].copy()

        length >>= 1

def anmsDescriptors(img,kps):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rad_factor = np.pi/180
    s = 2.7
    img_b = cv2.GaussianBlur(gray,(0,0),s)
    t_m = np.eye(3)
    t_m[0,0] = 1/5
    t_m[1, 1] = 1 / 5
    t_m[0,2] = 4
    t_m[1, 2] = 4
    # img_r = cv2.warpAffine(img_b,t_m,(img_b.shape[1]//5,img_b.shape[0]//5))
    # pintaI(img_r)
    kps2 = []
    descr = []
    for i, kp in enumerate(kps):
        x,y = kp.pt
        a = kp.angle*rad_factor
        s = 1/np.power(2.,kp.octave)

        #Calculate Matrix
        t = np.eye(3)
        t[0, 2] = -x*s
        t[1, 2] = -y*s

        t[0,0] = s
        t[1,1] = s

        r = np.eye(3)
        r[:2,:2] = np.array([
            [np.cos(a), -1*np.sin(a)],
            [np.sin(a), np.cos(a)]
        ])

        m = t_m.dot((r.dot(t)))

        sq = cv2.warpAffine(img_b,m[:2,:],dsize=(8,8),flags=cv2.INTER_LINEAR).reshape(-1)
        #sq = cv2.warpAffine(gray, m[:2, :], dsize=(8, 8), flags=cv2.INTER_LINEAR).reshape(-1)
        #print(kp.pt,kp.octave,s)
        #pintaI(sq.reshape((8,8)))


        #Normalize sq
        std = np.std(sq)
        if (std == 0):
            #Discar point
            continue
        sq = sq / np.std(sq)
        sq = sq - np.mean(sq)

        kps2.append(kp)

        #Haar wavelet transform
        wt = discreteHaarWaveletTransform(sq)
        descr.append(wt)

    return kps2, np.array(descr,dtype=np.float32)



def anmsAndSiftDes(img):
    kps = anmsPoints(img)
    sift = cv2.xfeatures2d_SURF.create(nOctaves=5)
    kp, ds = sift.compute(img, kps)
    return kp, ds

def anmsAndDes(img):
    kps = anmsPoints(img)
    kps, ds = anmsDescriptors(img, kps)
    return kps, ds

def bonus(imgs,ptAndDes=anmsAndDes,fH=(lambda p1,p2: cv2.findHomography(p1,p2,cv2.RANSAC,1))):
    hs = getHomografies(imgs,fH=fH,pointsAlg=ptAndDes)
    minx, maxx, miny, maxy = getTransNeeded((imgs[0].shape[1], imgs[0].shape[0]), hs)
    im = np.zeros((int(maxy - miny), int(maxx - minx), 3), dtype=np.uint8)
    ht = np.eye(3)
    ht[0:2, 2] = [-minx, -miny]

    for img, h in zip(imgs, hs):
        cv2.warpPerspective(img, ht.dot(h), (im.shape[1], im.shape[0]), dst=im, borderMode=cv2.BORDER_TRANSPARENT)

    # ims = []
    # for img, h in zip(imgs, hs):
    #     ims.append(cv2.warpPerspective(img, ht.dot(h), (im.shape[1], im.shape[0])))
    #
    # imstack = np.stack(ims, axis=-1)
    #
    # # Mean
    # # im = np.true_divide(imstack.sum(-1),(imstack!=0).sum(-1))
    # # im = np.nan_to_num(im,copy=False).astype(np.uint8)
    # # Max
    # im = np.max(imstack, axis=-1)

    return im

def main():
    # Ejercicio 1
    print("Ejercicio 1")
    img1 = leeimagen("imagenes/yosemite1.jpg",is_float=False)
    img2 = leeimagen("imagenes/yosemite2.jpg", is_float=False)
    siftKp1, siftDs1, _, _ = ej1(img1, pinta=True)
    siftKp2, siftDs2, _, _ = ej1(img2, pinta=False)

    # Ejercicio 2
    print("Ejercicio 2")
    ej2(img1,img2,siftKp1,siftKp2,siftDs1,siftDs2,pinta=True)


    #Ejercicio 3/4
    print("Ejercicio 3/4")
    imgs = [leeimagen("imagenes/yosemite{}.jpg".format(i),is_float=False) for i in range(1,5)]
    pintaI(ej3(imgs))

    imgs = [leeimagen("imagenes/yosemite{}.jpg".format(i), is_float=False) for i in range(5, 8)]
    pintaI(ej3(imgs))

    imgs = [leeimagen("imagenes/mosaico{:03d}.jpg".format(i), is_float=False) for i in range(2, 12)]
    im1 = ej3(imgs)
    im2 = ej4(imgs)
    pintaMI([im1,im2])
    cv2.destroyAllWindows()

    # Bonus
    imgs = [leeimagen("imagenes/mosaico{:03d}.jpg".format(i), is_float=False) for i in range(2, 12)]

    # Bonus.1
    print("Bonus 1")
    pintaI(bonus(imgs,ptAndDes=anmsAndSiftDes))

    # Bonus.2
    print("Bonus 2")
    pintaI(bonus(imgs,ptAndDes=anmsAndDes))

    # Bonus.3
    print("Bonus 3")
    pintaI(b3(imgs))
    #pintaI(bonus(imgs, ptAndDes=anmsAndDes, fH=ransacHomografy))

    # Uniendo todo
    print("Bonus 1+2+3")
    pintaI(bonus(imgs, ptAndDes=anmsAndDes, fH=ransacHomografy))

    # Yosemite 1-4
    print("Bonus 1+2+3 Yosemite")
    imgs = [leeimagen("imagenes/yosemite{}.jpg".format(i), is_float=False) for i in range(1, 5)]
    pintaI(bonus(imgs, ptAndDes=anmsAndDes, fH=ransacHomografy))

    # Yosemite 5-7 (Cambiar umbral de RANSAC a 0.5
    #imgs = [leeimagen("imagenes/yosemite{}.jpg".format(i), is_float=False) for i in range(5, 8)]
    #pintaI(bonus(imgs, ptAndDes=anmsAndDes, fH=ransacHomografy))



if __name__ == "__main__":
    main()