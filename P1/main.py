import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import exp,ceil

# In:
#   filename: File path
#   flagcolor: cv2 color flag [0 Grayscale, 1 Color]
# Out:
#   Image matrix (H,W,C[BGR])
def leeimagen(filename,flagcolor):
    M = cv2.imread(filename,flagcolor)
    return M

# In:
#   im: Image matrix (H,W,C[BGR])
#   window_name: Window's title
#   wait: call cv2.waitKey()
def pintaI(im,window_name="window",wait=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name,im)
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

def main():
    # Leemos la imagen y la definimos como coma flotante entre 0 y 1
    img = leeimagen("imagenes/motorcycle.bmp",1).astype(np.float)/255.
    imgbw = leeimagen("imagenes/einstein.bmp", 0).astype(np.float) / 255.
    #
    # ## Ejercicio 1.A
    #
    # # Vamos a definir tamaños de la convolución entre 3 y 15
    # # para ver el efecto, a mayor tamaño de kernel, como OpenCV
    # # calcula sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 esperamos
    # # ver imágenes más suavizadas, más adelante veremos ejemplos
    # # definiendo nosotros mismos el sigma
    #
    # print("Ejercicio 1.A")
    #
    # blurred_images = []
    # blurred_titles = []
    # for ksize in range(3,17,4): #kernel size: 3,7,11,15
    #     blurred_images.append(applyGaussianBlur(img,ksize,-1))
    #     blurred_titles.append("ksize {}".format(ksize))
    # pintaMISingleMPL(blurred_images,title="Gaussian",labels=blurred_titles,cols=2)
    #
    # wait()
    # ## Ejercicio 1.B
    #
    # # De la misma forma que en el caso anterior, como
    # # estamos calculando la derivada de primer orden
    # # sobre la gaussiana de ksize dado, a mayor valor
    # # de éste se calcula la derivada sobre una versión
    # # más suavizada de la imagen, perdiendo mayor cantidad
    # # de altas frecuencias y resaltando el contraste de
    # # frecuencias más bajas
    #
    # # De ahora en adelante cuando queramos visualizar una
    # # imagen a la que hemos aplicado una convolución
    # # que devuelve valores superiores inferiores a 0
    # # o superiores a 1 utilizamos la función visualizeM
    # # que no hace más que normalizar entre 0 y 255 la
    # # imagen de entrada
    #
    # print("Ejercicio 1.B")
    #
    # deriv_images = []
    # deriv_titles = []
    # for ksize in range(3,17,4): #1,3,5,7
    #     im = applyDeriv(img,ksize)
    #     im = visualizeM(im)
    #     deriv_images.append(im)
    #     deriv_titles.append("ksize {}".format(ksize))
    # pintaMISingleMPL(deriv_images,title="Derivative",labels=deriv_titles,cols=2)
    #
    # kx, ky = cv2.getDerivKernels(1, 0, 9)
    # pintaMISingleMPL([visualizeM(cv2.sepFilter2D(img,-1,kx,ky)),visualizeM(cv2.sepFilter2D(img,-1,ky,kx))], title = "Derivative", labels = ["dx","dy"], cols = 2)
    #
    #
    # wait()
    # #sigma variable?
    # #sigma – Gaussian standard deviation. If it is non-positive, it is computed from ksize as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 .
    #
    # ## Ejercicio 1.C
    #
    # # Para este ejercicio usamos dos tipos de borde
    # # cv2.BORDER_CONSTANT (borde constante negro),
    # # y cv2.BORDER_REFLECT_101 (dcb|abcd|cba) el borde
    # # por defecto en OpenCV
    # #
    # # Como no podemos definir el sigma de la gaussiana
    # # utilziaremos la inversa de la fórmula de OpenCV
    # # ksize = ceil((2 * sigma - 1) / 0.3) para calcular
    # # el ksize dado un sigma (primer impar igual o superior)
    # # en este caso f(1)=4=>5, y f(3)=17=>17
    # #
    # # El efecto del tipo de borde se aprecia claramente
    # # contrastando los dos a usar, el reflejo continua
    # # los valores que había próximos al borde, mientras
    # # que el constante da valores bajos en los extremos,
    # # como un marco negro
    # #
    # # Por otro lado, el efecto del valor de sigma, como
    # # efecto del pattern matching de la máscara que estamos
    # # aplicando busca detalles más gruesos a medida que
    # # aumentamos el tamaño de sigma
    #
    # print("Ejercicio 1.C")
    #
    # kx = cv2.getDerivKernels(2,0,31)
    # ky = cv2.getDerivKernels(0,2,31)
    #
    # kx = kx[0] * kx[1].transpose()
    # ky = ky[0] * ky[1].transpose()
    #
    # pintaMISingleMPL([visualizeM(kx),visualizeM(ky),visualizeM(kx+ky)],labels=["d2x","d2y","lap"],title="Laplacian")
    # wait()
    #
    # laplace_images = []
    # laplace_titles = []
    # for ksize in [5,17]:
    #     for borderType in [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT_101]:
    #         im = visualizeM(cv2.Laplacian(img,-1, ksize =ksize, borderType= borderType))
    #         laplace_images.append(im)
    #         st = "Constant" if borderType == cv2.BORDER_CONSTANT else "Reflect"
    #         laplace_titles.append("{}-{}".format(st,ksize))
    #
    # pintaMISingleMPL(laplace_images,labels=laplace_titles,title="Laplace",cols=2)
    #
    # wait()
    #
    #
    # ## Ejercicio 2.A,B,C
    #
    # # Vamos a aplicar tres tipos de kernels separables,
    # # gaussiana y derivada de la gaussiana de primer y
    # # segundo orden
    #
    # print("Ejercicio 2.A,B,C")
    #
    # ka = cv2.getGaussianKernel(5,-1)
    # kb = cv2.getDerivKernels(1,1,19)
    # kc = cv2.getDerivKernels(2,2,19)
    #
    # pintaMISingleMPL([
    #     visualizeM(kc[0].reshape((1, -1))),
    #     visualizeM(kc[1].reshape((-1, 1))),
    #     visualizeM(kc[0] * kc[1].transpose())
    # ], ["x", "y", "x*y"])
    #
    # ima = applySepConv(imgbw,(ka,ka),BORDER_MIRROR)
    # imb = visualizeM(applySepConv(imgbw, kb, BORDER_CONSTANT))
    # imc = visualizeM(applySepConv(imgbw, kc, BORDER_MIRROR))
    #
    # pintaMISingleMPL([ima, imb, imc],["Gauss","Der1","Der2"],title="Separable",cols=2)
    #
    # wait()
    #
    # ## Ejercicio 2.D
    #
    # # En este caso vamos a ver la diferencia del efecto
    # # la piramide gaussiana dependiendo del borde, como
    # # se puede esperar, el borde constante negro introduce
    # # un halo en forma de marco oscuro alrededor de la
    # # imagen. No ilustramos el ejemplo de otros marcos
    # # como REPLICATE (aaaaaa|abcdefgh|hhhhhhh), WRAP
    # # (cdefgh|abcdefgh|abcdefg), o REFLECT (No 101:
    # # fedcba | abcdefgh | hgfedcb) porque dependen
    # # altamente de la imagen y en el caso de esta no se
    # # aprecia gran diferencia por el fondo continuo
    # # homogeneo.
    # #
    # # En la tercera imagen mostramos la diferencia entre
    # # las dos fotos, y como cabría esperar sólo existen
    # # diferencias en el borde
    #
    # print("Ejercicio 2.D\n*Pulsa Enter en las imágenes*")
    #
    # imgg1 = gaussianPyramid(imgbw,5,cv2.BORDER_DEFAULT)
    # imgg2 = gaussianPyramid(imgbw, 5, cv2.BORDER_CONSTANT)
    #
    # # M = np.zeros((imgg1.shape[0],imgg1.shape[1]*2))
    # # M[:,:imgg1.shape[1]] = imgg1
    # # M[:,imgg1.shape[1]:imgg1.shape[1]*2] = imgg2
    # # #M[:,imgg1.shape[1]*2:] = visualizeM(imgg1 - imgg2)
    # # pintaI(M)
    #
    # pintaI(imgg1,"Gaussian Pyramid Reflect", wait=False)
    # pintaI(imgg2, "Gaussian Pyramid Constant")
    # pintaI(visualizeM(imgg1-imgg2), "Reflect-Constant")
    #
    # cv2.destroyAllWindows()
    #
    #
    # ## Ejercicio 2.E
    #
    # # En este caso, como usamos un kernel de tamaño cinco,
    # # el borde de la imagen es 2px por cada lado, afectando
    # # pues únicamente a los dos primeros píxeles de la imagen
    # # antes del downscaling por lo que finalmente resulta
    # # en un borde de un 1 pixel oscuro para el caso del borde
    # # constante negro
    #
    # print("Ejercicio 2.E\n*Pulsa Enter en las imágenes*")
    #
    # imgl1 = laplacianPyramid(img,4,cv2.BORDER_DEFAULT, pyrDown= cpyrDown, pyrUp=cpyrUp)
    # imgl2 = laplacianPyramid(img, 4, cv2.BORDER_CONSTANT)
    # pintaI(visualizeM(imgl1),"Laplacian Border Reflect", wait=False)
    # pintaI(visualizeM(imgl2),"Laplacian Border Constant")
    #
    # cv2.destroyAllWindows()
    #
    # ## Ejercicio 3
    #
    # print("Ejercicio 3\n*Pulsa Enter en las imágenes*")
    #
    # hlpairs = [
    #     ["imagenes/fish.bmp","imagenes/submarine.bmp",7,2.6],
    #     ["imagenes/bicycle.bmp", "imagenes/motorcycle.bmp", 5,2.6],
    #     ["imagenes/plane.bmp", "imagenes/bird.bmp", 5,1.85],
    #     ["imagenes/cat.bmp", "imagenes/dog.bmp", 7,2.6],
    #     ["imagenes/marilyn.bmp", "imagenes/einstein.bmp", 5, 1.4],
    # ]
    # for ar in hlpairs:
    #     imgl = leeimagen(ar[0], 0).astype(np.float) / 255.
    #     imgh = leeimagen(ar[1], 0).astype(np.float) / 255.
    #
    #     #pintaMI([hybridImg(imgl,imgh,ar[2],ar[3],hF=highFreq),hybridImg(imgl,imgh,ar[2],ar[3],hF=cHighFreq)])
    #     pintaI(hybridImg(imgl,imgh,ar[2],ar[3]))
    #
    # cv2.destroyAllWindows()
    #
    # ## Bonus 1
    #
    # print("Bonus 1")
    #
    # res = []
    # ks = []
    # for i in range(3):
    #     k = gaussianMaskV(5,i)
    #     ks.append(k)
    #     kx = k.reshape((-1,1))
    #     ky = k.reshape((1,-1))
    #     m = visualizeM(kx*ky)
    #     res.append(m)
    # pintaMISingleMPL(res,title="ksize formula")
    #
    # res = []
    # for k in ks:
    #     res.append(applySepConv(img,(k,k),BORDER_MIRROR))
    # pintaMISingleMPL(res,labels=[("ksize " + str(f.size)) for f in ks],title="Kernel Size calculation",cols=2)
    #
    # cv2.destroyAllWindows()
    #
    # ## Bonus 2,3
    #
    # print("Bonus 2,3.\n*Pulsa Enter en las imágenes*")
    #
    # print("Calculando 3 aplicaciones de convolución con los tres métodos")
    #
    # iters = 3
    # k = gaussianMaskV(5)
    # import time
    # time1 = time.time()
    # for i in range(iters):
    #     im1 = applySepConv(img,(k,k),BORDER_MIRROR)
    # time2 = time.time()
    # print('MAT took {:.3f} ms'.format((time2 - time1) * 1000.0))
    #
    # time1 = time.time()
    # for i in range(iters):
    #     im2 = conv2D(img,(k,k))
    # time2 = time.time()
    # print('SEQ took {:.3f} ms'.format((time2 - time1) * 1000.0))
    #
    # time1 = time.time()
    # for i in range(iters):
    #     im3 = cv2.sepFilter2D(img,-1,k,k)
    # time2 = time.time()
    # print('OpenCV took {:.3f} ms'.format((time2 - time1) * 1000.0))
    #
    # pintaMI([im1,im2,im3])
    #
    # cv2.destroyAllWindows()
    #
    # ## Bonus 4
    #
    # print("Bonus 4\n*Pulsa Enter en las imágenes*")
    # M1 = customGaussianPyramid(img,6)
    # pintaI(M1,"Custom Gaussian Pyramid")
    # cv2.destroyAllWindows()
    #
    # ## Bonus 5
    # print("Bonus 5\n*Pulsa Enter en las imágenes*")
    # hlpairs = [
    #     ["imagenes/fish.bmp", "imagenes/submarine.bmp", 7, 2.6],
    #     ["imagenes/bicycle.bmp", "imagenes/motorcycle.bmp", 5, 2.6],
    #     ["imagenes/plane.bmp", "imagenes/bird.bmp", 5, 1.85],
    #     ["imagenes/cat.bmp", "imagenes/dog.bmp", 7, 2.6],
    #     ["imagenes/marilyn.bmp", "imagenes/einstein.bmp", 5, 1.4],
    # ]
    # for ar in hlpairs:
    #     imgl = leeimagen(ar[0], 1).astype(np.float) / 255.
    #     imgh = leeimagen(ar[1], 1).astype(np.float) / 255.
    #
    #     # pintaMI([hybridImg(imgl,imgh,ar[2],ar[3],hF=highFreq),hybridImg(imgl,imgh,ar[2],ar[3],hF=cHighFreq)])
    #     pintaI(hybridImg(imgl, imgh, ar[2], ar[3],lF=cLowFreq,hF=cHighFreq))
    #
    # cv2.destroyAllWindows()
    #
    # k = cv2.getGaussianKernel(15,-1)
    # res = [img]
    # for i in range(5):
    #     res.append(cv2.sepFilter2D(res[i],-1,k,k))
    # pintaMISingleMPL(res,labels=list(map(str,range(6))),cols=2,title="Progesivo")
    #
    # wait()
    #
    # k1 = gaussianMaskV(1.08)
    # k1 = k1.reshape(1,-1)*k1.reshape(-1,1)
    # k2 = cv2.getGaussianKernel(5,-1)
    # k2 = k2*k2.transpose()
    # pintaMISingleMPL([visualizeM(k1),visualizeM(k2),visualizeM(k1-k2)],["Custom","Rigid","Diference"])
    #
    # wait()


    # dx,dy = cv2.getDerivKernels(2,0,21)
    # Dx = dx.reshape((-1,1))*dy.reshape((1,-1))
    #
    # dx, dy = cv2.getDerivKernels(0, 2, 21)
    # Dy = dx.reshape((-1, 1)) * dy.reshape((1, -1))
    #
    # L = Dx+Dy
    #
    #
    # pintaMISingleMPL([visualizeM(Dx),visualizeM(Dy),visualizeM(L)],labels=["Deriv2 x","Deriv2 y","Laplacian of Gaussian"])

    # nimg = imgbw[::4,::4]
    # nimg2 = np.zeros(imgbw.shape)
    # nimg2[::4,::4] = nimg
    # k = cv2.getGaussianKernel(15,-1)
    #
    # bimg = cv2.pyrDown(imgbw)
    # bimg2 = np.zeros(imgbw.shape)
    # bimg2[::4,::4] = cv2.pyrDown(bimg)
    #
    #
    #
    # dimg1 = 16*cv2.sepFilter2D(nimg2,-1,k,k)
    # dimg2 = 16*cv2.sepFilter2D(bimg2,-1,k,k)
    #
    # pintaMISingleMPL([dimg1,dimg2],labels=["No smoothing","Smoothed"],title="Downsampled twice and rescaled")

    k = cv2.getGaussianKernel(27,3)
    G = k*k.reshape((1,-1))
    d = np.array([[-1,0,1]])
    Gx = cv2.filter2D(G,-1,d.transpose())
    Gxx = cv2.filter2D(Gx,-1,d.transpose())


    pintaMISingleMPL([visualizeM(G),visualizeM(Gx),visualizeM(Gxx)],labels=["G","G'/x","G''/xx"])

if __name__ == "__main__":
    main()
