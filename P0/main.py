import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys, os

CURRENT_PATH = os.getcwd()

# Ejercicio 1
# In:
#   filename: File path
#   flagcolor: cv2 color flag [0 Grayscale, 1 Color]
# Out:
#   Image matrix (H,W,C[BGR])
def leeimagen(filename,flagcolor):
    M = cv2.imread(filename,flagcolor)
    return M

# Ejercicio 2
# In:
#   im: Image matrix (H,W,C[BGR])
#   window_name: Window's title
#   wait: call cv2.waitKey()
def pintaI(im,window_name="window",wait=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name,im)
    if wait:
        cv2.waitKey(0)

# Ejercicio 3
# In:
#   vim: Image matrix (H,W,C[BGR]) iterable [list, tuple, np.array...]
def pintaMI(vim):
    for i,im in enumerate(vim):
        pintaI(im, window_name= "Image{}".format(i+1), wait=False)
    cv2.waitKey(0)

# Ejercicio 4
# In:
#   im: Image matrix (H,W,C[BGR])
#   lpoints: N points list [N x 2](y,x). List or np.array
#   color: value for Grayscale or (B,G,R) for Color
# Out:
#   Returns nothing, it modifies the original image
def modificaI(im,lpoints,color):
    # Assert color shape is correct for image type
    assert((len(im.shape) == 2 and color < 255) or im.shape[2] == color.size)
    # Transform lpoints to np matrix if needed
    lpoints = np.array(lpoints)
    # Assert correct shape
    assert (lpoints.shape[1] == 2)
    # Transform transpose to tuple (Numpy likes it that way)
    p = tuple(np.transpose(lpoints))
    # Assign color to points
    im[p] = color


#####
#
#   Ejercicio 5 (Numpy+OpenCV version)
#
#####

# Add title to image with optional left/bottom offset
# In:
#   im: Image matrix (H,W,C[BGR])
#   title: Title string
#   textcolor: (B,G,R) color value
#   offset: distance to bottom left border
# Out:
#   Returns new image
def addTitle(im,title,textcolor,offset=10):
    return cv2.putText(im, title, (offset, im.shape[0]-offset), cv2.FONT_HERSHEY_PLAIN, 1, textcolor)

# Transforms every image to same height/width
# In:
#   vim: Image matrix (H,W,C[BGR]) iterable [list, tuple, np.array...]
#   imlabels: [String] List of labels
#   labelsize:  Bottom padding for text [Default: 0]
#   textcolor: (B,G,R) color value for text [Default: Red]
#   labelcolor: (B,G,R) color value for padding [Default: Red]
# Out:
#   Returns list of images with same height,width and optionally labels
def joinImg(vim,imlabels=None,labelsize=0,textcolor=(0,0,255),labelcolor=(255,255,255)):
    # Transform every image to BGR
    vim = np.array([
        cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if len(im.shape) == 2
        else im
        for im in vim])
    # Get max image width, and max image height
    siz = np.max([im.shape[:2] for im in vim],0)

    # Transform every image and add it to vimborder
    vimborder = []
    for im in vim:
        # Get border size [H,W]
        border = (siz - im.shape[:2])//2
        # Fill a matrix with zeroes
        M = np.zeros((siz[0]+labelsize,siz[1],3),dtype=np.uint8)
        # Replace center with image
        M[border[0]:border[0]+im.shape[0],border[1]:border[1]+im.shape[1],:] = im
        # If labelsize fill with labelcolor
        if labelsize != 0:
            M[-labelsize:,:,:] = labelcolor

        vimborder.append(M)

    # Add labels if needed
    if imlabels is not None:
        # Assert both list have the same length
        assert (len(vimborder) == len(imlabels))
        vimborder = [addTitle(im,title,textcolor) for im,title in zip(vimborder,imlabels)]
    return vimborder

# Prints every image in a single window
# In:
#   vim: Image matrix (H,W,C[BGR]) iterable [list, tuple, np.array...]
#   imlabels: [String] List of labels
#   labelsize:  Bottom padding for text [Default: 0]
#   textcolor: (B,G,R) color value for text [Default: White]

def pintaMISingle(vim,imlabels=None,labelsize=0,textcolor=(255,255,255)):
    vim = joinImg(vim,imlabels,labelsize,textcolor)
    hor = np.hstack(tuple(vim))
    cv2.imshow("window3",hor)
    cv2.waitKey(0)


#####
#
#   Ejercicio 5 (Matplotlib version)
#
#####

# Print every image in a single window
# In:
#   vim: Image matrix (H,W,C[BGR]) iterable [list, tuple, np.array...]
#   labels: [String] List of labels
#   cols: Number of columns
def pintaMISingleMPL(vim,labels=None,cols=3):
    # Transform every image to RGB
    vim = np.array([
        cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        if len(im.shape) == 2
        else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        for im in vim])

    # Calculate ncols and nrows
    import math
    cols = min(cols,len(vim))
    rows = math.ceil(len(vim)/cols)

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

def main():
    p = os.path.join(CURRENT_PATH,"lena2.jpg")

    # Ejercicio 1
    M1 = leeimagen(p,1)
    M0 = leeimagen(p, 0)

    # Create a list of randomly scaled images
    import random
    scaled_images = []
    for i in range(6):
        dsize = np.random.randint(150,300)
        dsize = (dsize,dsize)
        if i == 0:
            im = M0
        elif i == 1:
            im = M1
        else:
            im = random.choice([M0,M1])
        scaled_images.append(
            cv2.resize(im,dsize)
        )

    # Ejercicio 2
    pintaI(scaled_images[0])
    cv2.destroyAllWindows()

    # Ejercicio 3
    pintaMI(scaled_images)
    cv2.destroyAllWindows()

    ## Ejercicio 4 Grayscale
    # Get 100 random coordinates
    coords = np.random.randint(0,M0.shape[0],(100,2))
    # Get random color
    color = np.random.randint(0,255)
    modificaI(M0,coords,color)
    pintaI(M0)
    cv2.destroyAllWindows()

    ## Ejercicio 4 Color
    # Get 100 random coordinates
    coords = np.random.randint(0,M1.shape[0],(100,2))
    # Get random color
    color = np.random.randint(0,255,(3))
    modificaI(M1,coords,color)
    pintaI(M1)
    cv2.destroyAllWindows()

    ## Ejercicio 5 Numpy+OpenCV
    # Without title
    pintaMISingle(scaled_images)
    cv2.destroyAllWindows()

    # With overlay label
    pintaMISingle(scaled_images, ["Image {}".format(i+1) for i in range(len(scaled_images))])
    cv2.destroyAllWindows()

    # With padding for labels
    pintaMISingle(
        scaled_images,
        ["Image {}".format(i+1) for i in range(len(scaled_images))],
        labelsize=30,
        textcolor=(0,0,0))
    cv2.destroyAllWindows()

    ## Ejercicio 5 Matplotlib
    pintaMISingleMPL(scaled_images, ["Image {}".format(i+1) for i in range(len(scaled_images))])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    CURRENT_PATH = os.path.split(os.path.abspath(sys.argv[0]))[0]
    main()