'''Normando de Campos Amazonas Filho, 11561949
Image Enhancement, SCC0251_Turma01_1Sem_2020_ET
Short Assignment 2: Image Restoration
https://github.com/normandoamazonas/ShortAssignment3'''


import numpy as np
import imageio as imageio
import matplotlib as mpl
from skimage import morphology



'''Parameters input
colors_noise.png
2
1

'''   
filename = str(input()).rstrip()
input_img = imageio.imread(filename)
k = int(input()) # 0<= T <=1 threshold
option = int(input())



def myOpening(I,mydisk):
    opened_R = morphology.opening(I[:,:,0], mydisk).astype(np.uint8)
    opened_G = morphology.opening(I[:,:,1], mydisk).astype(np.uint8)
    opened_B = morphology.opening(I[:,:,2], mydisk).astype(np.uint8)
    return opened_R,opened_G,opened_B

def Scaling(I):
    Min = np.min(I)
    Max = np.max(I)
    #n,m =I.shape
    im= (I-Min)*(255.0/(Max-Min))
    return im#.astype(int)
def rmse(im,ref): 
    num_ele = im.shape[0]*im.shape[1]
    erro = np.sqrt(np.sum(np.square(np.subtract(im.astype(float),ref.astype(float))))/num_ele)
    return erro

def method_1(input_img,size):
    R_,G_,B_ = myOpening(input_img,morphology.disk(size))
    composicion =  np.array(input_img, copy=True).astype(np.uint32)
    composicion[:,:,0] = R_
    composicion[:,:,1] = G_
    composicion[:,:,2] = B_
    return  composicion

def method_2(input_img):
    #1: convert the input image to HSV
    img_hsv = mpl.colors.rgb_to_hsv(input_img)
    #2: get the H channel and normalize it to the interval 0 - 255
    H = img_hsv[:,:,0]
    H_norm = Scaling(H)

    #3. with the structuring element disk, perform the morphological gradient
    #applying gradient
    dilation = morphology.dilation(H_norm, morphology.disk(k)).astype(np.uint8) #.astype(np.uint8)
    erosion = morphology.erosion(H_norm,morphology.disk(k)).astype(np.uint8)
    gradient = dilation - erosion
    norm_gradient=Scaling(gradient.astype(np.uint8))


    '''
    5. compose a new RGB image having
    • the normalized gradient in the R channel
    • the opening of the normalized H (obtained in step 2) in
    G channel
    • the closing of the normalized H (obtained in step 2) in
    B channel
    '''
    Composed_RGB =  np.array(input_img, copy=True).astype(np.uint32)
    Composed_RGB[:,:,0] = norm_gradient
    Composed_RGB[:,:,1] = morphology.opening(H_norm,morphology.disk(k)).astype(np.uint32)
    Composed_RGB[:,:,2] =  morphology.closing(H_norm,morphology.disk(k)).astype(np.uint32)

    return Composed_RGB

def method_3(input_img):
    apMethod1  = method_1(input_img,2*k)
    apMehod2 = method_2(apMethod1)
    return apMehod2

if option ==1:
    print("%.4f"%rmse(input_img,method_1(input_img,k)))

if (option ==2):
    print("%.4f"%rmse(input_img,method_2(input_img)))

if option ==3:
    print("%.4f"%rmse(input_img,method_3(input_img)))




