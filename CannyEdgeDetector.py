from PIL import Image
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
# import pdb
from skimage import filters

parser = argparse.ArgumentParser()
parser.add_argument('param_file', type=Path)
args = parser.parse_args()
params = json.load(args.param_file.open())


if params["saveoutput"] == "True":
    Path("./result").mkdir(exist_ok=True)

image_file = params['image']
# Load image into variable and display it
lena = Image.open(image_file)

plt.figure()
plt.subplot(131)
plt.imshow(lena, cmap = plt.get_cmap('gray'))
plt.title('original image')
lena = np.asarray(lena)


# Convert color image to grayscale to help extraction of edges and plot it
lena_gray = np.dot(lena[...,:3], [0.299, 0.587, 0.114])
plt.subplot(132)
plt.imshow(lena_gray, cmap = plt.get_cmap('gray'))
plt.title('Step1: Grey image')
# Blur the grayscale image so that only important edges are extracted and the noisy ones ignored

lena_gray_blurred = ndimage.gaussian_filter(lena_gray, sigma=params['sigma']) # Note that the value of sigma is image specific so please tune it
plt.subplot(133)
plt.imshow(lena_gray_blurred, cmap = plt.get_cmap('gray'))
plt.title('Step2: Blur image')
plt.tight_layout()
if params["saveoutput"] == "True":
    plt.savefig("./result/pic1.png")
# pdb.set_trace()
# Apply Sobel Filter using the convolution operation
def SobelFilter(img, direction):
    if(direction == 'x'):
        Gx = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])
        Res = ndimage.convolve(img, Gx)
        #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
    if(direction == 'y'):
        Gy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
        Res = ndimage.convolve(img, Gy)
        #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)

    return Res


# Normalize the pixel array, so that values are <= 1
def Normalize(img):
    #img = np.multiply(img, 255 / np.max(img))
    img = img/np.max(img)
    return img


# Apply Sobel Filter in X direction
gx = SobelFilter(lena_gray_blurred, 'x')
gx = Normalize(gx)
plt.figure()
plt.subplot(221)
plt.imshow(gx, cmap = plt.get_cmap('gray'))
plt.title('Sobel Filter in X direction')

# Apply Sobel Filter in Y direction
gy = SobelFilter(lena_gray_blurred, 'y')
gy = Normalize(gy)
plt.subplot(222)
plt.imshow(gy, cmap = plt.get_cmap('gray'))
plt.title('Sobel Filter in Y direction')


# Calculate the magnitude of the gradients obtained
Mag = np.hypot(gx,gy)
Mag = Normalize(Mag)
plt.subplot(223)
plt.imshow(Mag, cmap = plt.get_cmap('gray'))
plt.title('Magnitude of the gradients')

# Calculate direction of the gradients
Gradient = np.degrees(np.arctan2(gy,gx))

plt.subplot(224)
plt.imshow(Gradient)
plt.colorbar()
plt.title('Direction of the gradients')
plt.tight_layout()
if params["saveoutput"] == "True":
    plt.savefig("./result/pic2.png")

# This is also non-maxima suppression but without interpolation i.e. the pixel closest to the gradient direction is used as the estimate
def NonMaxSupWithoutInterpol(Gmag, Grad):
    NMS = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS


# Get the Non-Max Suppressed output
NMS_1 = NonMaxSupWithoutInterpol(Mag, Gradient)
NMS_1 = Normalize(NMS_1)

plt.figure()
plt.imshow(NMS_1, cmap = plt.get_cmap('gray'))
plt.title('Non-Max Suppressed output')


highThresholdRatio = params["highThresholdRatio"]
lowThresholdRatio = params["lowThresholdRatio"]
Img = np.copy(NMS_1)
h = int(Img.shape[0])
w = int(Img.shape[1])
highThreshold = np.max(Img) * highThresholdRatio
lowThreshold = highThreshold * lowThresholdRatio
print('highThreshold : %f' % highThreshold)
print('lowThreshold : %f' % lowThreshold)
G_h = (Img > highThreshold).astype('uint8')*255
G_l = (Img > lowThreshold).astype('uint8')*255
G_mid = G_l - G_h
# img_h = Image.fromarray(G_h)
# img_l = Image.fromarray(G_l)
# img_h.show() # strong edge
# img_l.show() # strong and weak edge
# img_mid = Image.fromarray(G_mid)
# img_mid.show() # weak edge
plt.figure()
plt.subplot(131)
plt.imshow(G_h, cmap = plt.get_cmap('gray'))
plt.title('Strong edges')
plt.subplot(132)
plt.imshow(G_mid, cmap = plt.get_cmap('gray'))
plt.title('Weak edges')
plt.subplot(133)
plt.imshow(G_l, cmap = plt.get_cmap('gray'))
plt.title('Strong and weak edges')
plt.tight_layout()
if params["saveoutput"] == "True":
    plt.savefig("./result/pic3.png")

if params["uselibrary"] == "True":
    # pdb.set_trace()
    # use skimage apply_hysteresis_threshold function
    hyst = filters.apply_hysteresis_threshold(NMS_1, lowThreshold, highThreshold)
    plt.figure()
    plt.imshow(hyst, cmap = plt.get_cmap('gray'))
    plt.title('hyst from skimage')
else:
    # my implementation
    G_new = np.copy(G_h)
    num = 1
    num_old = 0
    while (num != num_old):
        for i in range(1,h-1):
            for j in range(1,w-1):
                if G_mid[i,j] == 255 and any(255 in e for e in G_new[i-1:i+2,j-1:j+2]):
                    G_new[i,j] = 255
        num_old = num
        num = np.sum(G_new == 255)
        print('number of pixel in the strong edge %i' % num)


    plt.figure()
    plt.imshow(G_new, cmap = plt.get_cmap('gray'))
    plt.title('Final image')
if params["saveoutput"] == "True":
    plt.savefig("./result/Final image.png")

plt.show()
