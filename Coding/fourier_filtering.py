import cv2
import numpy as np


#Part C
img = cv2.imread("Noisy_image.png", cv2.IMREAD_COLOR)
imageHeight = len(img)
imageWidth = len(img[0])

imgGray = np.zeros([imageHeight, imageWidth], dtype=np.uint8)
for i in range(imageHeight):
  for j in range(imageWidth):
    imgGray[i, j] =  int((0.299+img[i][j][0] + 0.587+img[i][j][1] + 0.114+img[i][j][2]) / 3)


imageHeight = len(imgGray)
imageWidth = len(imgGray[0])

fourierTrans = cv2.dft(np.float32(imgGray), flags = cv2.DFT_COMPLEX_OUTPUT)
fourierTransShift = np.fft.fftshift(fourierTrans)

magSpecFT = 20*np.log(cv2.magnitude(fourierTransShift[:,:,0],fourierTransShift[:,:,1]))
cv2.imwrite("converted_fourier.png", magSpecFT)

heightFT = len(fourierTransShift)
widthFT = len(fourierTransShift[0])

PadDft_shift = np.zeros([heightFT+2, widthFT+2,2], dtype=np.float32)

for i in range(imageHeight):
    for j in range(imageWidth):
        if i == 0 or j == 0:
            PadDft_shift[i,j] = 0
        else:
            PadDft_shift[i,j] = fourierTransShift[i-1,j-1]


midRow = int(imageHeight/2) 
midCol = int(imageWidth/2)

filter = np.zeros((imageHeight,imageWidth,2), dtype=np.uint8)
grayBack = np.zeros((imageHeight,imageWidth,2), dtype=np.uint8)

filter[midRow-30:midRow+30, midCol-30:midCol+30] = 1

fshift = fourierTransShift * filter

f_ishift = np.fft.ifftshift(fshift)
grayBack = cv2.idft(f_ishift)
grayBack = cv2.magnitude(grayBack[:,:,0],grayBack[:,:,1])

max=0
for i in range(imageHeight):
    for j in range(imageWidth):
        if grayBack[i,j] > max:
            max = grayBack[i,j]

grayBack = grayBack/max
grayBack = grayBack * 255

cv2.imwrite("gaussian_fourier.png", grayBack)