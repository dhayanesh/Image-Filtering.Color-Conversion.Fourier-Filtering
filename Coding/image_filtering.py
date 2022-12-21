import cv2
import numpy as np

#Part B
img = cv2.imread("Noisy_image.png", cv2.IMREAD_COLOR)
imageHeight = len(img)
imageWidth = len(img[0])

imgGray = np.zeros([imageHeight, imageWidth], dtype=np.uint8)
for i in range(imageHeight):
  for j in range(imageWidth):
    imgGray[i, j] =  int((0.299+img[i][j][0] + 0.587+img[i][j][1] + 0.114+img[i][j][2]) / 3)


padImage = np.zeros([imageHeight+2, imageWidth+2], dtype=np.uint8)
for i in range(imageHeight):
    for j in range(imageWidth):
        if i == 0 or j == 0:
            padImage[i,j] = 0
        else:
            padImage[i,j] = imgGray[i-1,j-1]

#convolution
convFilter = (1/9) * np.array([[1,1,1],[1,1,1],[1,1,1]])
convFilter = np.flip(convFilter,0)
convFilter = np.flip(convFilter,1)
convImage = np.zeros([imageHeight, imageWidth], dtype=np.uint8)

for i in range(imageHeight):
    for j in range(imageWidth):
        matThree = padImage[i:i+3, j:j+3]
        matMul  = np.multiply(matThree, convFilter)
        convImage[i,j] = np.sum(matMul)
cv2.imwrite("convolved_image.png", convImage)

#average
avgFilter = (1/9) * np.array([[1,1,1],[1,1,1],[1,1,1]])
avgImage = np.zeros([imageHeight, imageWidth], dtype=np.uint8)

for i in range(imageHeight):
    for j in range(imageWidth):
        matThree = padImage[i:i+3, j:j+3] 
        matMul  = np.multiply(matThree, avgFilter)
        avgImage[i,j] = np.sum(matMul)
cv2.imwrite("average_image.png", avgImage)

#gaussian
gausFilter = 1/16 * np.array([[1,2,1],[2,4,2], [1,2,1]])
gausImage = np.zeros([imageHeight, imageWidth], dtype=np.uint8)

for i in range(imageHeight):
    for j in range(imageWidth):
        matThree = padImage[i:i+3, j:j+3]
        matMul  = np.multiply(matThree, gausFilter)
        gausImage[i,j] = np.sum(matMul)
cv2.imwrite("gaussian_image.png", gausImage)

#median
newPadImage = np.zeros([imageHeight+4, imageWidth+4], dtype=np.uint8)
for i in range(imageHeight):
    for j in range(imageWidth):
        if i == 0 or j == 0:
            newPadImage[i,j] = 0
        else:
            newPadImage[i,j] = imgGray[i-2,j-2]

medianImage = np.zeros([imageHeight, imageWidth], dtype=np.uint8)
for i in range(imageHeight):
  for j in range(imageWidth):
    matThree = newPadImage[i:i+5, j:j+5]
    medianImage[i,j] = np.median(matThree)
cv2.imwrite("median_image.png", medianImage)


catImg = cv2.imread("Uexposed.png", cv2.IMREAD_COLOR)
imageHeight2 = len(catImg)
imageWidth2 = len(catImg[0])

shinyCat = np.zeros([imageHeight2,imageWidth2,3],dtype=np.uint8)

"""
for i in range(imageHeight2):
  for j in range(imageWidth2):
    rgb = catImg[i][j]
    if (rgb[2] * 2) < 255:
        shinyCat[i, j, 2]= rgb[2] + 50 
    else: 
        shinyCat[i, j, 2] = 255

    if (rgb[1] * 2) < 255:
        shinyCat[i, j, 1]= rgb[1] + 50 
    else: 
        shinyCat[i, j, 1] = 255

    if (rgb[0] * 2) < 255:
        shinyCat[i, j, 0]= rgb[0] + 50 
    else: 
        shinyCat[i, j, 0] = 255"""

#as cv2.addWeighted is allowed
shinyCat = cv2.addWeighted(catImg, 2.75, catImg, 2.75, 0.0)
     
cv2.imwrite("adjusted_image.png", shinyCat)
