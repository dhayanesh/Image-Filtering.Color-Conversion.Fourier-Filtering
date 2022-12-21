import cv2
import numpy as np
import math

#Part A : 1. RGB to HSV
img = cv2.imread("Lenna.png", cv2.IMREAD_COLOR)
imageHeight = len(img)
imageWidth = len(img[0])

hsvImg = np.zeros([imageHeight,imageWidth,3],dtype=np.uint8)

for i in range(imageHeight):
    for j in range(imageWidth):
        rgb = img[i][j]
        B = float(rgb[0])/float(255)
        G = float(rgb[1])/float(255)
        R = float(rgb[2])/float(255)

        V = max(R,G,B)
        if V != 0:
            S = (V - min(R,G,B))/V
        else:
            S = 0

        if V == R:
            H = 60*(G - B)/(V - min(R,G,B))
        elif V == G:
            H = 120 + 60*(B - R)/(V - min(R,G,B))
        elif V == B:
            H = 240 + 60*(R - G)/(V - min(R,G,B))
        elif R == G and G == B:
            H = 0
        
        if H < 0:
            H = H + 360
        H = H/2
        S = S * 255
        V = V * 255
        hsvImg[i,j] = [H,S,V]


filename = 'hsv_image_1.png'
cv2.imwrite(filename, hsvImg)

#Part A : 2. RGB to HSV
for i in range(imageHeight):
    for j in range(imageWidth):
        rgb = img[i][j]
        
        B = (rgb[0])/float(255)
        G = (rgb[1])/float(255)
        R = (rgb[2])/float(255)

        V = (R+G+B)/3.0

        S = 1.0 - ((3.0/(R+G+B)) * min(R,G,B))
        nr = 0.5 * ((R - G)+(R - B)) 
        dr = ((R - G)**2) + (R - B)*(G - B)
        dr = math.sqrt(dr)
        val = nr/dr
        theta = math.acos(val) * (180.0 / math.pi)

        if B <= G:
            H = theta
        elif B > G:
            H = 360.0 - theta

        H = H / 2
        S = S * 255.0
        V = V * 255.0
        
        hsvImg[i,j] = [H,S,V]

filename = 'hsv_image_2.png'
cv2.imwrite(filename, hsvImg)

#Part A : 3. RGB to CMYK
cmykImg = np.zeros([imageHeight,imageWidth,4],dtype=np.uint8)

for i in range(imageHeight):
    for j in range(imageWidth):
        rgb = img[i][j]
        B = rgb[0]/255.0
        G = rgb[1]/255.0
        R = rgb[2]/255.0

        C = 1 - R
        M = 1 - G
        Y = 1 - B
        K = min(C,M,Y)
        if K == 1:
            C, M, Y = 0 , 0, 0
        else:
            C = (C-K)/(1-K) 
            M = (M-K)/(1-K) 
            Y = (Y-K)/(1-K)
        R = C
        G = M
        B = Y
        A = K

        R = R * 255
        G = G * 255
        B = B * 255
        A = A * 255
        cmykImg[i,j] = [R,G,B,A]

#cmykImg = cmykImg.astype(np.float32)/255
filename = 'cmyk_image.png'
cv2.imwrite(filename, cmykImg)

#Part A : 4. RGB to Lab
labImg = np.zeros([imageHeight,imageWidth,3],dtype=np.uint8)

def calFunction(var):
    if  var > 0.008856:
        return (var ** (1/3.0))
    else:
        return ((7.787*var) + 16.0/116.0)


for i in range(imageHeight):
    for j in range(imageWidth):
        rgb = img[i][j]
        B = float(rgb[0])/float(255)
        G = float(rgb[1])/float(255)
        R = float(rgb[2])/float(255)   
        matA = [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
        matB = [R, G, B]
        
        XYZ  = np.dot(matA,matB)

        X = XYZ[0] / 0.950456
        Z = XYZ[2] / 1.088754
        Y = XYZ[1]
        if Y > 0.008856:
            L = 116.0 * (Y ** (1/3.0)) - 16.0
        elif Y <= 0.008856:
            L = 903.3 * Y   

        a = 500.0 * (calFunction (X) - calFunction (Y)) + 0
        b = 200.0 * (calFunction (Y) - calFunction (Z)) + 0

        L = L * 255/100 
        a = a + 128.0
        b = b + 128.0

        labImg[i,j] = [L,a,b]
filename = 'lab_image.png'
cv2.imwrite(filename, labImg) 

test = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
cv2.imwrite("test.png", test) 

#cv2.waitKey(0)
#cv2.destroyAllWindows()    


