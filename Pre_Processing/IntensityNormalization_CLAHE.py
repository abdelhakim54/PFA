import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def normalize(img):
   
    arr = np.array(img)
    arr = arr.astype('float')
   
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr





# normalize(Image.open('1.png').convert('RGBA'))


# #img = Image.open('1.png').convert('RGBA')
# #arr = np.array(img)
# #new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGBA')
# #new_img.save('normalized.png')


# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(img)
# ax[0].set_title('Original Image')
# ax[0].grid(False)
# ax[1].imshow(new_img)
# ax[1].set_title('normalized Image')
# ax[1].grid(True)
# plt.show()

def CLAHE(image):
    # Converting image to LAB Color model
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

#cv2.imshow("Result",CLAHE(cv2.imread('1.png')))
#img=CLAHE(cv2.imread('1.png'))
#cv2.imwrite('CLAHE.png',img)
#cv2.waitKey()
