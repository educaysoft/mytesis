import numpy
import math
import cv2
import sys
if len(sys.argv)==3:
    original=cv2.imread(sys.argv[1],0)
    contrast = cv2.imread(sys.argv[2],0)
    def psnr(img1, img2):
        mse = numpy.sum((img1.astype("float") - img2.astype("float")) ** 2)
	mse /= float(img1.shape[0] * img1.shape[1])
        if mse ==0:
            return 100
        PIXEL_MAX = 255.0
        return [20*math.log10(PIXEL_MAX/math.sqrt(mse)),mse]

    [p,m]=psnr(original,contrast)
    print("\n")
    print("PSNR:"+str(p))
    print("MSE:"+str(m))
else:
    print("Debe ingresar dos nombre de imagenes")

