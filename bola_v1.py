import numpy as np
import cv2
import imutils
import glob
import easyocr

def sharpFilter(img):
    ker = np.array([
                [-1, -1, -1],
                [-1, 18, -1],
                [-1, -1, -1]])
    ker = (1.0/10.0) * ker
    fliped = cv2.flip(ker, 0)
    #print(fliped)
    img = cv2.filter2D(img,-1,fliped, delta=0, anchor=(2,2))
    return img

# https://akshaysin.github.io/fourier_transform.html#.Yn-do1TMLrc
def fftFilter(img):
    rows, cols = img.shape
    crow,ccol = rows//2 , cols//2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # apply mask and inverse DFT
    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back

def normalizer(img):
    normalizedImg = np.zeros(img.shape)
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    (thresh, blackAndWhiteImage) = cv2.threshold(normalizedImg, 127, 255, cv2.THRESH_BINARY_INV)
    img = blackAndWhiteImage

    # img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

def getcontours(img, img_back, imgRef):
    contours = cv2.findContours(img_back.copy(), 2, 1)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:30]

    for c in contours:
        (x,y),radius = cv2.minEnclosingCircle(c)
        xRect, yRect, w, h = cv2.boundingRect(c)
        if (abs(w - h) < w*0.4):
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(imgRef, center, radius, (0,255,0), 2)    

    return imgRef, img_back

def main():
    show_images = False
    img_list = glob.iglob("images/*")
    for name in img_list:
        img = cv2.imread(name, cv2.IMREAD_COLOR)

        #RESIZE
        img = imutils.resize(img, width=512)
        if show_images: cv2.imshow('Resize', img)

        #COPIA
        img_filter = img.copy()
        img_base = img.copy()

        #NITIDEZ
        img_filter = sharpFilter(img_filter)
        img_filter = sharpFilter(img_filter)
        img_filter = sharpFilter(img_filter)

        #PRETO E BRANCO
        img_filter = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY)
        #MAIS CONTRASTE
        array_alpha = np.array([1.25])
        array_beta = np.array([-50.0])
        cv2.add(img_filter, array_beta, img_filter)                    
        cv2.multiply(img_filter, array_alpha, img_filter)

        #TRANSFORM E NORMALIZANDO
        img_back = fftFilter(img_filter) 
        img_back = normalizer(img_back)

        #ENCONTRANDO CONTORNOS A PARTIR DA IMAGEM COM FILTROS
        resultColored, resultBlack = getcontours(img, img_back, img_base.copy())
        resultBlack = cv2.cvtColor(resultBlack, cv2.COLOR_GRAY2RGB)
        img_and_magnitude = np.concatenate((resultColored, resultBlack), axis=1)
        cv2.imshow('com filtro e transformada', img_and_magnitude)
        cv2.waitKey(0)

        #ENCONTRANDO CONTORNOS A PARTIR DA IMAGEM SEM TRANSFORMADA
        resultColored, resultBlack = getcontours(img, img_filter, img_base.copy())
        resultBlack = cv2.cvtColor(resultBlack, cv2.COLOR_GRAY2RGB)
        img_and_magnitude = np.concatenate((resultColored, resultBlack), axis=1)
        cv2.imshow('com filtro e sem transformada', img_and_magnitude)
        cv2.waitKey(0)

        #ENCONTRANDO CONTORNOS A PARTIR DA IMAGEM SEM FILTROS (só preta e branca)
        resultColored, resultBlack = getcontours(img, cv2.cvtColor(img_base.copy(), cv2.COLOR_BGR2GRAY), img_base.copy())
        resultBlack = cv2.cvtColor(resultBlack, cv2.COLOR_GRAY2RGB)
        img_and_magnitude = np.concatenate((resultColored, resultBlack), axis=1)
        cv2.imshow('sem filtro', img_and_magnitude)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
