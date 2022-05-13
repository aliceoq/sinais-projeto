import glob
import cv2
import imutils

def main():
    img_list = glob.iglob("images/*")
    for name in img_list:
        img = cv2.imread(name, cv2.IMREAD_COLOR)
        img = imutils.resize(img, width=384)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow(name + ' redimensionada', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
