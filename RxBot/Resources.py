from Settings import *
from Initialize import *
import pyautogui
import mss
import mss.tools
#import pytesseract
from PIL import Image, ImageOps
import cv2
import numpy
from xlutils.copy import copy
import xlrd
import re
import cv2
import numpy as np
import pyscreeze

# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# pyautogui.FAILSAFE = False  # Might cause nuclear apocalypse

res = settings["RESOLUTION MODIFIER"] / 100

scrollToLineUpBottomDistance = -240
scrollToMoveUpOneBarDistance = -54
timesToScrollUp = 8
portraitOffset = -40

def cvToPil(cvImg):
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    pilImg = Image.fromarray(cvImg)
    del cvImg
    return pilImg


def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)





class resourcesClass:
    def __init__(self):
        self.width, self.height = pyautogui.size()
        self.userText = None
        self.IdText = None
        self.buyInText = None
        self.profitText = None
        self.cachedIdImage = None
        self.handsText = None
        self.lastLeaderboardHandCounts = []
        self.currentLeaderboardHandCounts = []
        self.wipeNextScan = False
        self.oldTempCache = []

    def holdKey(self, key, duration):
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)


    def findImageOnScreen(self, imgName, confidence):
        imageLocation = pyautogui.locateOnScreen("Resources/%s" % imgName, confidence=confidence)
        if not imageLocation:
            return False
        #print("Image found at " + str(imageLocation))
        return imageLocation

    def moveMouseToLocation(self, imageLocation):
        x, y = pyautogui.center(imageLocation)
        pyautogui.moveTo(x, y, 0.3)

    # def imgToText(self, img):
    #     text = pytesseract.image_to_string(img, config='--psm 10 --oem 3').replace("-", "")
    #     #print("OCR TEXT: \n" + text + "\n")
    #     return text.strip()

    def screenshotRegion(self, left, top, width, height, invert=None, filter=None):
        if settings["ALTERNATIVE SCREENSHOT"]:
            with mss.mss() as sct:
                # The screen part to capture
                region = {'top': top, 'left': left, 'width': width, 'height': height}

                # Grab the data
                sctimg = sct.grab(region)
                img = Image.frombytes("RGB", sctimg.size, sctimg.bgra, "raw", "BGRX")
        else:
            img = pyautogui.screenshot(region=(left, top, width, height))
        newImg = img

        # Upscale
        if filter:
            imgSize = img.size
            img = img.resize((imgSize[0] * 2, imgSize[1] * 2), resample=Image.BOX)

            newImg = img

        if invert:
            img = ImageOps.invert(img)

        if filter == "Normal":
            img = change_contrast(img, 142)

            cvImg = numpy.array(img)
            cvImg = cvImg[:, :, ::-1].copy()

            gray = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
            revisedCvImg = cv2.fastNlMeansDenoising(gray, cvImg, 67.0, 7, 21)
            (thresh, blackAndWhiteImage) = cv2.threshold(revisedCvImg, (143 + settings["IMAGE OFFSET"]), 255, cv2.THRESH_BINARY)
            newImg = cvToPil(blackAndWhiteImage)

        if filter == "Hands":
            img = change_contrast(img, 142)

            cvImg = numpy.array(img)
            cvImg = cvImg[:, :, ::-1].copy()

            gray = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
            revisedCvImg = cv2.fastNlMeansDenoising(gray, cvImg, 67.0, 7, 21)
            (thresh, blackAndWhiteImage) = cv2.threshold(revisedCvImg, (143 + settings["HANDS OFFSET"]), 255, cv2.THRESH_BINARY)
            newImg = cvToPil(blackAndWhiteImage)

        if filter == "ID":
            img = change_contrast(img, 40)

            cvImg = numpy.array(img)
            cvImg = cvImg[:, :, ::-1].copy()

            gray = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            revisedCvImg = cv2.fastNlMeansDenoising(gray, blackAndWhiteImage, 35.0, 7, 5)
            (thresh, blackAndWhiteImage) = cv2.threshold(revisedCvImg, (220 + settings["ID IMAGE OFFSET"]), 255, cv2.THRESH_BINARY)
            newImg = cvToPil(blackAndWhiteImage)

        #newImg.show()



        # thresh = 170
        # fn = lambda x: 255 if x > thresh else 0
        # img = img.convert('L').point(fn, mode='1')


        if settings["DEBUG SHOW IMAGE"]:
            newImg.show()
            print("Showed image, waiting for it to be closed or moved.")
            time.sleep(1)

        #newImg.show()
        return newImg


    def scrollDown(self):
        pyautogui.moveTo(int(resources.width / 2), int(resources.height / 2), 0.3)
        pyautogui.drag(0, int(scrollToMoveUpOneBarDistance * res), 0.8, button="left")
        time.sleep(1.8)


    def scrollUp(self):
        pyautogui.moveTo(int(resources.width / 2), int(resources.height / 2), 0.3)
        pyautogui.drag(0, 400, 0.8, button="left")
        time.sleep(1)



def detect_target_image():
    screenshot =  resources.screenshotRegion(int(949 * res), int(83 * res), int(800 * res), int(1000 * res))
    img = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
    # read bananas image template
    template = cv2.imread('Resources/farLeft.png', cv2.IMREAD_UNCHANGED)

    # extract bananas base image and alpha channel and make alpha 3 channels
    base = template[:, :, 0:3]
    alpha = template[:, :, 3]
    alpha = cv2.merge([alpha, alpha, alpha])

    # do masked template matching and save correlation image
    correlation = cv2.matchTemplate(img, base, cv2.TM_CCORR_NORMED, mask=alpha)

    # set threshold and get all matches
    threshhold = 4
    loc = np.where(correlation >= threshhold)
    for pt in zip(*loc[::-1]):
        print(pt)


def resetStartAgain():
    pass


def normalHit():
    print(res)
    print("Calculating time for hit, keep it as steady as you can.")
    prevTime = datetime.datetime.now()
    logged = False
    lapCount = 0
    while True:
        # First, time the counter for full time between rotations
        if pyscreeze.pixelMatchesColor(int(1163 * res), int(656 * res), (236, 203, 57), tolerance=50 + settings["ACCURACY MODIFIER"]):  # ALL OF THIS IS JUST TO GET THE TIMING
            lapCount += 1
            if lapCount > 1:

                if not logged:
                    prevTime = datetime.datetime.now()
                    logged = True

                elif logged:
                    timeDelta = (datetime.datetime.now() - prevTime)
                    delay_in_ms = timeDelta.total_seconds() * 1000
                    if delay_in_ms > 600:  # GOT TIME
                        logged = False
                        while True:  # GOT TIMING, DETECT POINT THEN WAIT CALCULATED TIME
                            if pyscreeze.pixelMatchesColor(int(1182 * res), int(611 * res), (236, 203, 57), tolerance=54 + settings["ACCURACY MODIFIER"]):
                                timeToSleep = ((delay_in_ms / 1000) / settings["DELAY MODIFIER"])
                                #print(timeToSleep)
                                time.sleep(timeToSleep)
                                pyautogui.click()
                                print("HIT!")
                                return


def putt():
    print("Calculating time for putt, keep it as steady as you can.")
    prevTime = datetime.datetime.now()
    logged = False
    lapCount = 0
    while True:
        # First, time the counter for full time between rotations
        if pyscreeze.pixelMatchesColor(int(1206 * res), int(747 * res), (236, 203, 57), tolerance=53 + settings["ACCURACY MODIFIER"]):  # ALL OF THIS IS JUST TO GET THE TIMING
            lapCount += 1
            if lapCount > 1:

                if not logged:
                    prevTime = datetime.datetime.now()
                    logged = True

                elif logged:
                    timeDelta = (datetime.datetime.now() - prevTime)
                    delay_in_ms = timeDelta.total_seconds() * 1000
                    if delay_in_ms > 600:  # GOT TIME
                        logged = False
                        print("AAA")
                        while True:  # GOT TIMING, DETECT POINT THEN WAIT CALCULATED TIME
                            if pyscreeze.pixelMatchesColor(int(1235 * res), int(717 * res), (236, 203, 57), tolerance=54 + settings["ACCURACY MODIFIER"]):
                                timeToSleep = ((delay_in_ms / 1000) / settings["DELAY MODIFIER"])
                                #print(timeToSleep)
                                time.sleep(timeToSleep)
                                pyautogui.click()
                                print("PUTT")
                                return


def startRequest():  # PUTT - 1274, 749 (249, 184, 60)
    print("Detecting hit or putt arrow")
    while True:
        im = resources.screenshotRegion(0, 0, 2560, 1440)
        if pyautogui.locate('Resources/arrowFind.png', im, confidence=0.75):
            print("FOUND ARROW FOR NORMAL HIT")
            normalHit()

            print("Hit should have been successful, waiting 5s before searching again")
            time.sleep(5)
        elif pyautogui.locate('Resources/putt.png', im, confidence=0.75):
            putt()

            print("Hit should have been successful, waiting 5s before searching again")
            time.sleep(5)






                        # 1206 747


                        # 1235 717


resources = resourcesClass()

