import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import pytesseract
import re
from dateparser.search import search_dates
from dateutil.parser import parse
from pytesseract import Output

def simple_gray_image(img):
    try:
        image = cv2.imread(img)
    except:
        image = img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

def threshold_gray_image(gray_img):

    threshold_gray = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return threshold_gray


def image_smoothening(img):
    BINARY_THREHOLD = 180

    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th3

def remove_noise_and_smooth(gray_img):

    img = gray_img

    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)

    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    img = image_smoothening(img)

    or_image = cv2.bitwise_or(img, closing)

    return or_image

def edged_find_contours_image(image):
    # convert the image to grayscale, blur it, and find edges
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # show the original image and the edge detected image
    #cv2.imshow("Image", image)
    #cv2.imshow("Edged", edged)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    try:
         # show the contour (outline) of the piece of paper

        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        #cv2.imshow("Outline", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    except:
        #print('No perfect contour found so no birds eye view')
        #print('\nNo date is extracted')
        screenCnt = 0

    return screenCnt, ratio


def order_points(pts):
    #initialzie a list of coordinates that will be ordered
    #such that the first entry in the list is the top-left,
    #the second entry is the top-right, the third is the
    #bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def birds_eye_view(img):

    try:
        image = cv2.imread(img)
    except:
        image = img

    screenCnt, ratio = edged_find_contours_image(img)

    orig = img.copy()

    try:
        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

#         #convert the warped image to grayscale, then threshold it
#         #to give it that 'black and white' paper effect
#         warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#         T = threshold_local(warped, 5, offset = 10, method = "gaussian")#
#         warped = (warped > T).astype("uint8") * 255

        # show the original and scanned images
        #print("STEP 3: Apply perspective transform")
        #print('Got the counter so trying to extract date with birds eye view' )
        #cv2.imshow("Original", imutils.resize(orig, height = 650))
        #cv2.imshow("Scanned", imutils.resize(warped, height = 650))
        #cv2.waitKey(0)

    except:
        warped = np.zeros(5)

    return warped


def text_ocr(img):

    text = pytesseract.image_to_string(img)

    return text

def date_parser(text):
    try:
        dates = search_dates(text)
    except:
        dates = []
        #print('Date is not present or not able to extract')
    return dates


def dateutil_parser_check(date_parser_dates):

    valid_dateutil_parser_dates = []
    dateutil_datetime = []
    try:
        for i in range(len(date_parser_dates)):
            try:
                datetime = parse(dates[i][0])
                valid_dateutil_parser_dates.append(dates[i][0])
                dateutil_datetime.append(datetime)
            except:
                  continue
    except:
        pass
    return valid_dateutil_parser_dates, dateutil_datetime

def regex_date_checker(valid_dateutil_parser_dates, date_parser_dates):

    regex_date = []
    actual_date = []
    actual_formatted_date = []

    try:
        for i in range(len(valid_dateutil_parser_dates) + len(date_parser_dates)):

            regex_date_len = len(regex_date)

            if len(valid_dateutil_parser_dates) > 0 :

                regex_date.extend(re.findall(r"[\d]{1,4}[\/\-\=\~\.][\d]{1,2}[\/\-\=\~\.][\d]{1,4}", valid_dateutil_parser_dates[i][0]))
                regex_date.extend(re.findall(r"[\d]{1,2} [ADFJMNOS]\w* [\d]{4}", valid_dateutil_parser_dates[i][0]) )
                regex_date.extend(re.findall(r"([\d]{1,2}[\s\/\-\=\~\.](?i)(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)[\s\/\-\=\~\.][\d]{4})", valid_dateutil_parser_dates[i][0]))

                if (regex_date_len + 1) == len(regex_date):

                    actual_date.append(valid_dateutil_parser_dates[i][0])
                    actual_formatted_date.append(valid_dateutil_parser_dates[i][1])


            if len(date_parser_dates) > 0:

                regex_date.extend(re.findall(r"[\d]{1,4}[\/\-\=\~\.][\d]{1,2}[\/\-\=\~\.][\d]{1,4}", date_parser_dates[i][0]))
                regex_date.extend(re.findall(r"[\d]{1,2} [ADFJMNOS]\w* [\d]{4}", date_parser_dates[i][0]) )
                regex_date.extend(re.findall(r"([\d]{1,2}[\s\/\-\=\~\.](?i)(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)[\s\/\-\=\~\.][\d]{4})", date_parser_dates[i][0]))

                if (regex_date_len + 1) == len(regex_date):

                    actual_date.append(date_parser_dates[i][0])
                    actual_formatted_date.append(date_parser_dates[i][1])
    except:
        pass
    return regex_date, actual_date, actual_formatted_date

def regex_checker_text(text):

    #regex_date = []

    regex_date.extend(re.findall(r"[\d]{1,2,4}[\/\-\=\~\.][\d]{1,2}[\/\-\=\~\.][\d]{1,4}", text))
    regex_date.extend(re.findall(r"[\d]{1,2} [ADFJMNOS]\w* [\d]{4}", text) )
    regex_date.extend(re.findall(r"([\d]{1,2}[\s\/\-\=\~\.](?i)(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)[\s\/\-\=\~\.][\d]{4})", text))

    return regex_date


def regex_checker_feeder(text, img):

    regex_date = regex_checker_text(text)

    return regex_date


def date_bounding_box(orig_img, img, actual_date):

    orig_img = cv2.imread(orig_img)

    data = pytesseract.image_to_data(img, output_type=Output.DICT)

    date_index = []

    for i in range(len(actual_date)):

        for j in range(len(data['text'])):

            if actual_date[i] == data['text'][j]:

                date_index.append(j)

    for idx in range(len(date_index)):

        i = date_index[idx]

        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

        bounding_box_img = cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bounding_box_img = cv2.resize(bounding_box_img, (1000, 600))
        #cv2.imshow('bb_img', bounding_box_img)
        #cv2.waitKey(0)



def ocr_pipeline_date_extraction(image):


    gray_image = simple_gray_image(image)
    thresholded_gray_image = threshold_gray_image(gray_image)
    remove_noise_and_smooth_image = remove_noise_and_smooth(gray_image)

    global regex_date, actual_date, actual_formatted_date

    try:



        text = text_ocr(remove_noise_and_smooth_image)
        #print('Text from removed_noise_and_smooth_image: \n\n{}'.format(text))
        #print('--------------------------------------------------------------')

        date_parser_dates = date_parser(text)
        #print('\nDate parser dates: {}\n'.format(date_parser_dates))

        valid_dateutil_parser_dates, dateutil_datetime = dateutil_parser_check(date_parser_dates)
        #print('\nValid dateutil parser dates: {}, Dateutil datetime: {}\n'.format(valid_dateutil_parser_dates, dateutil_datetime))

        regex_date, actual_date, actual_formatted_date = regex_date_checker(valid_dateutil_parser_dates, date_parser_dates)

        if len(regex_date) >= 1:
            #print('Extracting through remove_noise_and_smooth_image')
            #print('\nRegex date: {}, Actual date: {}, Actual formatted date: {}\n'.format(regex_date, actual_date, actual_formatted_date))

            date_bounding_box(img, remove_noise_and_smooth_image, regex_date)

        #print('--------------------------------------------------------------')
    except:

        try:


            if len(regex_date) == 0:

                text = text_ocr(thresholded_gray_image)
                #print('\nText from thresholded_gray_image: \n\n{}'.format(text))
                #print('--------------------------------------------------------------')

                date_parser_dates = date_parser(text)
                #print('\nDate parser dates: {}\n'.format(date_parser_dates))

                valid_dateutil_parser_dates, dateutil_datetime = dateutil_parser_check(date_parser_dates)
                #print('\nValid dateutil parser dates: {}, Dateutil datetime: {}\n'.format(valid_dateutil_parser_dates, dateutil_datetime))

                regex_date, actual_date, actual_formatted_date = regex_date_checker(valid_dateutil_parser_dates, date_parser_dates)

                if len(regex_date) >= 1:
                    #print('Extracting through thresholded_gray_image')
                    #print('\nRegex date: {}, Actual date: {}, Actual formatted date: {}\n'.format(regex_date, actual_date, actual_formatted_date))

                    bounding_box_img = date_bounding_box(img, thresholded_gray_image, regex_date)

                #print('--------------------------------------------------------------')

        except:
            if len(regex_date) == 0:

                text = text_ocr(gray_image)
                #print('\nText from gray_image: \n\n{}'.format(text))
                #print('--------------------------------------------------------------')

                date_parser_dates = date_parser(text)
                #print('\nDate parser dates: {}\n'.format(date_parser_dates))

                valid_dateutil_parser_dates, dateutil_datetime = dateutil_parser_check(date_parser_dates)
                #print('\nValid dateutil parser dates: {}, Dateutil datetime: {}\n'.format(valid_dateutil_parser_dates, dateutil_datetime))

                regex_date, actual_date, actual_formatted_date = regex_date_checker(valid_dateutil_parser_dates, date_parser_dates)

                if len(regex_date) >= 1:
                    #print('Extracting through gray_image')
                    #print('\nRegex date: {}, Actual date: {}, Actual formatted date: {}\n'.format(regex_date, actual_date, actual_formatted_date))

                    bounding_box_img = date_bounding_box(img, gray_image, regex_date)

                #print('--------------------------------------------------------------')


    return regex_date, actual_date, actual_formatted_date


def ocr_pipeline_run(image_name):

    images = 0
    images_date_detected = 0

    filestr = image_name.read()
    npimg = np.fromstring(filestr, np.uint8)
    image_name = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # for image_name in image_name:#glob.glob(image_name):
    #     images+=1
    #     only_image_name = image_name#.split('/')[1]
    #     print('\nImage no. {} : {}\n'.format(images, only_image_name))

    regex_date, actual_date, actual_formatted_date = ocr_pipeline_date_extraction(image_name)

    if len(regex_date) >= 1:
        images_date_detected+=1
        #print('Images detected with date {} of {} images successfully\n'.format(images_date_detected, images))
        #print('----------------------------------------')

    else:

        if len(regex_date) == 0:

            #img = image_name
            #resized_img = cv2.resize(img, (1000, 600))
            #cv2.imshow('Orig_resized_image_with_date_not _extracted', resized_img)


            gray_image = simple_gray_image(image_name)
            #gray_resized_img = cv2.resize(gray_image, (1000, 600))
            #cv2.imshow('Gray_resized_image_with_date_not _extracted', gray_resized_img)

            thresholded_gray_image = threshold_gray_image(gray_image)
            #thresholded_gray_resized_img = cv2.resize(thresholded_gray_image, (1000, 600))
            #cv2.imshow('thresholded_gray_image_with_date_not _extracted', thresholded_gray_resized_img)

            remove_noise_and_smooth_image = remove_noise_and_smooth(gray_image)

            text = text_ocr(remove_noise_and_smooth_image)
            regex_date = regex_checker_feeder(text, image_name)

            if len(regex_date) >= 1:
                #print('Extracting through directly feeding the text to regex')
                #print('\nregex_date: {}'.format(regex_date))

                images_date_detected+=1
                #print('\nImages detected with date {} of {} images successfully\n'.format(images_date_detected, images))
                #print('----------------------------------------')



            else:#if len(regex_date) == 0:

                warped = birds_eye_view(image_name)

                #try:
                if warped.all() != 0:

                    warped = imutils.resize(warped, height = 650)
                    regex_date, actual_date, actual_formatted_date = ocr_pipeline_date_extraction(warped)
                    #print('regex birds eye view')
                    if len(regex_date) >= 1:
                        #print('\nExtracting through birds eye view')
                        images_date_detected+=1
                        #print('Images detected with date {} of {} images successfully\n'.format(images_date_detected, images))
                        #print('----------------------------------------')
                    else:
                        pass#print('\nNo date is extracted')
                #except:
                    #pass

    return regex_date
