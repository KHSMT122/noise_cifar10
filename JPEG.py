import cv2 as cv

# JPEG圧縮
def img_encode(img, quality):

    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv.imencode(".jpg", img, encode_param)

    return cv.imdecode(encimg, cv.IMREAD_UNCHANGED)