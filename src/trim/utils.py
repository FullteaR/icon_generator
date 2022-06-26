import uuid
import random
from timeout_decorator import timeout, TimeoutError
import cv2
import numpy as np


def cross(left, up, right, down, _min_x, _min_y, _max_x, _max_y):
    if up<down and down<_min_y and _min_y<_max_y:
        return True
    elif _min_y<_max_y and _max_y<up and up<down:
        return True
    elif left<right and right<_min_x and _min_x<_max_x:
        return True
    elif _min_x<_max_x and _max_x<left and left<right:
        return True
    else:
        return False

    
def create_box(values, size, shape):
    result = []
    if shape[0] < size or shape[1] < size:
        return result
    for value in values:
        score = value["score"]
        if score < 0.8:
            continue
        bbox = value["bbox"]
        min_x, min_y, max_x, max_y = bbox
        min_x = int(min_x)
        min_y = int(min_y)
        max_x = int(max_x)
        max_y = int(max_y)
        if min(max_y-min_y, max_x-min_x) < size*0.7:
            continue
        a = max(size, max_y-min_y, max_x-min_x)
        b = min(shape[0], shape[1], max(max_y-min_y, max_x-min_x)*3.2)
        a = int(a)
        b = int(b)
        if a>b:
            continue
        while True:
            _size = random.randint(a, b)
            left = random.randint(max(0, max_x-_size), min(min_x, shape[1]-_size))
            right = left + _size

            up = random.randint(max(0, max_y-_size), min(min_y, shape[0]-_size))
            down = up + _size

            assert max_x <= right and right <= shape[1]
            assert 0 <= left and left <= min_x
            assert max_y <= down and down <= shape[0]
            assert 0 <= up and up <= min_y
            
            if len(values)==1:
                break
            for value in values:
                if bbox == value["bbox"]:
                    continue
                _min_x, _min_y, _max_x, _max_y = bbox
                if cross(left, up, right, down, _min_x, _min_y, _max_x, _max_y):
                    break
            else:
                break
        result.append((up,down,left,right))
    return result


def gen_angle():
    return random.uniform(-10,10)


def check_angle(img, angle, center, up, down, left, right):
    scale = 1.0
    trans = cv2.getRotationMatrix2D(center, angle, scale)
    ones = np.zeros_like(img)+1
    ones = cv2.warpAffine(ones, trans, (ones.shape[1], ones.shape[0]))[up:down, left:right]
    return np.sum(ones==0)==0


@timeout(600)
def _trim(item, bright=False):
    size = 256
    key, values = item
    img = cv2.imread(key)
    if img is None:
        print("error loading", key)
        return
    if bright:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        v = clahe.apply(v)
        hsv = cv2.merge((h,s,v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    scale = 1.0
    for up, down, left, right in create_box(values, size, img.shape):
        center = ((left+right)//2, (up+down)//2)
        for i in range(100):
            angle = gen_angle()
            if check_angle(img, angle, center, up, down, left, right):
                break
        else:
            angle = 0
        trans = cv2.getRotationMatrix2D(center, angle, scale)
        img_trim = cv2.warpAffine(img.copy(), trans, (img.shape[1], img.shape[0]))
        img_trim = img_trim[up:down, left:right]
        img_trim = cv2.resize(img_trim, (size, size))
        if bright:
            if "anime" in key:
                save_path = "/faces/bright_anim/{0}-{1}.jpg".format(str(uuid.uuid4()), str(uuid.uuid4()))
            elif "eva" in key:
                save_path = "/faces/bright_eva/{0}-{1}.jpg".format(str(uuid.uuid4()), str(uuid.uuid4()))
            else:
                save_path = "/faces/bright/{0}-{1}.jpg".format(str(uuid.uuid4()), str(uuid.uuid4()))
        else:
            if "anime" in key:
                save_path = "/faces/normal_anim/{0}-{1}.jpg".format(str(uuid.uuid4()), str(uuid.uuid4()))
            elif "eva" in key:
                save_path = "/faces/normal_eva/{0}-{1}.jpg".format(str(uuid.uuid4()), str(uuid.uuid4()))
            else:
                save_path = "/faces/normal/{0}-{1}.jpg".format(str(uuid.uuid4()), str(uuid.uuid4()))
        cv2.imwrite(save_path, img_trim)


def trim(item):
    try:
        _trim(item, bright=False)
    except TimeoutError:
        pass


def brighttrim(item):
    key, value=item
    if "anime" in key:
        return
    try:
        _trim(item, bright=True)
    except TimeoutError:
        pass