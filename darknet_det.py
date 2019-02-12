from ctypes import *
import math
import random
import os
import sys
import pickle as pkl
import cv2
import numpy as np
import random


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class IplROI(Structure):
    pass


class IplTileInfo(Structure):
    pass


class IplImage(Structure):
    pass


IplImage._fields_ = [
    ('nSize', c_int),
    ('ID', c_int),
    ('nChannels', c_int),
    ('alphaChannel', c_int),
    ('depth', c_int),
    ('colorModel', c_char * 4),
    ('channelSeq', c_char * 4),
    ('dataOrder', c_int),
    ('origin', c_int),
    ('align', c_int),
    ('width', c_int),
    ('height', c_int),
    ('roi', POINTER(IplROI)),
    ('maskROI', POINTER(IplImage)),
    ('imageId', c_void_p),
    ('tileInfo', POINTER(IplTileInfo)),
    ('imageSize', c_int),
    ('imageData', c_char_p),
    ('widthStep', c_int),
    ('BorderMode', c_int * 4),
    ('BorderConst', c_int * 4),
    ('imageDataOrigin', c_char_p)]


class iplimage_t(Structure):
    _fields_ = [('ob_refcnt', c_ssize_t),
                ('ob_type',  py_object),
                ('a', POINTER(IplImage)),
                ('data', py_object),
                ('offset', c_size_t)]


lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

#load_alphabet,draw_detections,save_image,    letterbox_image
load_alphabet = lib.load_alphabet
load_alphabet.argtypes = []
load_alphabet.restype = POINTER(POINTER(IMAGE))

draw_detections = lib.draw_detections
draw_detections.argtypes = [IMAGE, POINTER(DETECTION), c_int, c_float, POINTER(
    c_char_p), POINTER(POINTER(IMAGE)), c_int]
draw_detections.restype = IMAGE

save_image = lib.save_image
save_image.argtypes = [IMAGE, c_char_p]

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int,
                              c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

mat_to_image = lib.mat_to_image
mat_to_image.argtypes = [IplImage]
mat_to_image.restype = IMAGE

image_to_mat = lib.image_to_mat
image_to_mat.argtypes = [IMAGE]
image_to_mat.restype = IplImage

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def box_and_label(result, img):
    coords = list(map(lambda v: int(v), list(result[2])))
    c1 = tuple(coords[:2])
    c2 = tuple(coords[2:])
    print(result, c1, c2)

    label = result[0] + ' - '+str(np.around(float(result[1]), decimals=2))
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def detect(net, meta, image, output_file, thresh=.5, hier_thresh=.5, nms=.45,):
    h = lib.network_height(net)
    w = lib.network_width(net)

    im = load_image(image, 0, 0)
    im_sized = letterbox_image(im, h, w)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im_sized)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms):
        do_nms_sort(dets, num, meta.classes, nms)

    # save_image(im,'predictions')
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append(
                    (meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #draw_detections(im, dets, num, thresh, meta.names, load_alphabet(),meta.classes)
    map(lambda result: box_and_label(result, im), res)
    cv2.imsave(im, output_file)
    free_image(im)
    free_detections(dets, num)
    return res


os.system('wget -nc https://raw.githubusercontent.com/ayooshkathuria/pytorch-yolo-v3/master/pallete')
colors = pkl.load(open("pallete", "rb"))
#colors = [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86), (14, 89, 122), (80, 7, 65), (10, 102, 25),(90, 185, 109), (106, 110, 132), (169, 158, 85), (188, 185, 26), (103, 1, 17), (82, 144, 81), (92, 7, 184), (49, 81, 155), (179, 177, 69), (93, 187, 158), (13, 39, 73), (12, 50, 60), (16, 179, 33), (112, 69, 165), (15, 139, 63), (33, 191, 159), (182, 173, 32), (34, 113, 133), (90, 135, 34), (53, 34, 86), (141, 35, 190), (6, 171, 8), (118, 76, 112), (89, 60, 55), (15, 54, 88), (112, 75, 181), (42, 147, 38), (138, 52, 63), (128, 65, 149), (106, 103, 24), (168, 33, 45), (28, 136, 135), (86, 91, 108), (52, 11, 76), (142, 6, 189), (57, 81, 168), (55, 19, 148), (182, 101, 89), (44, 65, 179), (1, 33, 26), (122, 164, 26), (70, 63, 134), (137, 106, 82), (120, 118, 52), (129, 74, 42), (182, 147, 112), (22, 157, 50), (56, 50, 20), (2, 22, 177), (156, 100, 106), (21, 35, 42), (13, 8, 121), (142, 92, 28), (45, 118, 33), (105, 118, 30), (7, 185, 124), (46, 34, 146), (105, 184, 169), (22, 18, 5), (147, 71, 73), (181, 64, 91), (31, 39, 184), (164, 179, 33), (96, 50, 18), (95, 15, 106), (113, 68, 54), (136, 116, 112), (119, 139, 130), (31, 139, 34), (66, 6, 127), (62, 39, 2), (49, 99, 180), (49, 119, 155), (153, 50, 183), (125, 38, 3), (129, 87, 143), (49, 87, 40), (128, 62, 120), (73, 85, 148), (28, 144, 118), (29, 9, 24), (175, 45, 108), (81, 175, 64), (178, 19, 157), (74, 188, 190), (18, 114, 2), (62, 128, 96), (21, 3, 150), (0, 6, 95), (2, 20, 184), (122, 37, 185)]

weights_base_url = 'https://pjreddie.com/media/files/'
cfg_base_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/'
models_lst = ['yolov2', 'yolo2-tiny', 'yolov3', 'yolov3-tiny']
models = {}
for model_name in models_lst:
    models.update({model_name: {'name': model_name, 'cfg': 'cfg/' +
                                model_name+'.cfg', 'weights': model_name + '.weights'}})

if __name__ == "__main__":
    model_name = 'yolov2'
    input_file = 'data/dog.jpg'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        if len(sys.argv) > 2:
            input_file = sys.argv[-1]
    model = models[model_name]
    net = load_net(model['cfg'], model['weights'], 0)
    meta = load_meta("cfg/coco.data")
    r = detect(net, meta, input_file, 'predictionse.jpg')

    # data = cv2.imread('lena.jpg')  # 512 x 512
    # step = data.dtype.itemsize * 3 * data.shape[1]
    # size = data.shape[1], data.shape[0]
    # img = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 3)
    # cv.SetData(img, data, step)
    # ipl = iplimage_t.from_address(id(img))

    print(r)
