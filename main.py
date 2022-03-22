# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 13:49
# @Author  : tengwei
# @File    : __init__.py
# @Software: PyCharm

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import copy
import time

from model.ch_ppocr_mobile_v2_det_infer import predictor as predict_det
from model.ch_ppocr_mobile_v2_rec_infer import predictor as predict_rec
from model.common import *
from tools.visualize import *


def ocr_infer(img, params):
    ori_im = img.copy()
    dt_boxes, elapse = predict_det.inference(img)
    if dt_boxes is None:
        return None, None

    img_crop_list = []
    dt_boxes = sorted_boxes(dt_boxes)
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)

    rec_res = predict_rec.inference(img_crop_list)
    filter_boxes, filter_rec_res = [], []
    for box, rec_reuslt in zip(dt_boxes, rec_res):
        text, score = rec_reuslt
        if score >= params["drop_score"]:
            filter_boxes.append(box)
            filter_rec_res.append(rec_reuslt)

    return filter_boxes, filter_rec_res


if __name__ == "__main__":
    image_dir = r"test_data"
    image_file_list = os.listdir(image_dir)
    params = load_yaml("config/biz.yaml")
    font_path = params["vis_font_path"]
    drop_score = params["drop_score"]
    for image_file in image_file_list:
        start_time = time.time()
        img = cv2.imread(os.path.join(image_dir, image_file))
        dt_boxes, rec_res = ocr_infer(img, params)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]
        draw_img = draw_ocr_box_txt(image, boxes, txts, scores, drop_score=drop_score, font_path=font_path)
        cv2.imwrite(os.path.join("results", os.path.basename(image_file)), draw_img[:, :, ::-1])
        print("processing image is ", image_file, time.time() - start_time)
