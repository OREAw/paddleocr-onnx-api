# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 13:49
# @Author  : tengwei
# @File    : __init__.py
# @Software: PyCharm

import cv2
import math
import numpy as np
import os.path as osp
import onnxruntime

from model.common import *
from ppocr.postprocess import *

package_root = osp.relpath(osp.dirname(__file__))


class Predictor():
    def __init__(self):
        self.config = load_yaml(osp.join(package_root, "model.yaml"))
        self.rec_image_shape = [int(v) for v in self.config["rec_image_shape"].split(",")]
        self.character_type = self.config["rec_char_type"]
        self.rec_batch_num = self.config["rec_batch_num"]
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": self.config["rec_char_type"],
            "character_dict_path": osp.join(package_root, self.config["rec_char_dict_path"]),
            "use_space_char": self.config["use_space_char"]
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.model = onnxruntime.InferenceSession(osp.join(package_root, self.config["model_path"]))
        self.input_tensor, self.output_tensors = self.model.get_inputs()[0], None

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def inference(self, img_list):
        img_num = len(img_list)
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))

        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)

            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            outputs = self.model.run(self.output_tensors, input_dict)
            preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res


predictor = Predictor()
