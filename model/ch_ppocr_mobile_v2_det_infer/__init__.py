# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 13:49
# @Author  : tengwei
# @File    : __init__.py
# @Software: PyCharm

import os.path as osp
import onnxruntime
import time

from model.common import load_yaml
from ppocr.data.imaug import *
from ppocr.postprocess import *

package_root = osp.relpath(osp.dirname(__file__))


class Predictor:
    def __init__(self):
        self.config = load_yaml(osp.join(package_root, "model.yaml"))
        self.pre_process_list = [
            {'DetResizeForTest': {'limit_side_len': self.config["det_limit_side_len"], 'limit_type': self.config["det_limit_type"]}},
            {'NormalizeImage': {'std': [0.229, 0.224, 0.225], 'mean': [0.485, 0.456, 0.406], 'scale': '1./255.', 'order': 'hwc'}},
            {'ToCHWImage': None},
            {'KeepKeys': {'keep_keys': ['image', 'shape']}}
        ]
        self.postprocess_params = {
            "name": "DBPostProcess",
            "thresh": self.config["det_db_thresh"],
            "box_thresh": self.config["det_db_box_thresh"],
            "max_candidates": self.config["max_candidates"],
            "unclip_ratio": self.config["det_db_unclip_ratio"],
            "use_dilation": self.config["use_dilation"]
        }
        self.preprocess_op = create_operators(self.pre_process_list)
        self.postprocess_op = build_post_process(self.postprocess_params)
        self.model = onnxruntime.InferenceSession(osp.join(package_root, self.config["model_path"]))
        self.input_tensor, self.output_tensors = self.model.get_inputs()[0], None

    def order_points_clockwise(self, pts):
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost
        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))

        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue

            dt_boxes_new.append(box)

        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)

        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def inference(self, image):
        ori_im = image.copy()
        data = {'image': image}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()
        input_dict = {}
        input_dict[self.input_tensor.name] = img
        outputs = self.model.run(self.output_tensors, input_dict)
        preds = {}
        preds['maps'] = outputs[0]
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse


predictor = Predictor()
