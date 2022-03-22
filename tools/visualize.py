# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 14:49
# @Author  : tengwei
# @File    : visualize.py
# @Software: PyCharm

from PIL import Image, ImageDraw, ImageFont

import math
import numpy as np
import random


def draw_ocr_box_txt(image, boxes, txts, scores=None, drop_score=0.5, font_path="simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))
    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon([box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]],
                           outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)

    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)
