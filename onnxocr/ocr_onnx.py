import cv2
import time
import numpy as np
import argparse
import math
from PIL import Image, ImageDraw, ImageFont
import os
import copy
import predict_det as predict_det
import predict_cls as predict_cls
import predict_rec as predict_rec


def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img

def resize_img(img, input_size=600):
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img

def str_count(s):
    import string
    count_zh = count_pu = 0
    s_len = len(str(s))
    en_dg_count = 0
    for c in str(s):
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)

def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./onnx/fonts/simfang.ttf"):
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)

def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="/home/suma/projects/ocr_onnx/onnxocr/fonts/simfang.ttf"):
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image

def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

def str2bool(v):
    return v.lower() in ("true", "t", "1")

def infer_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--page_num", type=int, default=0)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default='/home/suma/projects/ocr_onnx/onnxocr/models/ppocrv4/det/det.onnx')
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')
    parser.add_argument("--det_box_type", type=str, default='quad')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # EAST parmas
    # parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    # parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    # parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)
    #
    # # SAST parmas
    # parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    # parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    #
    # # PSE parmas
    # parser.add_argument("--det_pse_thresh", type=float, default=0)
    # parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    # parser.add_argument("--det_pse_min_area", type=float, default=16)
    # parser.add_argument("--det_pse_scale", type=int, default=1)
    #
    # # FCE parmas
    # parser.add_argument("--scales", type=list, default=[8, 16, 32])
    # parser.add_argument("--alpha", type=float, default=1.0)
    # parser.add_argument("--beta", type=float, default=1.0)
    # parser.add_argument("--fourier_degree", type=int, default=5)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet')
    parser.add_argument("--rec_model_dir", type=str, default='/home/suma/projects/ocr_onnx/onnxocr/models/ppocrv4/rec/rec.onnx')
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default='/home/suma/projects/ocr_onnx/onnxocr/models/ch_ppocr_server_v2.0/ppocr_keys_v1.txt')
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="/home/suma/projects/ocr_onnx/onnxocr/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for e2e
    # parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    # parser.add_argument("--e2e_model_dir", type=str)
    # parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    # parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    # parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    # parser.add_argument(
    #     "--e2e_char_dict_path", type=str, default="./onnxocr/ppocr/utils/ic15_dict.txt")
    # parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    # parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_dir", type=str, default='/home/suma/projects/ocr_onnx/onnxocr/models/ppocrv4/cls/cls.onnx')
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    parser.add_argument("--warmup", type=str2bool, default=False)

    # SR parmas
    # parser.add_argument("--sr_model_dir", type=str)
    # parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
    # parser.add_argument("--sr_batch_num", type=int, default=1)

    #
    # parser.add_argument(
    #     "--draw_img_save_dir", type=str, default="./onnxocr/inference_results")
    # parser.add_argument("--save_crop_res", type=str2bool, default=False)
    # parser.add_argument("--crop_res_save_dir", type=str, default="./onnxocr/output")
    #
    # # multi-process
    # parser.add_argument("--use_mp", type=str2bool, default=False)
    # parser.add_argument("--total_process_num", type=int, default=1)
    # parser.add_argument("--process_id", type=int, default=0)
    #
    # parser.add_argument("--benchmark", type=str2bool, default=False)
    # parser.add_argument("--save_log_path", type=str, default="./onnxocr/log_output/")
    #
    # parser.add_argument("--show_log", type=str2bool, default=True)
    # parser.add_argument("--use_onnx", type=str2bool, default=False)
    return parser

class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0


    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])

        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        img=img
        ori_im = img.copy()
        # 文字检测
        dt_boxes = self.text_detector(img)

        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        # 图片裁剪
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # 方向分类
        if self.use_angle_cls and cls:
            img_crop_list, angle_list = self.text_classifier(img_crop_list)

        # 图像识别
        rec_res = self.text_recognizer(img_crop_list)

        # if self.args.save_crop_res:
        #     self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes



class ONNXPaddleOcr(TextSystem):
    def __init__(self, **kwargs):
        # 默认参数
        parser = infer_args()
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        params = argparse.Namespace(**inference_args_dict)

        # params.rec_image_shape = "3, 32, 320"
        params.rec_image_shape = "3, 48, 320"

        # 根据传入的参数覆盖更新默认参数
        params.__dict__.update(**kwargs)

        # 初始化模型
        super().__init__(params)

    def ocr(self, img, det=True, rec=True, cls=True):
        if det and rec:
            ocr_res = []
            dt_boxes, rec_res = self.__call__(img, cls)
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            ocr_res.append(tmp_res)
            return ocr_res

def sav2Img(org_img, result, name="draw_ocr.jpg"):
    # 显示结果
    from PIL import Image
    result = result[0]
    # image = Image.open(img_path).convert('RGB')
    # 图像转BGR2RGB
    image = org_img[:, :, ::-1]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores)
    im_show = Image.fromarray(im_show)
    im_show.save(name)


model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)


img = cv2.imread('/home/suma/projects/ocr_onnx/onnxocr/test_images/1.jpg')
s = time.time()
result = model.ocr(img)
e = time.time()
print("total time: {:.3f}".format(e - s))
print("result:", result)
for box in result[0]:
    print(box)

sav2Img(img, result)