import numpy as np
import cv2
import os


def convert_coords(pts, coef, intercept):
    """
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    coef = M[:, :2].T.astype('float32')
    intercept = M[:, 2:3].T.astype('float32')

    :param pts:         np.ndarray (None, 2)
    :param coef:        np.ndarray (2, 3)
    :param intercept:   np.ndarray (1, 3)
    :return:            np.ndarray (None, 2)
    """
    new_pts = pts @ coef + intercept
    new_pts = new_pts[:, :2] / new_pts[:, 2:3]
    return new_pts


def resize_and_augment(image, input_shape,
                       flip=True, flip_prob=0.5,
                       min_scale=0.5, max_scale=1.2,
                       side_shift=0.05, color_temp_shift=10, bright_shift=20, gamma=1.2):
    """
    :param image:              3d channel-last bgr numpy.ndarray (h,w,ch) in uint8
    :param key_points:         (n, 2) numpy.ndarray in int or float; [ [x1, y1], [x2, y2], ... [xn, yn] ]
    :param input_shape:        desired image shape (model input shape)
    :param flip:               bool
    :param flip_prob:          float in [0, 1]
    :param min_scale:          float in (0, max_scale)
    :param max_scale:          float in (min_scale, inf)
    :param side_shift:         float in [0, 1]
    :param color_temp_shift:   float/integer in [0, 255]
    :param bright_shift:       float/integer in [0, 255]
    :param gamma:              float in (0, inf)
    :return:
    """
    flip = np.random.choice([True, False], 1, p=[flip_prob, 1. - flip_prob])[0] if flip else flip

    # 得到当前图像映射到目标尺寸的最大尺寸
    h, w = image.shape[:2]
    dst_h, dst_w = input_shape[:2]
    if h / w > dst_h / dst_w:
        scale = dst_h / h
    else:
        scale = dst_w / w

    # 对图像做适当的缩小或放大
    scale_adj = np.random.uniform(min_scale, max_scale)
    new_h, new_w = h * scale * scale_adj, w * scale * scale_adj

    # 映射图上做细微挪动的拉伸变换
    this_side_shift = min(abs(1. / scale_adj - 1.), side_shift)
    dx_up = np.random.uniform(0, this_side_shift) * new_w
    dx_down = np.random.uniform(0, this_side_shift) * new_w
    dy_left = np.random.uniform(0, this_side_shift) * new_h
    dy_right = np.random.uniform(0, this_side_shift) * new_h

    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array(
        [
            [0 + dx_up, 0 + dy_left],
            [new_w + dx_up, 0 + dy_right],
            [new_w + dx_down, new_h + dy_right],
            [0 + dx_down, new_h + dy_left]
        ],
        dtype=np.float32)

    # 原图上做中心移动
    cx = np.random.uniform(low=0, high=max(0, np.max(dst_pts[:, 0]) - dst_w) / (scale * scale_adj))
    cy = np.random.uniform(low=0, high=max(0, np.max(dst_pts[:, 1]) - dst_h) / (scale * scale_adj))
    src_pts[:, 0] += cx
    src_pts[:, 1] += cy

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    dst_img = cv2.warpPerspective(image, M, (dst_w, dst_h))

    # 水平翻转
    if flip:
        dst_img = dst_img[:, ::-1, :]

    # 色温变换
    temp_shift = np.random.uniform(-color_temp_shift, color_temp_shift, (1, 1, 3)).astype('float32')
    # 亮度变换
    b_shift = np.random.uniform(-bright_shift, bright_shift)
    dst_img = (dst_img.astype('float32') + (temp_shift + b_shift)).clip(0, 255)
    # gamma对比度变换
    _gamma = abs(1. - gamma)
    g_shift = np.random.uniform(-_gamma, _gamma) + 1.
    dst_img = (np.power(dst_img / 255., g_shift) * 255.).astype('uint8')

    return dst_img


index = 1000
# cv2.imshow('image', cv2.circle(img.copy(), tuple(key_points[0]), radius=20, color=(0, 0, 255), thickness=-1))
for i in range(16):
    image_path = "/home/chenli/head_counts/head_data/head_save5_class/0_background2/"
    save_path = "/home/chenli/head_counts/head_data/head_save5_class/0_background/"
    image_names = os.listdir(image_path)
    for file_name in image_names:
        img = cv2.imread(image_path + file_name)
        img_aug = resize_and_augment(img, (64, 64, 3))
        # key_points_aug = key_points_aug.round().astype(int)
        cv2.imwrite(save_path + str(index) + ".jpg", img_aug)
        index += 1

cv2.destroyAllWindows()
