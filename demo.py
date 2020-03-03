#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/1/2
"""
import math
from UGATIT import UGATIT
from main import parse_args
from root_dir import DATA_DIR
from utils.project_utils import traverse_dir_files, mkdir_if_not_exist
from utils.ugatit_utils import *


class ImgPredictor(object):
    """
    图像预测类
    """

    def __init__(self, gan_type='lsgan', adv_weight=1, cycle_weight=10, identity_weight=10, cam_weight=1000):
        self.gan_type = gan_type

        self.adv_weight = adv_weight
        self.cycle_weight = cycle_weight
        self.identity_weight = identity_weight
        self.cam_weight = cam_weight

        self.gan, self.sess = self.init_model()

    def init_model(self):
        args = parse_args()
        if args is None:
            exit()
        args.phase = 'test'
        # args.dataset = 's2a4zhengsheng'
        # args.dataset = 'selfie2anime'
        args.dataset = 's2azsV1s2a'
        args.img_size = 256

        n_epoch = "1020000"

        args.gan_type = self.gan_type
        args.adv_weight = self.adv_weight
        args.cycle_weight = self.cycle_weight
        args.identity_weight = self.identity_weight
        args.cam_weight = self.cam_weight

        # open session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        gan = UGATIT(sess, args)
        # build graph
        gan.build_model()

        # show network architecture
        # show_all_variables()
        gan.init_model(sess, n_epoch=n_epoch)

        return gan, sess

    def predict_img(self, img_path):
        print('[Info] 预测图像路径: {}'.format(img_path))
        img_np = self.gan.read_img(img_path)
        print('[Info] 输入尺寸: {}'.format(img_np.shape))

        img_fake = self.gan.predict_img(img_np, self.sess)
        img_fake = np.squeeze(img_fake, axis=0)
        print('[Info] 输出尺寸: {}'.format(img_fake.shape))

        img_fake = inverse_transform(img_fake)
        img_fake = img_fake.astype(np.uint8)
        # show_img_rgb(img_fake)
        return img_fake

    def close_sess(self):
        self.sess.close()


def img_predictor_test():
    """
    图像预测测试
    """
    img_dir = os.path.join(DATA_DIR, 'heads-60')
    img_out_dir = os.path.join(DATA_DIR, 'outputs')
    mkdir_if_not_exist(img_out_dir)
    paths_list, names_list = traverse_dir_files(img_dir)

    ip = ImgPredictor()
    for path, name in zip(paths_list, names_list):
        img_fake = ip.predict_img(path)
        img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
        img_fake_path = os.path.join(img_out_dir, '{}.out.jpg'.format(name))
        cv2.imwrite(img_fake_path, img_fake)
        print('[Info] 写入图像路径: {}'.format(img_fake_path))

    print('[Info] 测试完成!')


def img_predictor_test_v2():
    """
    图像预测测试
    """
    img_dir = os.path.join(DATA_DIR, 'imgs')
    img_out_dir = os.path.join(DATA_DIR, 'outputs-v2')
    mkdir_if_not_exist(img_out_dir)
    paths_list, names_list = traverse_dir_files(img_dir)

    ip = ImgPredictor(gan_type='dragan')

    for path, name in zip(paths_list, names_list):
        img_fake = ip.predict_img(path)
        img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
        img_fake_path = os.path.join(img_out_dir, '{}.out.jpg'.format(name))
        cv2.imwrite(img_fake_path, img_fake)
        print('[Info] 写入图像路径: {}'.format(img_fake_path))

    print('[Info] 测试完成!')


def img_predictor_test_v3():
    """
    图像预测测试
    """
    img_dir = os.path.join(DATA_DIR, 'imgs')
    img_out_dir = os.path.join(DATA_DIR, 'outputs-v3')
    mkdir_if_not_exist(img_out_dir)
    paths_list, names_list = traverse_dir_files(img_dir)

    ip = ImgPredictor(gan_type='dragan', adv_weight=1, cycle_weight=15, identity_weight=10, cam_weight=1500)

    for path, name in zip(paths_list, names_list):
        img_fake = ip.predict_img(path)
        img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
        img_fake_path = os.path.join(img_out_dir, '{}.out.jpg'.format(name))
        cv2.imwrite(img_fake_path, img_fake)
        print('[Info] 写入图像路径: {}'.format(img_fake_path))

    print('[Info] 测试完成!')


def merge_imgs(imgs, cols=6, rows=6):
    """
    合并图像
    :param imgs: 图像序列
    :param cols: 行数
    :param rows: 列数
    :param sk: 间隔，当sk=2时，即0, 2, 4, 6
    :return: 大图
    """
    if not imgs:
        raise Exception('[Exception] 合并图像的输入为空!')

    img_shape = imgs[0].shape
    h, w, _ = img_shape

    large_imgs = np.ones((rows * h, cols * w, 3)) * 255  # 大图

    for j in range(rows):
        for i in range(cols):
            idx = j * cols + i
            if idx > len(imgs) - 1:  # 少于帧数，输出透明帧
                break
            # print('[Info] 帧的idx: {}, i: {}, j:{}'.format(idx, i, j))
            large_imgs[(j * h):(j * h + h), (i * w): (i * w + w)] = imgs[idx]
            # print(large_imgs.shape)
            # show_png(large_imgs)
    # show_png(large_imgs)
    return large_imgs


def merge_one_img():
    img_dir = os.path.join(DATA_DIR, 'imgs')
    img_out_dir = os.path.join(DATA_DIR, 'outputs')
    img_merge_dir = os.path.join(DATA_DIR, 'merge')

    paths_list, names_list = traverse_dir_files(img_dir)
    out_paths_list, out_names_list = traverse_dir_files(img_out_dir)
    merge_paths_list, merge_names_list = traverse_dir_files(img_merge_dir)

    img_size = 256

    img_list = []
    for path, out_path, merge_path in zip(paths_list, out_paths_list, merge_paths_list):
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        img_list.append(img)
        img_out = cv2.imread(out_path)
        img_out = cv2.resize(img_out, (img_size, img_size))
        img_list.append(img_out)
        img_merge = cv2.imread(merge_path)
        img_merge = cv2.resize(img_merge, (img_size, img_size))
        img_list.append(img_merge)

    large_img = merge_imgs(img_list, cols=3, rows=7)
    large_img_path = os.path.join(DATA_DIR, 'large_img.jpg')
    cv2.imwrite(large_img_path, large_img)


def merge_outputs():
    img_dir = os.path.join(DATA_DIR, 'outputs-100')
    paths_list, names_list = traverse_dir_files(img_dir)
    print('[Info] 图像数: {}'.format(len(paths_list)))
    img_size = 256

    img_list = []
    for path, name in zip(paths_list, names_list):
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        img_list.append(img)

    large_img = merge_imgs(img_list, cols=10, rows=6)
    large_img_path = os.path.join(DATA_DIR, 'xxx-out-100.jpg')
    cv2.imwrite(large_img_path, large_img)


def merge_sample():
    # sample_dir = 'UGATIT_s2a4zhengsheng_dragan_4resblock_6dis_1_1_10_10_1000_sn_smoothing'
    # sample_dir = 'UGATIT_s2a4zhengsheng_dragan_4resblock_6dis_1_1_15_10_1500_sn_smoothing'
    # sample_dir = 'UGATIT_selfie2anime4zs_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing'
    sample_dir = 'UGATIT_s2a4zhengsheng_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing'

    img_dir = os.path.join(DATA_DIR, 'samples', sample_dir)
    paths_list, names_list = traverse_dir_files(img_dir)
    img_size = 256

    img_fake_list = []
    for path, name in zip(paths_list, names_list):
        if not name.startswith('fake'):
            continue
        img_fake_list.append(path)

    img_real_list = []
    for path, name in zip(paths_list, names_list):
        if not name.startswith('real'):
            continue
        img_real_list.append(path)

    img_list = []
    for p_fake, p_real in zip(img_fake_list, img_real_list):
        img = cv2.imread(p_real)
        img = cv2.resize(img, (img_size, img_size * 2))
        img_list.append(img)

        img = cv2.imread(p_fake)
        img = cv2.resize(img, (img_size, img_size * 2))
        img_list.append(img)

    n_imgs = len(img_list)
    print('图像数量: {}'.format(n_imgs))
    cols = 64
    rows = int(math.ceil(n_imgs / float(cols)))

    large_img = merge_imgs(img_list, cols=cols, rows=rows)
    large_img_path = os.path.join(DATA_DIR, '{}.jpg'.format(sample_dir))
    cv2.imwrite(large_img_path, large_img)


def resize_folder():
    img_dir = os.path.join(DATA_DIR, 'heads-60')
    img_out_dir = os.path.join(DATA_DIR, 'heads-60w256')

    mkdir_if_not_exist(img_out_dir)

    paths_list, names_list = traverse_dir_files(img_dir)
    print('[Info] 图像数: {}'.format(len(paths_list)))
    img_size = 256

    for path, name in zip(paths_list, names_list):
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(os.path.join(img_out_dir, name), img)


def main():
    img_predictor_test()
    # img_predictor_test_v2()
    # img_predictor_test_v3()
    # merge_outputs()
    # merge_sample()
    # resize_folder()


if __name__ == '__main__':
    main()
