#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/2/11
"""

import os
import cv2
import sys
import random

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR
from utils.project_utils import traverse_dir_files, mkdir_if_not_exist


class DataProcessor(object):
    def __init__(self):
        self.persons_path = os.path.join(ROOT_DIR, 'dataset', 'trainA_extra')
        self.cartoons_path = os.path.join(ROOT_DIR, 'dataset', 'trainB_extra')

        self.std_dim = 256

    def process_dataset(self):
        """
        处理数据集
        """
        c_paths_list, c_names_list = traverse_dir_files(self.cartoons_path)
        p_paths_list, p_names_list = traverse_dir_files(self.persons_path)

        random.seed(47)
        random.shuffle(c_paths_list)
        random.shuffle(p_paths_list)

        train_size = 1500  # 训练集量
        test_size = 100  # 测试集量
        print_size = 100

        count = 0
        train_person_dir = os.path.join(ROOT_DIR, 'dataset', 's2a4zsV1', 'trainA')
        test_person_dir = os.path.join(ROOT_DIR, 'dataset', 's2a4zsV1', 'testA')
        mkdir_if_not_exist(train_person_dir)
        mkdir_if_not_exist(test_person_dir)

        print('[Info] 真人样本总数: {}'.format(len(p_paths_list)))
        for p_path in p_paths_list:
            try:
                p_img = cv2.imread(p_path)
                p_img = cv2.resize(p_img, (256, 256))

                if count < train_size:
                    p_file_name = os.path.join(train_person_dir, u"p_{:04d}.jpg".format(count+1))
                else:
                    p_file_name = os.path.join(test_person_dir, u"p_{:04d}.jpg".format(count+1))

                cv2.imwrite(p_file_name, p_img)
                count += 1
            except Exception as e:
                print('[Error] error {}'.format(e))
                continue

            if count % print_size == 0:
                print(u'[Info] run count: {}'.format(count))

            if count == train_size + test_size:
                break

        train_cartoon_dir = os.path.join(ROOT_DIR, 'dataset', 's2a4zsV1', 'trainB')
        test_cartoon_dir = os.path.join(ROOT_DIR, 'dataset', 's2a4zsV1', 'testB')
        mkdir_if_not_exist(train_cartoon_dir)
        mkdir_if_not_exist(test_cartoon_dir)

        count = 0
        print('[Info] 卡通样本总数: {}'.format(len(c_paths_list)))
        for c_path in c_paths_list:
            try:
                c_img = cv2.imread(c_path)
                c_img = cv2.resize(c_img, (256, 256))

                if count < train_size:
                    c_file_name = os.path.join(train_cartoon_dir, u"c_{:04d}.jpg".format(count+1))
                    cv2.imwrite(c_file_name, c_img)
                else:
                    c_file_name = os.path.join(test_cartoon_dir, u"c_{:04d}.jpg".format(count+1))
                    cv2.imwrite(c_file_name, c_img)

                count += 1
            except Exception as e:
                print('[Error] error {}'.format(e))
                continue

            if count % print_size == 0:
                print(u'[Info] run count: {}'.format(count))

            if count == train_size + test_size:
                break

        print('[Info] 数据处理完成')


def data_processor_test():
    dp = DataProcessor()
    dp.process_dataset()


def main():
    data_processor_test()


if __name__ == '__main__':
    main()
