#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/3/5
"""
import base64
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

from root_dir import ROOT_DIR, DATA_DIR
from utils.ugatit_utils import load_test_data


def gan_test():
    export_path = os.path.join(ROOT_DIR, "checkpoint", "model-tf")
    img_path = os.path.join(DATA_DIR, "imgs-real", "head-1.jpg")
    # export_path = os.path.join(DATA_DIR, 'model-tf')  # 模型文件

    # self.test_fake_B: Tensor("generator_B/Tanh:0", shape=(1, 256, 256, 3), dtype=float32)
    # self.test_domain_A: Tensor("test_domain_A:0", shape=(1, 256, 256, 3), dtype=float32)
    # fake_img = sess.run(self.test_fake_B, feed_dict={self.test_domain_A: img_np})

    img = np.asarray(load_test_data(img_path, size=256))
    img = np.squeeze(img, axis=0)
    plt.imshow(img)
    plt.show()
    print(img.shape)

    img_b64 = base64.b64encode(img.tostring()).decode('utf-8')

    img_b64 = img_b64.encode('utf-8')
    # img_np = np.fromstring(base64.b64decode(img_b64), dtype=np.float64)
    # img_np = img_np.reshape((256, 256, 3))
    img_tf = tf.decode_base64(img_b64)
    img_tf = tf.decode_raw(img_tf, tf.float64)
    img_tf = tf.image.convert_image_dtype(img_tf, tf.float64)
    img_tf = tf.reshape(img_tf, [-1, 224, 224, 3])
    # img_np = np.expand_dims(img_np, axis=0)
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], export_path)
        # graph = tf.get_default_graph()
        # print(graph.get_operations())
        res = sess.run("generator_B/Tanh:0",
                       feed_dict={"test_domain_A:0": img_tf})
        res = np.squeeze(res)
        plt.imshow(res)
        plt.show()
        print('[Info] 最终结果: {}'.format(res.shape))


def main():
    gan_test()


if __name__ == '__main__':
    main()
