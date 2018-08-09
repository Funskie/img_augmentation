"""
the main program about image augmentation
"""
import os
import argparse
import random
import math
from multiprocessing import Process, cpu_count
import cv2
import utils

def parse_args():
    """setup the arg we need"""
    parser = argparse.ArgumentParser(
        description='An Image Augmentation Tool!!',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='Directory for original image data')
    parser.add_argument('output_dir', help='Directory for augmented images')
    parser.add_argument('num', help='Number of images to be augmented', type=int)
    parser.add_argument('--num_process', help='Number of processes for paralleled augmentation', type=int, default=cpu_count())
    parser.add_argument('--p_mirror', help='Ratio to mirror an image', type=float, default=0.5)
    #以下為調整剪裁照片的參數
    parser.add_argument('--p_crop', help='Ratio to randomly crop an image', type=float, default=1.0)
    parser.add_argument('--crop_size', help='Ratio of cropped image to original in area', type=float, default=0.8)
    parser.add_argument('--crop_hw_noise', help='Variation of h/w ratio', type=float, default=0.1)
    #以下為調整旋轉照片的參數
    parser.add_argument('--p_rotate', help='Ratio to randomly rotate image', type=float, default=1.0)
    parser.add_argument('--p_rotate_crop', help='Ratio to randomly crop out empty part in rotated image', type=float, default=1.0)
    parser.add_argument('--random_angle', help='Tha angle range of rotated', type=float, default=20.0)
    #以下為調整hsv的參數
    parser.add_argument('--p_hsv', help='Ratio to randomly change hsv of an image', type=float, default=1.0)
    parser.add_argument('--hue_rand', help='Ratio of range to change hue', type=int, default=10)
    parser.add_argument('--sat_rand', help='Ratio of range to change saturation', type=float, default=0.1)
    parser.add_argument('--val_rand', help='Ratio of range to change value', type=float, default=0.1)
    #以下為跳整gamma的參數
    parser.add_argument('--p_gamma', help='Ratio to randomly change gamma of an image', type=float, default=1.0)
    parser.add_argument('--gamma_rand', help='Ratio of range to change gamma', type=float, default=2.0)

    args = parser.parse_args()
    args.input_dir = args.input_dir.rstrip('/')
    args.output_dir = args.output_dir.rstrip('/')
    return args

def generate_img_list(args):
    """
    according number of processing and number of augmented image to 
    generate the list of created images per processing
    """
    filenames = os.listdir(args.input_dir)
    num_imgs = len(filenames)
    #計算每張照片要生成多少張新照片
    num_avg_imgs = int(math.floor(args.num / num_imgs))
    #計算多出來的數量，隨機分配
    rem = args.num - num_avg_imgs * num_imgs
    rand_rem_seq = [True] * rem + [False] * (num_imgs - rem)
    random.shuffle(rand_rem_seq)
    #創建一個img_list = [(filename1, 張數), (filename2, 張數), ...]長度為input_img數量
    img_list = [(os.sep.join([args.input_dir, filename]), num_avg_imgs + 1 if plus else num_avg_imgs) for filename, plus in zip(filenames, rand_rem_seq)]
    random.shuffle(img_list)
    #計算每個process要處理的原照片張數與index
    lenght = float(num_imgs) / float(args.num_process)
    indices = [int(round(i * lenght)) for i in (range(args.num_process) + 1)]
    #一次return一個進程要處理的照片數
    return [img_list[indices[i]:indices[i + 1]] for i in range(args.num_process)]

def augment_image(filelist, args):
    """每個processing執行image augmentation"""
    for filepath, n in filelist:
        img = cv2.imread(filepath)
        filename = filepath.split(os.sep)[-1]
        dot_position = filename.rfind('.')

        imgname = filename[:dot_position]
        ext = filename[dot_position:]
        print('Augmenting {}...'.format(filename))
        for i in range(n):
            img_changed = img.copy()
            changed_imgname = '{}_{:0>3d}_'.format(imgname, i)
            
            if random.random() < args.p_mirror:
                img_changed = cv2.flip(img_changed, 1)
                changed_imgname += 'm'
            
            if random.random() < args.p_crop:
                img_changed = utils.random_crop(img_changed, args.crop_size, args.crop_hw_noise)
                changed_imgname += 'c'

            if random.random() < args.p_rotate:
                img_changed = utils.random_rotate_img(img_changed, args.rand_angle, args.p_rotate_crop)
                changed_imgname += 'r'
            
            if random.random() < args.p_hsv:
                img_changed = utils.random_hsv_transform(img_changed, args.hue_rand, args.sat_rand, args.val_rand)
                changed_imgname += 'h'
            
            if random.random() < args.p_gamma:
                img_changed = utils.random_gamma_transform(img_changed, args.gamma_rand)
                changed_imgname += 'g'
            
            output_filepath = os.sep.join([args.output_dir, '{}{}'.format(changed_imgname, ext)])
            cv2.imwrite(output_filepath, img_changed)

