import os
import argparse
import configparser
import importlib
import itertools
from PIL import Image, ExifTags
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils.preprocess
import utils.postprocess
from PIL import ImageGrab
import os
import time
import datetime

def std(image):
    return utils.preprocess.per_image_standardization(image)


def darknet(image):
    return image / 255.


def read_image(path):
    image = Image.open(path)
    for key in ExifTags.TAGS.keys():
        if ExifTags.TAGS[key] == 'Orientation':
            break
    try:
        exif = dict(image._getexif().items())
    except AttributeError:
        return image
    if exif[key] == 3:
        image = image.rotate(180, expand=True)
    elif exif[key] == 6:
        image = image.rotate(270, expand=True)
    elif exif[key] == 8:
        image = image.rotate(90, expand=True)
    return image


def detect(sess, model, names, image, path):
    preprocess = eval(args.preprocess)
    _, height, width, _ = image.get_shape().as_list()
    _image = read_image(path)
    image_original = np.array(np.uint8(_image))
    if len(image_original.shape) == 2:
        image_original = np.repeat(np.expand_dims(image_original, -1), 3, 2)
    image_height, image_width, _ = image_original.shape
    image_std = preprocess(np.array(np.uint8(_image.resize((width, height)))).astype(np.float32))
    feed_dict = {image: np.expand_dims(image_std, 0)}
    tensors = [model.conf, model.xy_min, model.xy_max]
    conf, xy_min, xy_max = sess.run([tf.check_numerics(t, t.op.name) for t in tensors], feed_dict=feed_dict)
    boxes = utils.postprocess.non_max_suppress(conf[0], xy_min[0], xy_max[0], args.threshold, args.threshold_iou)
    scale = [image_width / model.cell_width, image_height / model.cell_height]
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(image_original)
    colors = [prop['color'] for _, prop in zip(names, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    cnt = 0
    for _conf, _xy_min, _xy_max in boxes:
        index = np.argmax(_conf)
        if _conf[index] > args.threshold:
            wh = _xy_max - _xy_min
            _xy_min = _xy_min * scale
            _wh = wh * scale
            linewidth = min(_conf[index] * 10, 3)
            ax.add_patch(patches.Rectangle(_xy_min, _wh[0], _wh[1], linewidth=linewidth, edgecolor=colors[index], facecolor='none'))
            ax.annotate(names[index] + ' (%.1f%%)' % (_conf[index] * 100), _xy_min, color=colors[index])
            cnt += 1
    fig.canvas.set_window_title('%d objects detected' % cnt)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def main():
    model = config.get('config', 'model')
    yolo = importlib.import_module('model.' + model)
    width = config.getint(model, 'width')
    height = config.getint(model, 'height')
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, height, width, 3], name='image')
        builder = yolo.Builder(args, config)
        builder(image)
        global_step = tf.contrib.framework.get_or_create_global_step()
        model_path = tf.train.latest_checkpoint(utils.get_logdir(config))
        tf.logging.info('load ' + model_path)
        slim.assign_from_checkpoint_fn(model_path, tf.global_variables())(sess)
        tf.logging.info('global_step=%d' % sess.run(global_step))
        path = os.path.expanduser(os.path.expandvars(args.path))
        if os.path.isfile(path):
            detect(sess, builder.model, builder.names, image, path)
            plt.show()
        else:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    if os.path.splitext(filename)[-1].lower() in args.exts:
                        _path = os.path.join(dirpath, filename)
                        print(_path)
                        detect(sess, builder.model, builder.names, image, _path)
                        plt.show()

def Screen_Capture():
    directory_path =  "Temp/" # 폴더 경로 정의
    file_name = str(datetime.datetime.now()).replace(" ","-")[0:19].replace(":","-") + ".jpg" # 파일명 정의
    file_path = os.path.join(directory_path, file_name).replace("\\", "/")
    ImageGrab.grab().save(file_path,"JPEG", quality=100)
    return file_path

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='input image path')
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--preprocess', default='std', help='the preprocess function')
    parser.add_argument('-t', '--threshold', type=float, default=0.3)
    parser.add_argument('--threshold_iou', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('-e', '--exts', nargs='+', default=['.jpg', '.png'])
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    file_path = Screen_Capture()
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
