"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import argparse
import configparser
import importlib
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils.data
import utils.visualize


class Drawer(object):
    def __init__(self, sess, names, cell_width, cell_height, image, labels, model, feed_dict):
        self.sess = sess
        self.names = names
        self.cell_width, self.cell_height = cell_width, cell_height
        self.image, self.labels = image, labels
        self.model = model
        self.feed_dict = feed_dict
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        height, width, _ = image.shape
        self.scale = [width / self.cell_width, height / self.cell_height]
        self.ax.imshow(image)
        self.plots = utils.visualize.draw_labels(self.ax, names, width, height, cell_width, cell_height, *labels)
        self.ax.set_xticks(np.arange(0, width, width / cell_width))
        self.ax.set_yticks(np.arange(0, height, height / cell_height))
        self.ax.grid(which='both')
        self.ax.tick_params(labelbottom='off', labelleft='off')
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.colors = [prop['color'] for _, prop in zip(names, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    
    def onclick(self, event):
        for p in self.plots:
            p.remove()
        self.plots = []
        height, width, _ = self.image.shape
        ix = int(event.xdata * self.cell_width / width)
        iy = int(event.ydata * self.cell_height / height)
        self.plots.append(self.ax.add_patch(patches.Rectangle((ix * width / self.cell_width, iy * height / self.cell_height), width / self.cell_width, height / self.cell_height, linewidth=0, facecolor='black', alpha=.2)))
        index = iy * self.cell_width + ix
        prob, iou, xy_min, wh = self.sess.run([self.model.prob[0][index], self.model.iou[0][index], self.model.xy_min[0][index], self.model.wh[0][index]], feed_dict=self.feed_dict)
        xy_min = xy_min * self.scale
        wh = wh * self.scale
        for _prob, _iou, (x, y), (w, h), color in zip(prob, iou, xy_min, wh, self.colors):
            index = np.argmax(_prob)
            name = self.names[index]
            _prob = _prob[index]
            _conf = _prob * _iou
            linewidth = min(_conf * 10, 3)
            self.plots.append(self.ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor=color, facecolor='none')))
            self.plots.append(self.ax.annotate(name + ' (%.1f%%, %.1f%%)' % (_iou * 100, _prob * 100), (x, y), color=color))
        self.fig.canvas.draw()


def main():
    model = config.get('config', 'model')
    cachedir = utils.get_cachedir(config)
    with open(os.path.join(cachedir, 'names'), 'r') as f:
        names = [line.strip() for line in f]
    width = config.getint(model, 'width')
    height = config.getint(model, 'height')
    yolo = importlib.import_module('model.' + model)
    cell_width, cell_height = utils.calc_cell_width_height(config, width, height)
    tf.logging.info('(width, height)=(%d, %d), (cell_width, cell_height)=(%d, %d)' % (width, height, cell_width, cell_height))
    with tf.Session() as sess:
        paths = [os.path.join(cachedir, profile + '.tfrecord') for profile in args.profile]
        num_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(path)) for path in paths)
        tf.logging.warn('num_examples=%d' % num_examples)
        image_rgb, labels = utils.data.load_image_labels(paths, len(names), width, height, cell_width, cell_height, config)
        image_std = tf.image.per_image_standardization(image_rgb)
        image_rgb = tf.cast(image_rgb, tf.uint8)
        ph_image = tf.placeholder(image_std.dtype, [1] + image_std.get_shape().as_list(), name='ph_image')
        global_step = tf.contrib.framework.get_or_create_global_step()
        builder = yolo.Builder(args, config)
        builder(ph_image)
        variables_to_restore = slim.get_variables_to_restore()
        ph_labels = [tf.placeholder(l.dtype, [1] + l.get_shape().as_list(), name='ph_' + l.op.name) for l in labels]
        with tf.name_scope('total_loss') as name:
            builder.create_objectives(ph_labels)
            total_loss = tf.losses.get_total_loss(name=name)
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        _image_rgb, _image_std, _labels = sess.run([image_rgb, image_std, labels])
        coord.request_stop()
        coord.join(threads)
        feed_dict = dict([(ph, np.expand_dims(d, 0)) for ph, d in zip(ph_labels, _labels)])
        feed_dict[ph_image] = np.expand_dims(_image_std, 0)
        logdir = utils.get_logdir(config)
        assert os.path.exists(logdir)
        model_path = tf.train.latest_checkpoint(logdir)
        tf.logging.info('load ' + model_path)
        slim.assign_from_checkpoint_fn(model_path, variables_to_restore)(sess)
        tf.logging.info('global_step=%d' % sess.run(global_step))
        tf.logging.info('total_loss=%f' % sess.run(total_loss, feed_dict))
        _ = Drawer(sess, names, builder.model.cell_width, builder.model.cell_height, _image_rgb, _labels, builder.model, feed_dict)
        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--profile', nargs='+', default=['train'])
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
