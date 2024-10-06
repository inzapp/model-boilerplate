"""
Authors : inzapp

Github url : https://github.com/inzapp/model-boilerplate

Copyright (c) 2024 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['KMP_AFFINITY'] = 'noverbose'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NCCL_P2P_DISABLE'] = '1'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import cv2
import yaml
import random
import numpy as np

from tqdm import tqdm
from time import time
from model import Model
from eta import ETACalculator
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ckpt_manager import CheckpointManager


class TrainingConfig:
    def __init__(self, cfg_path):
        self.__d = self.load(cfg_path)
        self.set_attribute()

    def set_attribute(self):
        for key, value in self.__d.items():
            setattr(self, key, value)

    def get_config(self, cfg, key, default, parse_type, required):
        try:
            value = parse_type(cfg[key])
            if parse_type is str and value.lower() in ['none', 'null']:
                value = None
            return value
        except:
            if required:
                print(f'cfg parse failure, {key} is required')
                exit(-1)
            return default

    def set_config(self, key, value):
        self.__d[key] = value
        self.set_attribute()

    def load(self, cfg_path):
        cfg = None
        if not (os.path.exists(cfg_path) and os.path.isfile(cfg_path)):
            print(f'cfg not found, path : {cfg_path}')
            exit(-1)

        with open(cfg_path, 'rt') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        d = {}
        d['pretrained_model_path'] = self.get_config(cfg, 'pretrained_model_path', '', str, required=False)
        d['train_data_path'] = self.get_config(cfg, 'train_data_path', None, str, required=True)
        d['validation_data_path'] = self.get_config(cfg, 'validation_data_path', None, str, required=True)
        d['input_rows'] = self.get_config(cfg, 'input_rows', None, int, required=True)
        d['input_cols'] = self.get_config(cfg, 'input_cols', None, int, required=True)
        d['input_channels'] = self.get_config(cfg, 'input_channels', None, int, required=True)
        d['model_name'] = self.get_config(cfg, 'model_name', 'model', str, required=False)
        d['lr_policy'] = self.get_config(cfg, 'lr_policy', 'step', str, required=False)
        d['lr'] = self.get_config(cfg, 'lr', None, float, required=True)
        d['lrf'] = self.get_config(cfg, 'lrf', 0.05, float, required=False)
        d['l2'] = self.get_config(cfg, 'l2', 0.0005, float, required=False)
        warm_up = self.get_config(cfg, 'warm_up', 1000, float, required=False)
        d['warm_up'] = float(warm_up) if 0.0 <= warm_up <= 1.0 else int(warm_up)
        d['momentum'] = self.get_config(cfg, 'momentum', 0.9, float, required=False)
        d['batch_size'] = self.get_config(cfg, 'batch_size', None, int, required=True)
        d['max_q_size'] = self.get_config(cfg, 'max_q_size', 1024, int, required=False)
        d['iterations'] = self.get_config(cfg, 'iterations', None, int, required=True)
        d['checkpoint_interval'] = self.get_config(cfg, 'checkpoint_interval', 0, int, required=False)
        d['training_view'] = self.get_config(cfg, 'training_view', False, bool, required=False)
        return d

    def save(self, cfg_path):
        with open(cfg_path, 'wt') as f:
            yaml.dump(self.__d, f, default_flow_style=False, sort_keys=False)

    def print_cfg(self):
        print(self.__d)


class Boilerplate(CheckpointManager):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.input_rows % 32 == 0, f'input_rows must be multiple of 32'
        assert cfg.input_cols % 32 == 0, f'input_cols must be multiple of 32'
        assert cfg.input_channels in [1, 3], f'input_channels must be in [1, 3]'
        self.cfg = cfg
        if self.cfg.checkpoint_interval == 0:
            self.cfg.checkpoint_interval = self.cfg.iterations

        self.set_model_name(self.cfg.model_name)
        self.training_view_previous_time = time()

        if not self.is_valid_path(self.cfg.train_data_path, path_type='dir'):
            print(f'train image path is not valid : {self.cfg.train_data_path}')
            exit(0)

        if not self.is_valid_path(self.cfg.validation_data_path, path_type='dir'):
            print(f'validation image path is not valid : {self.cfg.validation_data_path}')
            exit(0)

        self.pretrained_iteration_count = 0
        if self.cfg.pretrained_model_path is None:
            self.model = Model(cfg=self.cfg).build()
        else:
            self.model, pretrained_iteration_count = self.load_model(self.cfg.pretrained_model_path)
            self.pretrained_iteration_count = pretrained_iteration_count
            print(f'load model success => {self.cfg.pretrained_model_path}')

        self.train_data_generator = DataGenerator(cfg=self.cfg, training=True)
        self.validation_data_generator = DataGenerator(cfg=self.cfg)

    def set_global_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

    def load_model(self, path):
        if path == 'auto':
            auto_model_path = None
            if auto_model_path is None:
                auto_model_path = self.get_best_model_path(path='.')
            if auto_model_path is None:
                auto_model_path = self.get_last_model_path(path='.')
            if auto_model_path is not None:
                self.cfg.set_config('pretrained_model_path', auto_model_path)
                path = auto_model_path

        if not self.is_valid_path(path, path_type='file'):
            print(f'file not found : {self.cfg.pretrained_model_path}')
            exit(0)

        model = tf.keras.models.load_model(path, compile=False, custom_objects={'tf': tf})
        input_shape = model.input_shape[1:]
        self.cfg.set_config('input_rows', input_shape[0])
        self.cfg.set_config('input_cols', input_shape[1])
        self.cfg.set_config('input_channels', input_shape[2])
        pretrained_iteration_count = self.parse_pretrained_iteration_count(path)
        return model, pretrained_iteration_count

    def is_valid_path(self, path, path_type):
        assert path_type in ['file', 'dir']
        if path_type == 'file':
            return (path is not None) and os.path.exists(path) and os.path.isfile(path)
        else:
            return (path is not None) and os.path.exists(path) and os.path.isdir(path)

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    # TODO
    def predict(self, model, img):
        x = self.train_data_generator.preprocess(img)
        x = np.asarray(x).reshape((1,) + x.shape)
        y = np.array(self.graph_forward(model, x)[0])
        img = self.train_data_generator.postprocess(y)
        return img

    # TODO
    def evaluate(self):
        print()
        loss_sum = 0.0
        data_paths = self.validation_data_generator.data_paths
        for path in tqdm(data_paths):
            x, y_true = self.validation_data_generator.load_xy(path)
            x = np.asarray(x).reshape((1,) + x.shape)
            y_pred = np.array(self.graph_forward(self.model, x)[0])
            loss = np.mean(np.abs(y_true - y_pred) ** 2.0)
            loss_sum += loss
        loss_avg = loss_sum  / len(data_paths)
        print(f'loss : {loss_avg:.4f}\n')
        return loss_avg

    def print_loss(self, progress_str, loss):
        loss_str = f'\r{progress_str}'
        loss_str += f' loss : {loss:>8.4f}'
        print(loss_str, end='')

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.training_view_previous_time > 3.0:
            self.training_view_previous_time = cur_time
            data_path = np.random.choice(self.train_data_generator.data_paths)
            img_x = self.train_data_generator.load_image(data_path)
            img_x = self.train_data_generator.resize(img_x)
            img_y = self.predict(self.model, img_x)
            img_cat = np.concatenate([img_x, img_y], axis=1)
            cv2.imshow('training_view', img_cat)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                self.cfg.training_view = False

    def train(self):
        self.model.summary()
        print()
        self.cfg.print_cfg()
        print(f'\ntrain on {len(self.train_data_generator.data_paths)} samples')
        print('start training')
        self.init_checkpoint_dir()
        self.cfg.save(f'{self.checkpoint_path}/cfg.yaml')
        iteration_count = self.pretrained_iteration_count
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.lr)
        lr_scheduler = LRScheduler(lr=self.cfg.lr, iterations=self.cfg.iterations, warm_up=self.cfg.warm_up, policy='step')
        eta_calculator = ETACalculator(iterations=self.cfg.iterations)
        eta_calculator.start()
        self.train_data_generator.start()
        while True:
            batch_x, batch_y = self.train_data_generator.load()
            lr_scheduler.update(optimizer, iteration_count)
            loss = self.compute_gradient(self.model, optimizer, batch_x, batch_y)
            iteration_count += 1
            progress_str = eta_calculator.update(iteration_count)
            self.print_loss(progress_str, loss)
            if self.cfg.training_view and iteration_count >= lr_scheduler.warm_up_iterations:
                self.train_data_generator.pause()
                self.training_view_function()
                self.train_data_generator.resume()
            if iteration_count % 2000 == 0:
                self.save_last_model(self.model, iteration_count)
            if iteration_count % self.cfg.checkpoint_interval == 0:
                self.train_data_generator.pause()
                loss = self.evaluate()
                self.save_best_model(self.model, iteration_count, metric=loss, mode='min', content=f'_loss_{loss:.4f}')
                self.train_data_generator.resume()
            if iteration_count == self.cfg.iterations:
                self.train_data_generator.stop()
                print('train end successfully')
                return

