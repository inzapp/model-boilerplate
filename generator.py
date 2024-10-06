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
import sys
import cv2
import signal
import threading
import numpy as np
import tensorflow as tf

from glob import glob
from time import sleep
from collections import deque


class DataGenerator:
    def __init__(self, cfg, training=False):
        self.cfg = cfg
        self.training = training
        self.data_index = 0
        self.data_paths = self.get_data_paths()
        self.q_max_size = 4096
        self.q_thread = threading.Thread(target=self.load_xy_to_q)
        self.q_thread.daemon = True
        self.q_lock = threading.Lock()
        self.q_thread_running = False
        self.q_thread_pause = False
        self.q_indices = list(range(self.q_max_size))
        self.q = deque()

    # TODO
    def get_data_paths(self):
        data_paths = []
        if self.training:
            data_paths = glob(f'{self.cfg.train_data_path}/**/*.jpg', recursive=True)
        else:
            data_paths = glob(f'{self.cfg.validation_data_path}/**/*.jpg', recursive=True)
        return data_paths

    def signal_handler(self, sig, frame):
        print('\nSIGINT signal detected, please wait until the end of the thread')
        self.stop()
        sys.exit(0)

    def start(self):
        self.q_thread_running = True
        self.q_thread.start()
        signal.signal(signal.SIGINT, self.signal_handler)
        np.random.shuffle(self.data_paths)
        while self.q_thread_running:
            sleep(1.0)
            percentage = (len(self.q) / self.q_max_size) * 100.0
            if self.training:
                print(f'prefetching data... {percentage:.1f}%')
            with self.q_lock:
                if len(self.q) >= self.q_max_size:
                    print()
                    break

    def stop(self):
        if self.q_thread_running:
            self.q_thread_running = False
            while self.q_thread.is_alive():
                sleep(0.1)

    def pause(self):
        if self.q_thread_running:
            self.q_thread_pause = True

    def resume(self):
        if self.q_thread_running:
            self.q_thread_pause = False

    def load_image(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if self.cfg.input_channels == 1 else cv2.IMREAD_COLOR)

    def preprocess(self, img):
        img = self.resize(img)
        if self.cfg.input_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.asarray(img).reshape((self.cfg.input_rows, self.cfg.input_cols, self.cfg.input_channels)).astype(np.float32) / 255.0
        return x

    # TODO
    def load_xy(self, data_path):
        img = self.load_image(data_path)
        img_f = self.preprocess(img)
        x = img_f
        y = img_f
        return x, y

    def load_xy_to_q(self):
        while self.q_thread_running:
            if self.q_thread_pause:
                sleep(1.0)
            else:
                x, y = self.load_xy(self.next_data_path())
                with self.q_lock:
                    if len(self.q) == self.q_max_size:
                        self.q.popleft()
                    self.q.append((x, y))

    def load(self):
        assert self.q_thread_running
        batch_x, batch_y = [], []
        for i in np.random.choice(self.q_indices, self.cfg.batch_size, replace=False):
            with self.q_lock:
                x, y = self.q[i]
                batch_x.append(x)
                batch_y.append(y)
        batch_x = np.asarray(batch_x).astype(np.float32)
        batch_y = np.asarray(batch_y).astype(np.float32)
        return batch_x, batch_y

    # TODO
    def postprocess(self, y):
        img = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)
        return img

    def next_data_path(self):
        path = self.data_paths[self.data_index]
        self.data_index += 1
        if self.data_index == len(self.data_paths):
            self.data_index = 0
            np.random.shuffle(self.data_paths)
        return path

    def resize(self, img, size='auto'):
        if size == 'auto':
            size = (self.cfg.input_cols, self.cfg.input_rows)
        interpolation = None
        img_h, img_w = img.shape[:2]
        if size[0] > img_w or size[1] > img_h:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
        return cv2.resize(img, size, interpolation=interpolation)

