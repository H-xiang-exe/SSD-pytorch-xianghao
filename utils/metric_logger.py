from collections import deque, defaultdict

import numpy as np
import torch


class SmoothedValue(object):
    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)  # 双端队列
        self.value = np.nan
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value
        self.value = value

    @property
    def avg(self):
        # 统计平均值
        values = np.array(self.deque)
        return np.mean(values)

    @property
    def global_avg(self):
        return self.total / self.count

    def median(self):
        # 统计中位数
        values = np.array(self.deque)
        return np.median(values)


class MetricLogger(object):
    """建立一个管理所有metric的类，其实例化对象是一个dict，key是loss等需要统计的量，value是对应的SmoothedValue管理器，用于累计训练过程中出现的值

    Args:
        delimiter(str): 各个metric之间的分隔符
    """

    def __init__(self, delimiter=", "):
        self.meters = defaultdict(SmoothedValue)
        self.delimeter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """定义了该方法之后，print输出该对象时，会自动返回该方法的return"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.avg:.3f}({meter.global_avg})")
        return self.delimeter.join(loss_str)


if __name__ == "__main__":
    # dict_1 = defaultdict(SmoothedValue)
    # print(dict_1['a'].count)

    meters = MetricLogger()
    meters.update(loss=3.2, loss_l=2.1)
    print(meters.loss)
