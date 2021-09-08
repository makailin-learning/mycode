import torch
import torch.nn as nn
import os
import random
import numpy as np
from torch.utils.data import DataLoader

"""
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
           
数据加载顺序完全由用户定义的迭代器控制
shuffle参数，将自动构建顺序采样或打乱的采样器
用户也可以使用sampler参数指定一个自定义的 Sampler 对象，该对象每次都会生成下一个要提取的索引/关键字
batch_size和drop_last参数本质上是用sampler构造batch_sampler

dataset (Dataset)–要从中加载数据的数据集。
batch_size (python：int ， 可选）–每批次要加载多少个样本。(默认值：1）
shuffle (bool ， 可选）–设置为True时，数据每轮会被重新打乱。(默认值：False )
sampler (Sampler ， 可选）–采样器定义了从数据集中抽取样本的策略。 如果指定了采样器，则shuffle必须为False。

batch_sampler (Sampler ， 可选）–类似sampler，但每次只返回一个批次的索引。 与batch_size，shuffle，sampler和drop_last互斥。
num_workers (python：int ， 可选）–数据加载需要的子进程数目。 0表示将在主进程中加载​​数据。 (默认：0）
collat​​e_fn (可调用的， 可选）–合并样本列表以形成一个小批次的张量。 从映射式数据集中使用批量加载时使用。
pin_memory (bool ， 可选）–如果值为True，则数据加载器将把张量复制到 CUDA 固定的内存中，然后返回。 

drop_last (bool ， 可选）–当数据集大小无法被批次大小整除时，若该参数设置为True，则最后一个不完整的批次将被删除；
设置为False，则最后一个批次的大小将比设定的批次大小要小。(默认：False）

timeout(数字 ， 可选）–如果为正，则表示从工作进程中收集批次的超时值。 应始终为非负数。 (默认：0）
worker_init_fn (可调用 ， 可选）–如果不是None，则该函数将在生成种子之后和数据加载之前，在每个工作进程中被调用，
其中工作进程的id（[0, num_workers - 1]范围内的整数值）是它的输入。(默认：None）
"""

class Myloader(DataLoader):
    def __init__(self,dataset,batch_size=1,shuffle=False,sampler=None,batch_sampler=None,
                 num_workers=0,collate_fn=None,pin_memory=False,drop_last=False,
                 timeout=0,worker_init_fn=None):
        self.dataset=dataset
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.collate_fn=collate_fn
        self.timeout=timeout
        self.worker_init_fn=worker_init_fn

        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler与batch_size，shuffle，sampler和drop_last互斥')

        if sampler is not None and shuffle:
            raise ValueError('sampler与shuffle互斥，如果指定了采样器，则shuffle必须为False')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    # dataset.__len__()在sampler中被使用
                    # 目的是生成一个长度为len(dataset)的序列索引(随机的)
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                else:
                    # 目的是生成一个长度为len(dataset)的序列索引(顺序的)
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            # sampler是一个迭代器，一次只返回一个索引
            # batchsampler也是个迭代器，但一次返回batch_size个索引
            batch_sampler=torch.utils.data.sampler.BatchSampler(sampler,self.batch_size,drop_last)
            self.sampler=sampler
            self.batch_sampler=batch_sampler

        if collate_fn is None:
            if self._auto_collation:
                self.collate_fn = _utils.collate.default_collate
            else:
                self.collate_fn = _utils.collate.default_convert

    def __iter__(self) -> '_BaseDataLoaderIter':
        # When using a single worker the returned iterator should be
        # created everytime to avoid reseting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    def __len__(self):
        return len(self.batch_sampler)

    def mosaic_close(self):
        self.dataset.mosaic_close()

