#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Instance(object):
    ' 样本类 主要有数据、标签、权重(weight)'
    def __init__(self, data, label):
        """
        :param data: 数据向量
        :param label: 真实标签
        """
        self.label = label
        self.data = data
        self.weight = None
        self.entropy = None

    def get_label(self):
        return self.label

    def get_data(self):
        return self.data

    def get_weight(self):
        return self.weight

