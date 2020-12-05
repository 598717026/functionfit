# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:31:04 2020

@author: Administrator
"""


# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch
import numpy as np


class Net:
  def __init__(self):
    # y = a*x*x + b*x + c
    self.a = torch.rand(1, requires_grad=True) # 参数a
    self.b = torch.rand(1, requires_grad=True) # 参数b
    self.c = torch.rand(1, requires_grad=True) # 参数c
    self.__parameters = dict(a=self.a, b=self.b, c=self.c) # 参数字典
    self.___gpu = False # 是否使用gpu来拟合

  def cuda(self):
    if not self.___gpu:
      self.a = self.a.cuda().detach().requires_grad_(True) # 把a传输到gpu
      self.b = self.b.cuda().detach().requires_grad_(True) # 把b传输到gpu
      self.c = self.c.cuda().detach().requires_grad_(True) # 把c传输到gpu
      self.__parameters = dict(a=self.a, b=self.b, c=self.c) # 更新参数
      self.___gpu = True # 更新标志，表示参数已经传输到gpu了
    # 返回self，以支持链式调用
    return self

  def cpu(self):
    if self.___gpu:
      self.a = self.a.cpu().detach().requires_grad_(True)
      self.b = self.b.cpu().detach().requires_grad_(True)
      self.c = self.c.cpu().detach().requires_grad_(True)
      self.__parameters = dict(a=self.a, b=self.b, c=self.c) # 更新参数
      self.___gpu = False
    return self

  def forward(self, inputs):
    return self.a * inputs * inputs + self.b * inputs + self.c

  def parameters(self):
    for name, param in self.__parameters.items():
      yield param


def main():

  # 生成虚假数据
  x = np.linspace(1, 50, 50)

  # 系数a、b
  a = 2.3
  b = 1.78
  c = 0.35

  # 生成y
  y = a * x * x + b * x + c

  # 转换为Tensor
  x = torch.from_numpy(x.astype(np.float32))
  y = torch.from_numpy(y.astype(np.float32))

  # 定义网络
  net = Net()

  # 传输到GPU
  if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    net = net.cuda()

  # 定义优化器
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005)

  # 定义损失函数
  loss_op = torch.nn.MSELoss(reduction='sum')

  # 最多优化20001次
  for i in range(1, 40001, 1):
    # 向前传播
    out = net.forward(x)
    # 计算损失
    loss = loss_op(y, out)
    # 清空梯度（非常重要）
    optimizer.zero_grad()
    # 向后传播，计算梯度
    loss.backward()
    # 更新参数
    optimizer.step()
    # 得到损失的numpy值
    loss_numpy = loss.cpu().detach().numpy()
    if i % 1000 == 0: # 每1000次打印一下损失
      print(i, loss_numpy)

    if loss_numpy < 0.00001: # 如果损失小于0.00001
      # 打印参数
      a = net.a.cpu().detach().numpy()
      b = net.b.cpu().detach().numpy()
      c = net.c.cpu().detach().numpy()
      print(a, b, c)
      # 退出
      # exit()
      return


if __name__ == '__main__':
  main()
