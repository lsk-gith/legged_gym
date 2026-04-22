# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    # 仅应用四元数的偏航（yaw）分量来旋转向量，忽略俯仰和横滚。
    quat_yaw = quat.clone().view(-1, 4)
    # 将四元数的前两个分量（x, y）置零。四元数表示为 (x, y, z, w)，其中 x, y 对应横滚和俯仰轴的分量。
    quat_yaw[:, :2] = 0.
    # 归一化处理，因为置零后模长不再是1。归一化后得到仅表示绕 Z 轴旋转的单位四元数。
    quat_yaw = normalize(quat_yaw)
    # 调用 Isaac Gym 的 quat_apply 函数（四元数旋转向量），返回向量经过偏航旋转后的结果。
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    # 将角度（弧度）限制在 [-π, π) 范围内。
    angles %= 2*np.pi # 取模运算，使角度落在 [0, 2π)。
    # 对于大于 π 的角度，减去 2π，使其落入 (-π, π]。注意边界：当 angles == np.pi 时，条件为 False，保留 π；当 angles 略大于 π 时变为负值。
    angles -= 2*np.pi * (angles > np.pi)
    return angles


'''
    生成在 [lower, upper] 区间内服从平方根分布的随机数。
    分布推导：
        首先生成 r 在 [-1, 1] 均匀分布：2*torch.rand(...)-1
        然后应用变换：r = torch.where(r<0, -torch.sqrt(-r), torch.sqrt(r))
        若 r<0：-sqrt(-r) 得到负数，平方根使分布密度在0附近更高。
        若 r>=0：sqrt(r) 得到非负数。
        变换后的 r 范围仍是 [-1, 1]，但概率密度函数为 f(r) = |r|（线性，在0处最小？需要验证）。实际上，若原始 u~U(-1,1)，则 v = sign(u)*sqrt(|u|)，其分布为 f(v) = 2|v|（在 [-1,1] 上）。
        然后 r = (r + 1)/2 将范围映射到 [0, 1]，此时分布密度为 f(r) = 2*(2r-1?) 等等，这里有点绕。最终线性缩放至 [lower, upper]。
        实际效果：生成的随机数在区间两端概率较高，中间概率较低（类似抛物线形状？）。具体来说，原始 u~U(0,1) 经 sqrt 变换后密度 f(x)=2x，这里是双向对称版本。
'''
# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower