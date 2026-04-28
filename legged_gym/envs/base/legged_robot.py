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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            # 计算力矩
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # 将力矩施加到仿真器中的驱动器
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # 执行一次仿真子步（物理更新）
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            # 刷新关节状态（位置、速度等）张量
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
            检查终止条件，计算观测值以及奖励，
            调用函数self._post_physics_step_callback()做通用计算
            如果必须，则调用self._draw_debug_vis()
        """
        # 刷新，获取最新的状态值
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # 增加episode长度计数器和全局计数器。
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities 计算基座四元数、线速度、角速度在机体坐标系下的表示，以及投影重力。
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # 调用用户自定义回调
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        # 检查终止条件、计算奖励、获取需要重置的环境ID并重置它们。
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        # 计算观测值。
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        # 保存上一次的动作、关节速度、脚部速度。
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # 如果需要调试可视化，绘制。
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
            检查是否终止，如果满足终止条件，重置环境
        """
        # self.contact_forces 形状：(num_envs, num_bodies, 3)
        # self.termination_contact_indices：一个列表或一维张量，包含那些接触即视为失败的刚体索引（例如机器人的脚、小腿等不应受力的部位）。
        # self.contact_forces[:, self.termination_contact_indices, :] 从所有环境中选出特定的刚体（按索引），形状变为 (num_envs, len(indices), 3)。
        # torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) 对最后一个维度 欧几里得范数L2  -> (num_envs, len(indices))。
        # torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1) -> (num_envs)。
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.episode_length_buf：形状(num_envs, )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
            重置env_ids中的环境
            重置self._reset_dofs(env_ids)关节
            重置self._reset_root_states(env_ids)，脚部位置
            重置self._resample_commands(env_ids) 采样命令
            当某些环境因失败或超时而需要重新开始时，该方法负责将它们的物理状态、内部缓冲区、课程进度等恢复到初始状态，
            并记录刚结束的 episode 的统计信息。
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        # 如果启用了地形难度自适应（如从平地向复杂地形过渡），
        # 则调用 _update_terrain_curriculum 根据这些环境的表现调整它们的地形等级。
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # 如果启用了命令课程（例如逐渐提高速度指令的上限），
        # 并且整个训练步数恰好是 max_episode_length 的整数倍（即每完整 episode 才更新一次，避免频繁变化），
        # 则调用 update_command_curriculum 调整命令范围（如最大线速度）。
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states 重置机器人物理状态
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        # 重新采样命令
        self._resample_commands(env_ids)

        # reset buffers 重置内部缓冲区
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras 记录 Episode 统计信息
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # self.episode_sums 是一个字典，存储每个 episode 的累积奖励分量（例如 rew_lin_vel、rew_torque 等），形状均为 (num_envs,)。
            # 对于每个奖励项 key，计算刚结束的 episode 中这些环境的平均每秒钟奖励（累积值 / max_episode_length_s），存入 extras["episode"]['rew_' + key]，供 TensorBoard 或 WandB 记录。
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info 记录额外的课程信息
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm 传递超时信息（可选）
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
            负责将多个奖励项按比例组合成最终的单步奖励，
            并累计每个 episode 的奖励分量，
            同时支持可选的奖励裁剪和终止奖励处理。
        """
        self.rew_buf[:] = 0. # 存储当前步每个环境的即时奖励。
        # 遍历所有奖励函数
        # self.reward_functions：一个列表，存储了所有奖励项的计算函数（例如 _reward_lin_vel、_reward_torque 等），这些函数返回形状为 (num_envs,) 的张量。
        # self.reward_scales：字典，为每个奖励项提供缩放系数（正数表示鼓励，负数表示惩罚）。
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # 可选的正向奖励裁剪 如果配置中要求只保留正奖励（即禁止负奖励），则将所有负值裁剪为 0
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping  添加终止奖励
        # 终止奖励是一个特殊的奖励项，通常只在环境因为失败（如跌倒）而终止时给予一个大的负值，在正常步时返回 0。
        # 将其放在最后添加，且在正向裁剪之后，是为了避免负的终止奖励被 only_positive_rewards 选项误裁剪掉（因为终止惩罚应该总是生效，不应该被屏蔽）。
        # 它会累加到总奖励和 episode 累计和中。
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        将机器人当前的各种状态量（速度、姿态、关节信息等）进行缩放、拼接，形成策略网络输入的张量 self.obs_buf。
        同时，根据配置可选地加入地形高度感知和观测噪声（域随机化）。
        self.base_lin_vel * self.obs_scales.lin_vel,           # (num_envs, 3)
        self.base_ang_vel  * self.obs_scales.ang_vel,          # (num_envs, 3)
        self.projected_gravity,                                # (num_envs, 3)
        self.commands[:, :3] * self.commands_scale,            # (num_envs, 3)
        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # (num_envs, num_dof)
        self.dof_vel * self.obs_scales.dof_vel,                # (num_envs, num_dof)
        self.actions                                          # (num_envs, num_actions)


        各分量的含义与形状（假设有 E 个并行环境，D 个自由度，A 个动作维度，通常 A == D）：

        分量	                                 形状	                说明
        self.base_lin_vel	                 (E, 3)	                机器人基座在机体坐标系下的线速度 (vx, vy, vz)。乘以缩放因子 lin_vel（例如 2.0），使值落在合理范围（如 -1~1）。
        self.base_ang_vel	                 (E, 3)	                机体坐标系下的角速度 (roll, pitch, yaw rate)。乘以 ang_vel 缩放。
        self.projected_gravity	             (E, 3)	                重力矢量在机体坐标系下的投影（即机器人的倾斜感知）。值域约 [-1, 1]，无需额外缩放。
        self.commands[:, :3]	             (E, 3)	                期望的运动指令：通常为 (线速度_x, 线速度_y, 角速度_yaw)。乘以 commands_scale（例如 1.0 或 2.0）归一化。
        self.dof_pos - self.default_dof_pos	 (E, D)	                关节位置相对于默认姿态（standing pose）的偏差。乘以 dof_pos 缩放（如 0.25）。
        self.dof_vel	                     (E, D)	                关节速度。乘以 dof_vel 缩放（如 0.05）。
        self.actions	                     (E, A)	                上一时刻策略输出的动作（例如关节位置目标值）。通常直接加入观测，提供反馈循环信息。

        最终观测维度 = 3+3+3+3+D+D+A = 12 + 2D + A（通常 D=A，所以约为 12+3D）。

        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind 可选的地形高度感知（非盲模式）
        if self.cfg.terrain.measure_heights:
            # self.measured_heights：形状 (E, H)，其中 H 是高度扫描点数（例如前向 10 个点 × 侧向 10 个点 = 100）。
            # 这些值表示机器人周围地形相对于机器人站立平面的高度。
            # 计算逻辑：
            #    self.root_states[:, 2]：机器人基座在世界坐标系下的 Z 坐标（高度），形状 (E,)。
            #    减去 0.5：假设机器人基座离地约 0.5 米，得到地面大致高度。
            #    再减去 self.measured_heights，得到地面相对于机器人足端平面的相对高度差。
            #    torch.clip(..., -1, 1)：将差值限制在 [-1, 1] 米范围内，避免极端值。
            #    乘以 height_measurements 缩放因子（例如 1.0）。
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            # 将高度感知信息附加到观测向量的末尾，使策略能够感知地形起伏，适用于非盲（non-blind）训练。
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed 添加观测噪声（域随机化）
        if self.add_noise:
            # torch.rand_like(self.obs_buf)：生成 [0, 1) 均匀分布的随机张量，形状与观测相同。
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
            创建仿真核心、地形和所有并行环境的关键初始化方法。它按照以下步骤完成整个仿真世界的搭建。
            sim_device_id：物理计算的设备（例如 0 表示 GPU，-1 表示 CPU）。
            graphics_device_id：图形渲染设备（通常与 sim_device_id 相同或为 -1 禁用渲染）。
            physics_engine：物理引擎，通常是 gymapi.SIM_PHYSX 或 gymapi.SIM_FLEX。
            sim_params：之前配置的 gymapi.SimParams 对象（包含 dt、解算器、GPU 管线等设置）。
        """
        # 设置重力方向是z轴
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        # 创建self.sim
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        # 根据地形类型创建地形
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        # 该方法会在 self.sim 中实例化 self.num_envs 个独立的环境。
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        这段代码是 Isaac Gym 环境中用于随机化刚体形状属性的回调函数，通常在创建每个环境时被调用。
        它的核心作用是根据配置，为每个环境（或每个物体形状）独立地随机化摩擦系数，从而增加域随机化（Domain Randomization），提升策略的泛化能力。

        props：一个列表，包含当前环境中每个刚体形状（如机器人的腿、躯干的碰撞形状）的物理属性（gymapi.RigidShapeProperties）。每个元素包含 friction（摩擦系数）、restitution（弹性系数）等字段。
        env_id：当前正在创建的环境的 ID（从 0 到 num_envs-1）。
        返回值：修改后的 props 列表，供 Isaac Gym 创建该环境的刚体形状时使用。

        为所有环境预先计算好摩擦系数，而不是在每个环境创建时独立生成。这样做既保证随机性，
        又确保同一环境中所有形状共享相同的摩擦系数（避免同一机器人不同部位摩擦不同导致物理异常）。
        """
        # 检查是否启用摩擦随机化
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization 为所有环境生成摩擦系数（仅一次）
                # 从配置中读取摩擦系数的上下界（例如 [0.5, 1.5]）。
                friction_range = self.cfg.domain_rand.friction_range
                # 将摩擦系数的取值范围离散化为 64 个“桶”（bucket）。
                num_buckets = 64
                # 为每个环境随机分配一个桶索引（形状 (num_envs, 1)），值在 [0, 63] 之间均匀随机。
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                # 生成 64 个随机摩擦系数，每个系数在 [min, max] 内均匀分布（使用 torch_rand_float 函数，该函数可能来自 legged_gym 的工具库）。
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                # 通过索引 friction_buckets[bucket_ids] 得到一个形状为 (num_envs, 1) 的张量，即每个环境最终的摩擦系数。
                self.friction_coeffs = friction_buckets[bucket_ids]
            # 为当前环境的所有形状设置摩擦系数
            for s in range(len(props)):
                # 遍历当前环境的所有刚体形状（props 列表），将每个形状的 friction 属性设置为该环境预先计算好的摩擦系数（一个标量值）。
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            一个类似数组的对象（通常是 gymapi.DofProperties 的列表或结构），包含了当前环境中每个关节的属性，例如 lower（位置下限）、upper（位置上限）、velocity（最大速度）、effort（最大力矩）等。

            env_id (int): Environment id 当前正在创建的环境的 ID（从 0 到 num_envs-1）。

        Returns:
            [numpy.array]: Modified DOF properties
            从 URDF 文件中提取关节的物理限制（位置上下限、速度上限、力矩上限），并将其存储为 PyTorch 张量，
            同时根据配置计算“软限制”（soft limits）用于奖励函数
        """
        if env_id==0:
            # 仅在第一个环境创建时初始化限制张量 创建三个张量用于存储所有关节的限制：
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                # 从 URDF 属性中填充原始限制 遍历所有关节，将 props 中的原始限制值（来自 URDF 文件）填入张量。
                # .item() 将 numpy 或 Python 数值转换为标量。
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits 计算“软限制”（Soft Limits）
                # 目的：将原始的位置限制缩小一个比例，得到一个更窄的、允许自由运动的舒适区间。超出这个软限制（但仍在硬限制内）时，奖励函数可以施加惩罚，从而引导策略避免关节靠近物理极限。
                # 计算过程：
                #    m：关节位置的中点（均值）。
                #    r：关节的运动范围（upper - lower）。
                #    soft_dof_pos_limit：配置参数（通常在 0.0 到 1.0 之间），表示软限制占硬限制的比例。例如 soft_dof_pos_limit = 0.9，则软限制区间为硬限制区间的 90%，左右各缩进 5%。
                #    新的下限 = m - 0.5 * r * soft_limit_ratio
                #    新的上限 = m + 0.5 * r * soft_limit_ratio
                #    示例：原始关节范围 [-1.0, 1.0]，soft_dof_pos_limit = 0.8 → 新范围 [-0.8, 0.8]。
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    '''
    输入值：
        props：一个列表，包含当前环境中每个刚体（如基座、大腿、小腿等）的物理属性（gymapi.RigidBodyProperties），每个元素包含 mass（质量）、com（质心位置）等字段。
        env_id：当前正在创建的环境的 ID（从 0 到 num_envs-1）。
    
    返回值：修改后的 props 列表，供 Isaac Gym 创建该环境的刚体时使用。
    '''
    def _process_rigid_body_props(self, props, env_id):
        # 处理刚体属性（主要是质量）的回调函数，通常在创建每个环境的机器人时被调用。
        # 它的核心作用是根据配置，随机化机器人基座（base）的质量，以增加域随机化（Domain Randomization），提升策略对负载变化的鲁棒性。

        # 这部分代码被注释掉了，原本的作用是在第一个环境（env_id==0）创建时，打印出所有刚体的原始质量以及总质量，用于调试或验证 URDF 的质量分布。
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")

        # randomize base mass随机化基座质量
        if self.cfg.domain_rand.randomize_base_mass:
            # 在原始质量的基础上增加的质量偏移量（可以为负，即减轻质量）。
            rng = self.cfg.domain_rand.added_mass_range
            # np.random.uniform(rng[0], rng[1]) 从该范围内均匀采样一个偏移值。
            # props[0] 通常对应机器人的基座（base），因为 URDF 中第一个刚体往往是基座。将其原始质量加上随机偏移。
            # 其他刚体不变：只随机化基座质量，其他部位（如四肢）的质量保持 URDF 中的原始值。
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
            在每个物理步进后、计算终止/奖励/观测之前执行的回调钩子。
            它的主要作用是根据配置重新采样命令（运动目标）、处理航向命令、测量地形高度以及随机推动机器人，以增加环境多样性和策略鲁棒性。
        """
        # 每隔固定的仿真时间（resampling_time 秒），为部分环境重新生成新的运动指令（例如期望的线速度、角速度或航向角）
        #    self.episode_length_buf：每个环境已经持续的仿真步数。
        #    int(self.cfg.commands.resampling_time / self.dt)：重采样间隔对应的仿真步数（因为 dt 是仿真步长，例如 0.00833 秒，resampling_time=0.5 秒 → 60 步）。
        #    % 取模运算：判断当前步数是否为重采样周期的整数倍。
        #    nonzero()：找出满足条件的环境 ID，即那些需要重新采样命令的环境。
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # 为指定的环境随机生成新的命令（如 (vx, vy, yaw_rate) 或航向目标）。这使策略不断适应变化的目标，避免过拟合固定速度。
        self._resample_commands(env_ids)
        # 处理command命令模式
        # 当命令模式为 heading_command 时，用户期望的不是直接给出角速度，而是给出一个目标航向角（存储在 commands[:, 3]），控制器需要计算合适的角速度来跟踪该航向。
        if self.cfg.commands.heading_command:
            # 将机器人局部前向向量（例如 (1,0,0)）旋转到世界坐标系，得到世界系下的前向方向。
            forward = quat_apply(self.base_quat, self.forward_vec)
            # 计算机器人当前在世界系下的航向角（yaw）。
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # self.commands[:, 3] - heading：期望航向角减去当前航向角，得到航向误差。
            # wrap_to_pi(...)：将误差归一化到 [-π, π] 区间。
            # 0.5 * ...：比例控制系数（P 控制器），将误差映射为角速度命令。
            # torch.clip(..., -1., 1.)：限制角速度指令在 [-1, 1] 范围内（通常对应实际角速度的缩放）。
            # 将计算结果赋值给 self.commands[:, 2]（通常角速度命令存储在第 3 列，索引 2）。
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            # 测量地形高度 当配置要求测量地形高度（即非盲模式）时，调用 _get_heights() 获取机器人周围地形的高度数据。
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            # 随机推动机器人 周期性给机器人施加随机的外部推力，模拟外界扰动（如踢打、风力），增强策略的抗干扰能力
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids

        重置指定环境的关节状态的核心方法。
        当某些环境因失败或超时而需要重置时，该方法会随机初始化这些环境中机器人的关节位置（在默认姿态附近扰动），并将关节速度清零，然后将更新后的状态同步到物理仿真器中。
        """
        # 随机化关节位置
        #    self.dof_pos：形状为 (num_envs, num_dof) 的张量，存储所有环境中所有关节的当前位置。
        #    self.default_dof_pos：形状为 (num_dof,) 的张量，存储机器人处于站立姿态时的默认关节角度（来自 URDF 或配置文件）。
        #    torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof))：生成一个形状为 (len(env_ids), num_dof) 的张量，每个元素独立从 [0.5, 1.5] 均匀分布中采样。
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        #  关节速度置零
        self.dof_vel[env_ids] = 0.
        # 将更新后的状态同步到仿真器
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
            重置指定环境的机器人基座（根）状态的核心方法。它负责将机器人的基座位置、姿态和速度重置为初始值，并根据配置添加随机偏移，然后将更新后的状态同步到物理仿真器中。
        """
        # base position 基座位置重置
        '''
        self.root_states：形状为 (num_envs, 13) 的张量，存储每个环境机器人基座的状态。具体字段索引：
        [0:3]：世界坐标系下的位置 (x, y, z)
        [3:7]：四元数姿态 (qw, qx, qy, qz)
        [7:10]：线速度 (vx, vy, vz)
        [10:13]：角速度 (wx, wy, wz)
        self.base_init_state：形状为 (13,) 的张量，包含机器人基座的初始状态（通常来自配置文件，例如 [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]），其中位置通常为 (0,0, height)，姿态为单位四元数，速度全零。
        self.custom_origins：布尔标志，指示是否允许自定义原点（通常用于地形训练，每个环境有自己的地形块中心）。如果为 True：
        先将 self.root_states[env_ids] 整体设置为 base_init_state（包括姿态和速度）。
        然后加上 self.env_origins[env_ids]（每个环境的地形原点偏移，例如 (x_i, y_i, 0)），使机器人放置在地形块的对应位置。
        额外随机化 XY 位置：在 x 和 y 方向上各添加一个 [-1, 1] 米范围内的随机偏移，使机器人在每个地形块中心 ±1 米范围内随机起始，增加初始位置多样性。
        如果 self.custom_origins 为 False（通常用于平坦地面或单环境）：
        同样先设置 base_init_state，再加上 env_origins（但此时 env_origins 可能只是一个统一的偏移，例如所有环境在平面上按网格排列）。
        '''
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities 基座速度随机化
        '''
        将指定环境的线速度（索引 7-9）和角速度（索引 10-12）设置为随机值。
        随机范围：每个速度分量独立从 [-0.5, 0.5] 均匀分布中采样。
        线速度单位：m/s。
        角速度单位：rad/s。
        目的：让机器人每次重置时具有随机的初始速度（例如向前/向后/侧向滑动或旋转），迫使策略学会从各种非零速度状态中恢复，增强鲁棒性。
        '''
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 将更新后的状态同步到仿真器
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        这段 _push_robots 函数是 Isaac Gym 环境中随机推动机器人的实现，用于模拟外部扰动（如踢打、风力等），增强策略的抗干扰能力。它通过直接修改所有机器人基座的 XY 平面线速度，来模拟一个瞬时冲量，然后将更新后的状态同步到物理仿真器。
        """
        # 从配置中获取最大推动速度（单位：m/s）。例如 max_push_vel_xy = 1.0，表示速度范围在 [-1.0, 1.0] 之间。
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        # 随机生成水平速度
        #   self.root_states：形状 (num_envs, 13) 的张量，存储所有环境机器人基座的状态。
        #   索引 7:9 对应线速度的 x 和 y 分量（注意索引 9 是 vz，此处未修改）。
        #   torch_rand_float(-max_vel, max_vel, (self.num_envs, 2))：生成形状 (num_envs, 2) 的张量，每个元素独立从 [-max_vel, max_vel] 均匀分布中采样。
        #   赋值：直接覆盖所有环境的 vx 和 vy，而 vz 保持不变（通常为 0 或之前的值）。
        #   注意：这里没有随机化角速度（root_states[:, 10:13]），只改变了线速度的水平和垂直分量。
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        # 将更新后的状态同步到仿真器
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        用于生成观测噪声缩放向量的关键方法。它根据配置文件中定义的噪声参数，为观测向量的每个分量计算一个标准差缩放因子，后续在 compute_observations 中会使用该向量向观测添加均匀噪声，从而增强策略对传感器噪声的鲁棒性。
        """
        # 初始化噪声向量
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        '''
        观测组	                                维度	索引范围	噪声计算
        线速度 (base_lin_vel)	                3	0:3	    noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        角速度 (base_ang_vel)	                3	3:6	    noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        重力投影 (projected_gravity)	            3	6:9	    noise_scales.gravity * noise_level（无额外 obs_scale）
        命令 (commands)	                        3	9:12	0（命令通常不加噪声）
        关节位置偏差 (dof_pos - default_dof_pos)	12	12:24	noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        关节速度 (dof_vel)	                    12	24:36	noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        上一动作 (actions)	                    12	36:48	0（动作指令通常不加噪声）
        地形高度测量（可选）	                    187	48:235	noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        '''
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors

        # Gym角色可以包含一个或多个刚体。所有角色都有一个根刚体。根状态张量保存仿真中所有角色根刚体的状态。
        # 每个根刚体的状态使用13个浮点数表示，布局与GymRigidBodyState相同：3个浮点数表示位置，4个浮点数表示四元数，3个浮点数表示线速度，3个浮点数表示角速度。
        # 获取基座状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # 获取关节状态张量
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 获取净接触力张量 形状为(num_rigid_bodies, 3)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # 填充最新的关节，基座，接触力张量数据
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        # 要访问张量的内容，可以使用提供的gymtorch互操作模块将其包装在PyTorch张量对象中
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        奖励函数初始化的核心方法。它根据配置文件中的奖励缩放系数（reward_scales），动态地筛选出需要计算的奖励项，将其名称映射为对应的成员方法（如 _reward_lin_vel），并准备好用于累积每个 episode 各奖励项总和的字典。
        """
        # remove zero scales + multiply non-zero ones by dt
        # 移除零缩放系数，并将非零系数乘以仿真步长 dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions 准备奖励函数列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                # 跳过 "termination"：终止奖励通常单独处理（在 compute_reward 中最后添加，且不乘以 dt，因为终止事件是离散的）。因此不放入常规奖励函数列表中。
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            # 获取方法对象：getattr(self, name) 从当前实例中获取名为 name 的方法。这些方法必须已在环境类中定义（例如 def _reward_lin_vel(self): ...），并返回形状为 (num_envs,) 的奖励张量
            self.reward_functions.append(getattr(self, name))

        # reward episode sums 初始化 episode 累计奖励字典
        '''
        为每个奖励项（name）创建一个形状为 (num_envs,) 的全零张量，用于累计每个环境中当前 episode 的该奖励项的总和。
        这些张量会随着每个 step 累加（在 compute_reward 中执行 self.episode_sums[name] += rew），并在环境重置时（reset_idx）记录平均值并清零。
        requires_grad=False：累计奖励仅用于监控，不参与梯度计算
        '''
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
             创建所有并行仿真环境的核心方法。它负责加载机器人模型（URDF/MJCF）、为每个环境独立配置刚体/关节属性、实例化环境中的机器人，并记录关键刚体（如脚、惩罚接触部位、终止接触部位）的索引，供后续奖励和终止条件使用。
        """
        # 加载机器人资产 (URDF/MJCF)
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        # 设置各种 asset_options（驱动器模式、关节合并、胶囊替换、密度、阻尼等）
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # 获取资产的基本信息
        # 获取机器人的自由度数量、刚体数量。
        # 获取默认的 DOF 属性（位置限位、速度限位、力矩限位）和刚体形状属性（摩擦、弹性等），这些将作为后续每个环境修改的基础。
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset 解析刚体名称，分类存储
        # 获取所有刚体名称和 DOF 名称。
        # 根据配置中的 foot_name（如 "foot"）筛选出脚部刚体名称列表。
        # 根据 penalize_contacts_on（如 ["shank", "thigh"]）筛选出需要惩罚接触的刚体名称列表。
        # 根据 terminate_after_contacts_on（如 ["base"]）筛选出一旦接触就终止 episode 的刚体名称列表。
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        # 初始化机器人初始状态
        '''
        将配置中的初始位置、四元数、线速度、角速度拼接成一个长度为 13 的列表。
        转换为 PyTorch 张量保存为 self.base_init_state。
        创建 start_pose 变换，仅设置位置（姿态在创建 actor 时可通过 start_pose.rot 设置，但这里未设置，默认单位四元数？实际上在后面创建 actor 时，
        start_pose 只包含了位置，姿态可能由 URDF 中的默认关节角度决定？通常需要设置初始四元数，但此处可能漏掉了？
        实际上 base_init_state 包含四元数，但在 start_pose 中未使用，而是在创建 actor 后通过 set_actor_root_state_tensor 设置？
        检查 _reset_root_states 确实会设置完整状态。因此这里仅用位置作为初始摆放。）
        '''
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # 获取环境原点并创建每个环境
        '''
        _get_env_origins 计算每个环境的基准原点（例如地形块的中心点）。
        对每个环境，创建一个 Isaac Gym 环境容器（create_env），参数中 env_lower/env_upper 均为零向量，表示环境边界无限或由地形决定；最后一个参数是环境排列的列数（用于可视化布局，实际不影响物理）。
        将环境原点复制并加上水平方向的随机偏移（[-1,1] 米），然后将 start_pose 的位置设置为该点。
        '''
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            # 设置刚体形状属性、创建 actor、设置 DOF 属性、设置刚体属性
            '''
            形状属性：调用 _process_rigid_shape_props（可随机化摩擦）修改属性，然后应用到资产上（注意：set_asset_rigid_shape_properties 会影响该资产后续创建的所有 actor，因此必须在每次创建 actor 前设置，否则所有环境会共享同一属性。这里的设计是每个环境独立调用，确保每个环境有独立的摩擦系数）。
            创建 actor：create_actor 将机器人实例添加到环境 env_handle 中，使用修改后的资产和起始位姿。参数 self.cfg.asset.self_collisions 控制是否启用自碰撞。
            DOF 属性：调用 _process_dof_props（可修改关节限位、刚度、阻尼等），然后通过 set_actor_dof_properties 应用到该 actor。
            刚体属性：获取当前 actor 的刚体属性（质量、质心、惯性），调用 _process_rigid_body_props（可随机化基座质量），然后通过 set_actor_rigid_body_properties 写回，并设置 recomputeInertia=True 让仿真器根据新质量重新计算惯性张量。
            '''
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        # 存储关键刚体的索引 由于所有环境的机器人具有相同的刚体名称，只需在第一个环境中查找刚体句柄，这些句柄在所有环境中是一致的（因为 asset 相同，创建顺序相同）。因此可以安全地将这些索引用于所有环境的批量操作（例如获取脚部接触力）。
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
            确定每个并行环境的初始位置（原点） 的方法。根据地形的不同类型（复杂地形 vs. 平面网格），它采用不同的策略来放置机器人，确保所有环境在空间上互不重叠且适应地形结构。
        """
        # 复杂地形（高度场或三角网格）
        # 地形由高度场（heightfield）或三角网格（trimesh）构成，每个环境占据一块独立的地形块（例如山坡、楼梯等）。每个地形块有其自己的中心原点（存储在 self.terrain.env_origins 中，形状为 (num_rows, num_cols, 3)）。
        '''
        设置标志：self.custom_origins = True 表明每个环境的位置需要基于地形块原点，并允许后续在重置时添加随机偏移。
        确定地形难度等级范围：
            max_init_level 指定初始地形等级的最大值（等级越高地形越复杂）。
            如果启用了课程学习（curriculum=True），则使用 cfg.terrain.max_init_terrain_level（通常为 0 或较低值），让所有环境从简单地形开始。
            如果没有课程学习，则使用最大行索引 num_rows-1，即直接使用全部地形难度。
            随机分配地形等级：self.terrain_levels 为每个环境随机分配一个等级（0 到 max_init_level 之间的整数）。
            分配地形类型（列）：self.terrain_types 将环境 ID 按顺序循环分配到不同的列（地形类型，例如不同纹理或障碍物模式），以实现多样化的地形组合。计算公式：terrain_types[i] = floor(i / (num_envs/num_cols))，确保每个地形列的负载大致均衡。
            存储最大等级：self.max_terrain_level 用于课程学习（后续可逐步提高等级）。
            获取地形块原点：self.terrain_origins 是从 Terrain 类中预先计算好的所有地形块的中心坐标（形状 (num_rows, num_cols, 3)），转换为 PyTorch 张量。
            为每个环境选择原点：通过 self.terrain_origins[self.terrain_levels, self.terrain_types] 索引，得到每个环境的基准位置（z 坐标通常为地形表面高度）。
        '''
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            # 平面地形（或未使用复杂地形）
            '''
            设置标志：self.custom_origins = False，表示重置时不需要添加额外的地形偏移，但依然会加上 env_origins。
                计算网格行列数：
                num_cols ≈ sqrt(num_envs)（向下取整）。
                num_rows = ceil(num_envs / num_cols)，确保能容纳所有环境。
                生成网格点坐标：使用 torch.meshgrid 创建 (num_rows, num_cols) 的网格索引 xx 和 yy。
                缩放间距：将网格索引乘以 cfg.env.env_spacing（例如 3.0 米），得到实际世界坐标。
                展平并截取：xx.flatten()[:num_envs] 取前 num_envs 个值，按行优先顺序排列。
                赋值：env_origins[:, 0] 为 x 坐标，env_origins[:, 1] 为 y 坐标，z 坐标设为 0（平面地面高度为 0）。
            '''
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
            作用是在 GUI 查看器中绘制出机器人感知到的地形高度测量点，帮助开发者直观地验证地形感知模块（measured_heights）是否正确工作。
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        # 清除查看器上之前绘制的所有线条或图元，避免残影。
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # 刷新刚体状态张量，确保 self.root_states 包含最新的机器人位姿（因为可视化需要当前时刻的基座位置和四元数）。
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        # 创建一个线框球体几何体，半径 0.02 米，颜色为黄色 (1,1,0)。该几何体将用于在每个测量点处绘制一个球。
        for i in range(self.num_envs):
            '''
            base_pos：机器人在世界坐标系下的基座位置（x, y, z）。
            heights：该环境当前测得的各采样点处的地形高度（相对于世界坐标系的高度值）。
            height_points：
                self.height_points[i]：预先存储的局部坐标系下的测量点相对位置（通常是机器人周围水平面上的一系列点，例如前向 1.0 米、侧向 0.5 米等）。形状为 (num_height_points, 2) 或 (num_height_points, 3)，仅包含 x, y 偏移。
                quat_apply_yaw(self.base_quat[i].repeat(...), self.height_points[i])：
                将局部偏移点旋转到世界坐标系，但只考虑机器人的偏航角（忽略横滚和俯仰）。这是因为高度测量通常只关心水平方向的地形起伏，与机器人倾斜无关。
                结果 height_points 是每个采样点在世界坐标系下的 水平位置偏移（相对机器人基座的水平投影）。
            计算世界坐标：x = height_points[j, 0] + base_pos[0]，y = height_points[j, 1] + base_pos[1]，z = heights[j]。即采样点的三维世界坐标。
            绘制球体：在计算出的坐标处放置一个黄色小球。
            '''
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
            用于初始化地形高度测量采样点的方法。它根据配置中定义的 x、y 坐标网格，生成一组位于机器人基座坐标系（局部坐标系）下的三维点，后续通过射线投射或高度场查询获取这些点处的地形高度，从而构成地形感知观测。
        """
        # 从配置中读取 measured_points_x 和 measured_points_y，它们通常是一维列表，例如：
        # measured_points_x = [-0.8, -0.4, 0.0, 0.4, 0.8]（前向/后向偏移）
        # measured_points_y = [-0.5, 0.0, 0.5]（侧向偏移）
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        # 将这些列表转换为 PyTorch 张量，并放置到指定设备（CPU/GPU）上。
        grid_x, grid_y = torch.meshgrid(x, y)
        # 计算总采样点数量：len(x) * len(y)。
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        # 创建输出张量，第三维初始为 0。
        # 将展平后的 grid_x 和 grid_y 分别赋值给所有环境的 x、y 坐标。由于所有环境共享相同的采样点布局，因此直接广播（points[:, :, 0] 形状 (num_envs, num_height_points)，grid_x.flatten() 形状 (num_height_points,)，利用广播机制自动复制到每个环境）。
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
            根据机器人当前的基座位置和朝向，采样其周围预设点处的地形高度，用于构建非盲观测（non‑blind observations）。

        函数作用
            输入：env_ids（可选）——需要采样高度的环境子集（默认为 None，表示所有环境）。

            输出：形状为 (num_envs, num_height_points) 的张量，每个元素表示对应环境、对应采样点处的地形高度（单位：米）。

            核心逻辑：

            如果地形是平面（mesh_type == 'plane'），直接返回全零高度（因为平面高度处处相同且已知，通常不需要测量）。
            如果地形类型为 'none'，抛出异常（无法测量高度）。
            否则，根据机器人的位姿将局部采样点变换到世界坐标系，然后从预先构建的高度场（self.height_samples）中通过双线性（取最小）插值获取地形高度。
        """
        if self.cfg.terrain.mesh_type == 'plane':
            # 平面地形快速返回
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            # 无效地形类型报错
            raise NameError("Can't measure height with terrain mesh type 'none'")
        # 计算世界坐标系下的采样点
        '''
        self.height_points：形状 (num_envs, num_height_points, 3)，存储局部坐标系下的采样点（x, y, 0），由 _init_height_points 生成。
        quat_apply_yaw：将局部采样点仅按机器人偏航角（忽略横滚/俯仰）旋转到世界水平方向。这样高度采样点会跟随机器人转向，但不会因地形倾斜而歪斜（符合通常的高度感知假设）。
        self.base_quat：形状 (num_envs, 4)，机器人基座的四元数。
        self.root_states[:, :3]：世界坐标系下的基座位置 (x, y, z)。
        unsqueeze(1)：将位置从 (N, 3) 变为 (N, 1, 3)，以便与旋转后的采样点（(N, M, 3)）广播相加。
        结果 points：形状 (N, M, 3)，每个采样点在世界坐标系中的三维坐标。
        '''
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        # 将世界坐标转换为高度场索引
        '''
        border_size：地形网格的边界偏移量（通常为使采样点落在有效网格内而添加的偏移）。
        horizontal_scale：地形网格的单元格间距（例如 0.1 米）。除以该值将世界坐标（米）转换为网格索引。
        .long()：转换为整数索引。
        展平：px 和 py 变成一维张量，长度为 num_envs * num_height_points。
        边界裁剪：确保索引不超出高度场数组的边界（减 2 是为了留出插值所需的下一个像素）。
        '''
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        # 从高度场采样并取最小高度
        '''
        self.height_samples：二维数组（形状 (H, W)），存储每个地形网格顶点的高度值（单位：米，未经垂直缩放）。
        这里不是双线性插值，而是取三个相邻点（自身、右邻、下邻）的最小值。这是一种简化的“保守”高度估计，可能用于避免机器人脚部穿透地形（取最低点模拟脚底接触）。
        实际实现中，通常采用双线性插值或双三次插值，但此代码选择取最小，可能是为了安全（确保机器人不会低于地形）或性能。
        '''
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # 恢复形状并应用垂直缩放
        '''
        view：将一维结果重塑为 (num_envs, num_height_points)。
        vertical_scale：地形高度数据的缩放因子（例如 1.0 表示原始高度，0.5 表示将地形高度减半）。最终高度单位为米。
        '''
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # 用于鼓励机器人在运动时产生较长的步态周期（即脚在空中停留时间足够长），从而促进自然、高效的行走或奔跑步态
        ''''
          1. 获取当前脚部接触力 > 1N → contact
          2. contact_filt = contact OR last_contacts（滤波）
          3. 更新 last_contacts = contact
          4. first_contact = (feet_air_time > 0) AND contact_filt
          5. feet_air_time += dt
          6. 奖励 = sum( (feet_air_time - 0.5) * first_contact )
          7. 若期望速度 < 0.1，奖励置零
          8. feet_air_time *= NOT contact_filt（接触时清零）
          9. 返回奖励
        '''
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
