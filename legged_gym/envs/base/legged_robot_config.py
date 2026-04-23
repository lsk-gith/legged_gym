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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096 # 并行环境数量（向量化环境数）。
        num_observations = 235 # 观测向量的维度（策略网络的输入维度）。
        num_privileged_obs = None # 特权观测维度（用于非对称训练，如给 critic 提供完整状态），若为 None 则不使用。 if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12 # 动作向量的维度（通常等于可控关节数）。
        env_spacing = 3.  # 环境中每个机器人实例之间的间距（米），在使用高度场或三角网格地形时无效。 not used with heightfields/trimeshes
        send_timeouts = True # 是否将 episode 超时信息传递给算法（用于区分终止与截断）。 send time out information to the algorithm
        episode_length_s = 20 # 每个 episode 的最大时长（秒），步数 = episode_length_s / sim.dt。 episode length in seconds

    class terrain:
        mesh_type = 'trimesh' # 地形类型：'trimesh'（三角网格）、'heightfield'（高度场）、'plane'（平面）或 'none'。"heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # 地形水平分辨率（米/像素），用于高度场或测量点采样。 [m]
        vertical_scale = 0.005 # 地形垂直缩放（米/单位高度值）。 [m]
        border_size = 25 # 地形边界宽度（米），用于防止机器人掉出世界。 [m]
        curriculum = True # 是否启用课程学习（随训练进度增加地形难度）。

        # 地形的静摩擦、动摩擦和恢复系数（所有地形共用或作为基础值）。
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        # rough terrain only:
        measure_heights = True # 是否测量机器人周围的高度点（用于观测）。

        # 测量点的局部坐标（x 和 y 方向），形成网格，用于获取高度信息
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        selected = False # 是否选择单一地形类型（否则使用混合地形）。select a unique terrain type and pass all arguments
        terrain_kwargs = None # 当 selected=True 时，传递给特定地形生成器的参数。 Dict of arguments for selected terrain
        max_init_terrain_level = 5 # 课程学习的起始难度等级（0 为最易）。 starting curriculum state

        # 每个地形块的长度和宽度（米）。
        terrain_length = 8.
        terrain_width = 8.

        num_rows= 10 # 地形难度级别的行数（课程等级数）。  number of terrain rows (levels)
        num_cols = 20 # 地形类型列数（不同地形特征）。 number of terrain cols (types)

        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # 各地形类型的比例：[平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散障碍]。
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # 斜率阈值，超过此值的斜坡会被修正为垂直面（仅 trimesh）。slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False # 命令是否使用课程学习（逐步扩大速度范围）。
        max_curriculum = 1. # 课程学习最大倍率（相对于初始范围）。
        num_commands = 4 # 命令维度（通常为 4：x 速度、y 速度、偏航角速度、航向）。default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # 命令重新采样间隔（秒）。 time before command are changed[s]
        heading_command = True # 是否使用航向角命令模式（若为 True，则根据航向误差计算角速度命令）。 if true: compute ang vel command from heading error
        # 各命令的取值范围（最小/最大）。
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m] 机器人基座的初始位置 (x, y, z) 米。
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat] 基座初始姿态四元数 (x, y, z, w)。
        # 初始线速度 (m/s) 和角速度 (rad/s)。
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = { # target angles when action = 0.0 各关节的默认角度（当动作为零时的目标位置）。
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # 控制模式：'P'（位置控制）、'V'（速度控制）、'T'（力矩控制）。  P: position, V: velocity, T: torques

        # PD Drive parameters: 各关节的 PD 刚度 (N·m/rad) 和阻尼 (N·m·s/rad)。
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]

        # 动作缩放因子：目标角度 = action_scale * action + default_angle。
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # 控制决策频率降采样倍数：每个策略步对应 decimation 个仿真步。decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = "" # 机器人描述文件路径。
        name = "legged_robot"  # 仿真中机器人的名称。 actor name
        foot_name = "None" # 足端刚体名称，用于计算接触力。 name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = [] # 发生接触时会施加惩罚的刚体名称列表。
        terminate_after_contacts_on = [] # 发生接触即终止 episode 的刚体名称。
        disable_gravity = False # 是否禁用重力。
        collapse_fixed_joints = True # 是否合并固定连接的刚体（减少计算量）。 merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False #是否固定基座（用于调试）。  fixe the base of the robot
        default_dof_drive_mode = 3 # 默认驱动模式（0:无,1:位置,2:速度,3:力矩）。 see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 自碰撞检测（0 启用，1 禁用）。1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True #是否将圆柱碰撞体替换为胶囊体（更稳定）。 replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True #是否翻转视觉网格的 Y 轴到 Z 轴（适配不同导出格式）。 Some .obj meshes must be flipped from y-up to z-up

        # 刚体密度、角阻尼、线阻尼（如果 URDF 未指定）。
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.

        # 最大允许角速度/线速度（rad/s, m/s）。
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.

        armature = 0. # 关节的电机惯量（转子等效转动惯量）。
        thickness = 0.01 # 碰撞体的厚度（用于胶囊体等）。

    class domain_rand:
        randomize_friction = True # 是否随机化地面摩擦系数。
        friction_range = [0.5, 1.25] # 摩擦系数随机范围 [min, max]。
        randomize_base_mass = False # 是否随机化基座质量。
        added_mass_range = [-1., 1.] # 基座额外质量范围（kg）。
        push_robots = True # 是否对机器人施加外部推力。
        push_interval_s = 15 # 推力施加的时间间隔（秒）。
        max_push_vel_xy = 1. # 推力产生的最大水平速度增量 (m/s)。

    class rewards:
        class scales:
            termination = -0.0 # 终止 episode 的惩罚（通常为负值）。
            tracking_lin_vel = 1.0 # 跟踪线速度命令的奖励（通常用指数误差）。
            tracking_ang_vel = 0.5 # 跟踪角速度命令的奖励。
            lin_vel_z = -2.0 # 竖直速度惩罚（希望保持小）。
            ang_vel_xy = -0.05 # 俯仰/滚转角速度惩罚。
            orientation = -0. # 基座姿态偏离水平的惩罚。
            torques = -0.00001 # 力矩平方和惩罚（节能）。
            dof_vel = -0. # 关节速度平方惩罚（平滑运动）。
            dof_acc = -2.5e-7 # 关节加速度平方惩罚（减少抖动）。
            base_height = -0.  # 基座高度偏离目标值的惩罚。
            feet_air_time =  1.0 # 足端腾空时间奖励（鼓励步态周期性）。
            collision = -1. # 非足端碰撞惩罚。
            feet_stumble = -0.0  # 足端异常撞击（如拖拽）惩罚。
            action_rate = -0.01 # 动作变化率惩罚（平滑控制）。
            stand_still = -0. # 静止时小动作的惩罚（避免抖动）。

        only_positive_rewards = True # 是否将总奖励截断到非负（避免早期终止问题）。 if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # 跟踪奖励的方差参数：reward = exp(-error²/σ)。 tracking reward = exp(-error^2/sigma)

        # 软限制百分比，超过该范围施加惩罚。
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.

        base_height_target = 1. # 目标基座高度（米）。
        max_contact_force = 100. # 接触力超过此值会施加惩罚。 forces above this value are penalized

    class normalization:

        # 各观测分量的缩放因子（乘到原始值上）。
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100. # 观测值裁剪范围（绝对值上限）。
        clip_actions = 100. # 动作值裁剪范围（绝对值上限）。

    class noise:
        add_noise = True # 是否添加噪声。
        noise_level = 1.0 # 全局噪声缩放因子。 scales other values

        # 各观测分量的噪声标准差（乘以 noise_level）。
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0 # 参考环境索引（用于聚焦相机）。
        pos = [10, 0, 6]  #相机位置 (x, y, z) 米。 [m]
        lookat = [11., 5, 3.]  #相机注视点 (x, y, z) 米。 [m]

    class sim:
        dt =  0.005 # 仿真时间步长（秒）。
        substeps = 1 # 每个仿真步内的子步数（实际步长 = dt/substeps）。
        gravity = [0., 0. ,-9.81]  # 重力矢量 (x, y, z) m/s²。[m/s^2]
        up_axis = 1  # 向上轴：0 为 Y 轴，1 为 Z 轴。 0 is y, 1 is z

        class physx:
            num_threads = 10 # PhysX 求解器线程数。
            solver_type = 1  # 求解器类型：0=PGS, 1=TGS。 0: pgs, 1: tgs

            # 位置/速度迭代次数。
            num_position_iterations = 4
            num_velocity_iterations = 0

            # 接触偏移和静止偏移（米）。
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]

            bounce_threshold_velocity = 0.5 #弹跳阈值速度（m/s）。 0.5 [m/s]
            max_depenetration_velocity = 1.0 # 最大分离速度（m/s）。
            max_gpu_contact_pairs = 2**23 #GPU 最大接触对数量。 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5 # 缓冲区大小乘数。
            contact_collection = 2 # 接触收集时机：0=从不,1=仅最后子步,2=所有子步。 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1 # 随机种子。
    runner_class_name = 'OnPolicyRunner' # 运行器类名（通常为 'OnPolicyRunner'）。
    class policy:
        init_noise_std = 1.0 # 初始动作噪声标准差（用于探索）。

        # Actor 和 Critic 网络的隐藏层维度列表
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

        activation = 'elu' # 激活函数类型： can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # 若使用循环网络（如 LSTM）时的配置。
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0 # 价值损失系数。
        use_clipped_value_loss = True # 是否使用裁剪的价值损失。
        clip_param = 0.2 # PPO 裁剪参数 ε（通常 0.2）。
        entropy_coef = 0.01 # 熵奖励系数（鼓励探索）。
        num_learning_epochs = 5 # 每次策略更新时，对同一批数据学习的轮数。
        num_mini_batches = 4 #小批量数量（总样本数 = num_envs * num_steps_per_env，每个 mini-batch 大小 = 总样本数 / num_mini_batches）。 mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #学习率。5.e-4
        schedule = 'adaptive' # 学习率调度策略：'adaptive'（根据 KL 自适应调整）或 'fixed'。could be adaptive, fixed
        gamma = 0.99 # 折扣因子。
        lam = 0.95 # GAE (Generalized Advantage Estimation) 参数 λ。
        desired_kl = 0.01 # 目标 KL 散度（用于自适应学习率）。
        max_grad_norm = 1. # 梯度裁剪范数。

    class runner:
        policy_class_name = 'ActorCritic' # 策略类名（通常 'ActorCritic'）。
        algorithm_class_name = 'PPO' # 算法类名
        num_steps_per_env = 24 # 每个环境每次迭代收集的步数（即 rollout 长度） per iteration
        max_iterations = 1500 # 最大训练迭代次数（每次迭代更新一次策略）。 number of policy updates

        # logging
        save_interval = 50 # 模型保存间隔（迭代次数）。check for potential saves every this many iterations

        # 实验名称和运行名称（用于日志和保存路径）。
        experiment_name = 'test'
        run_name = ''

        # load and resume
        resume = False # 是否从检查点恢复训练。
        load_run = -1 # 要加载的运行编号（-1 表示最新运行）。 -1 = last run
        checkpoint = -1 # 要加载的检查点编号（-1 表示最新模型）。-1 = last saved model
        resume_path = None # 恢复路径（通常由 load_run 和 checkpoint 自动构建）。 updated from load_run and chkpt