# scripts/demos/hand_thumbs_up.py
# 让手循环摆出点赞手势

import argparse
import math
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Thumbs up demo for hand robot.")
parser.add_argument("--hand_usd", type=str, required=True, help="Path to the hand USD file.")
parser.add_argument("--prim_path", type=str, default="/World/Hand", help="Prim path to place the hand.")
parser.add_argument("--hz", type=float, default=0.3, help="Gesture frequency in Hz (slower for clear visibility).")
parser.add_argument("--dt", type=float, default=0.01, help="Physics dt.")
parser.add_argument("--hold_time", type=float, default=2.0, help="Time to hold thumbs up pose in seconds.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 启动 Omniverse/Isaac
app = AppLauncher(args).app
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ========== 点赞手势关节角度配置 ==========
# 定义点赞手势的关节角度（单位：弧度）
THUMBS_UP_POSE = {
    # 拇指伸直向上
    "thumb_Joint_1": 0.5,   # 拇指根部稍微外展
    "thumb_Joint_2": -0.2,  # 拇指第二关节稍微弯曲
    "thumb_Joint_3": -0.1,  # 拇指第三关节稍微弯曲
    "thumb_Joint_4": 0.0,   # 拇指末端伸直
    
    # 食指弯曲握拳
    "index_Joint_1": 1.4,   # 食指根部弯曲
    "index_Joint_2": 1.2,   # 食指中间关节弯曲
    "index_Joint_3": 0.8,   # 食指末端关节弯曲
    
    # 中指弯曲握拳
    "middle_Joint_1": 1.4,  # 中指根部弯曲
    "middle_Joint_2": 1.2,  # 中指中间关节弯曲
    "middle_Joint_3": 0.8,  # 中指末端关节弯曲
    
    # 无名指弯曲握拳
    "ring_Joint_1": 1.4,    # 无名指根部弯曲
    "ring_Joint_2": 1.2,    # 无名指中间关节弯曲
    "ring_Joint_3": 0.8,    # 无名指末端关节弯曲
    
    # 小指弯曲握拳
    "little_Joint_1": 1.4,  # 小指根部弯曲
    "little_Joint_2": 1.2,  # 小指中间关节弯曲
    "little_Joint_3": 0.8,  # 小指末端关节弯曲
}

# 自然放松状态的关节角度
RELAXED_POSE = {
    "thumb_Joint_1": 0.0,
    "thumb_Joint_2": 0.0,
    "thumb_Joint_3": 0.0,
    "thumb_Joint_4": 0.0,
    "index_Joint_1": 0.0,
    "index_Joint_2": 0.0,
    "index_Joint_3": 0.0,
    "middle_Joint_1": 0.0,
    "middle_Joint_2": 0.0,
    "middle_Joint_3": 0.0,
    "ring_Joint_1": 0.0,
    "ring_Joint_2": 0.0,
    "ring_Joint_3": 0.0,
    "little_Joint_1": 0.0,
    "little_Joint_2": 0.0,
    "little_Joint_3": 0.0,
}

# ========== 搭建场景 ==========
sim_cfg = sim_utils.SimulationCfg(dt=args.dt, device=args.device)
sim = sim_utils.SimulationContext(sim_cfg)

# 地面与环境光
gp = sim_utils.GroundPlaneCfg()
gp.func("/World/ground", gp)
dl = sim_utils.DomeLightCfg(intensity=3500.0, color=(0.9, 0.9, 0.9))
dl.func("/World/Light", dl)

# 创建环境容器
prim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))

parent_path = args.prim_path.rsplit("/", 1)[0]
if parent_path != "/World":
    prim_utils.create_prim(parent_path, "Xform")

# 创建 Articulation 配置
hand_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=args.hand_usd,
        # 添加物理约束确保稳定性
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            max_effort=2.0,      # 适中的驱动力
            max_velocity=1.0,    # 限制速度
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=0.1,    # 大幅限制线性速度，近似固定效果
            max_angular_velocity=0.5,   # 大幅限制角速度
            max_depenetration_velocity=0.1,
            linear_damping=50.0,        # 增加线性阻尼
            angular_damping=50.0,       # 增加角度阻尼
        ),
        # 移除articulation_props配置，避免固定根链接的错误
    ),
    prim_path=f"{args.prim_path}/Robot",
    articulation_root_prim_path="/base/base",  # 根据之前的经验设置
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),  # 保持在1.0米高度
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),  # 初始线性速度为0
        ang_vel=(0.0, 0.0, 0.0),  # 初始角速度为0
        joint_pos={".*": 0.0},    # 初始时所有关节都是放松状态
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=100.0,    # 适中的刚性，确保手势能够保持
            damping=20.0,       # 适当阻尼，避免震荡
            effort_limit=2.0,   # 限制驱动力
            velocity_limit=1.0, # 限制速度
        ),
    },
)

hand = Articulation(hand_cfg)

# 设置摄像机位置，便于观察点赞手势（调整到新的手部高度）
sim.set_camera_view(eye=[1.0, -0.8, 1.2], target=[0.0, 0.0, 1.0])

def interpolate_pose(pose1, pose2, t):
    """在两个手势之间进行插值
    Args:
        pose1: 起始手势字典
        pose2: 目标手势字典  
        t: 插值参数 (0.0 到 1.0)
    Returns:
        插值后的手势字典
    """
    result = {}
    for joint_name in pose1.keys():
        if joint_name in pose2:
            result[joint_name] = pose1[joint_name] * (1.0 - t) + pose2[joint_name] * t
        else:
            result[joint_name] = pose1[joint_name]
    return result

def get_joint_targets(target_pose, actual_joint_names):
    """根据目标手势和实际关节名称生成关节目标列表"""
    targets = []
    for joint_name in actual_joint_names:
        if joint_name in target_pose:
            targets.append(target_pose[joint_name])
        else:
            targets.append(0.0)  # 未定义的关节保持中性位置
    return targets

# ========== 主循环 ==========
sim.reset()
print("[INFO] Thumbs up demo started!")
print(f"[INFO] Number of joints found: {hand.num_joints}")
print(f"[INFO] Joint names: {hand.joint_names}")

actual_joint_names = hand.joint_names
t = 0.0
cycle_time = 1.0 / args.hz  # 完整周期时间
transition_time = 1.0       # 过渡时间（从放松到点赞，或从点赞到放松）

print(f"[INFO] Cycle time: {cycle_time:.2f}s, Transition time: {transition_time:.2f}s, Hold time: {args.hold_time:.2f}s")

while app.is_running():
    # 持续保持手部在指定位置（模拟固定效果）
    target_pos = torch.tensor([0.0, 0.0, 1.0], device=sim.device)
    current_pos = hand.data.root_pos_w[0]
    
    # 如果位置偏移超过阈值，轻柔地拉回目标位置
    pos_diff = target_pos - current_pos
    pos_error = torch.norm(pos_diff)
    
    if pos_error > 0.02:  # 如果偏离目标位置超过2cm
        # 创建一个轻柔的恢复力
        correction_velocity = pos_diff * 2.0  # 比例控制
        correction_velocity = torch.clamp(correction_velocity, -0.5, 0.5)  # 限制速度
        
        # 设置根部速度来轻柔地移动到目标位置
        root_vel = torch.zeros((hand.num_instances, 6), device=sim.device)
        root_vel[0, :3] = correction_velocity
        hand.write_root_velocity_to_sim(root_vel)

    # 计算当前在周期中的位置
    cycle_position = (t % cycle_time) / cycle_time
    
    # 定义动作序列
    if cycle_position < 0.25:  # 前25%时间：从放松状态过渡到点赞
        transition_progress = cycle_position / 0.25
        current_pose = interpolate_pose(RELAXED_POSE, THUMBS_UP_POSE, transition_progress)
        if cycle_position < 0.01:  # 只在周期开始时打印
            print("[INFO] Transitioning to thumbs up...")
    
    elif cycle_position < 0.75:  # 中间50%时间：保持点赞手势
        current_pose = THUMBS_UP_POSE.copy()
        if 0.25 <= cycle_position < 0.26:  # 只在开始保持时打印一次
            print("[INFO] Holding thumbs up pose!")
    
    else:  # 后25%时间：从点赞过渡到放松状态
        transition_progress = (cycle_position - 0.75) / 0.25
        current_pose = interpolate_pose(THUMBS_UP_POSE, RELAXED_POSE, transition_progress)
        if 0.75 <= cycle_position < 0.76:  # 只在开始过渡时打印一次
            print("[INFO] Transitioning to relaxed pose...")

    # 生成关节目标
    joint_targets = get_joint_targets(current_pose, actual_joint_names)
    
    # 检查手的位置是否正常（相对于目标位置1.0米）
    hand_pos = hand.data.root_pos_w[0]
    target_height = torch.tensor([0.0, 0.0, 1.0], device=sim.device)
    pos_deviation = torch.norm(hand_pos - target_height)
    
    if pos_deviation > 2.0:  # 如果偏离目标位置超过2米（异常情况）
        print(f"[WARNING] Hand position abnormal: {hand_pos}, target: {target_height}, resetting...")
        sim.reset()
        # 重置时间
        t = 0.0
        continue
    
    # 应用关节目标
    if len(joint_targets) == hand.num_joints:
        q_target = torch.tensor([joint_targets], dtype=torch.float32, device=sim.device)
        hand.set_joint_position_target(q_target)
        hand.write_data_to_sim()
    else:
        print(f"[ERROR] Joint count mismatch: expected {hand.num_joints}, got {len(joint_targets)}")

    # 执行仿真步骤
    sim.step()
    t += args.dt
    hand.update(args.dt)

print("[INFO] Thumbs up demo ended.")
