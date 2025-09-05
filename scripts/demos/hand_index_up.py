# scripts/demos/hand_index_up.py
# 让手循环摆出食指竖立手势

import argparse
import math
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Index finger up demo for hand robot.")
parser.add_argument("--hand_usd", type=str, required=True, help="Path to the hand USD file.")
parser.add_argument("--prim_path", type=str, default="/World/Hand", help="Prim path to place the hand.")
parser.add_argument("--hz", type=float, default=0.3, help="Gesture frequency in Hz (slower for clear visibility).")
parser.add_argument("--dt", type=float, default=0.01, help="Physics dt.")
parser.add_argument("--hold_time", type=float, default=2.0, help="Time to hold index up pose in seconds.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 启动 Omniverse/Isaac
app = AppLauncher(args).app
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ========== 食指竖立手势关节角度配置 ==========
# 定义食指竖立手势的关节角度（单位：弧度）
INDEX_UP_POSE = {
    # 拇指弯曲
    "thumb_Joint_1": 0.8,   # 拇指根部内收
    "thumb_Joint_2": 1.1,   # 拇指第二关节弯曲
    "thumb_Joint_3": -0.3,   # 拇指第三关节弯曲
    "thumb_Joint_4": -0.2,   # 拇指末端稍微弯曲
    
    # 食指伸直竖立
    "index_Joint_1": 0.0,   # 食指根部伸直
    "index_Joint_2": 0.0,   # 食指中间关节伸直
    "index_Joint_3": 0.0,   # 食指末端关节伸直
    
    # 中指弯曲
    "middle_Joint_1": 1.4,  # 中指根部弯曲
    "middle_Joint_2": 1.2,  # 中指中间关节弯曲
    "middle_Joint_3": 0.8,  # 中指末端关节弯曲
    
    # 无名指弯曲
    "ring_Joint_1": 1.4,    # 无名指根部弯曲
    "ring_Joint_2": 1.2,    # 无名指中间关节弯曲
    "ring_Joint_3": 0.8,    # 无名指末端关节弯曲
    
    # 小指弯曲
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
            max_effort=5.0,      # 增加驱动力，特别是为了大拇指关节
            max_velocity=1.0,    # 保持速度限制
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
        pos=(0.0, 0.0, 0.0),  # 手底部放置在地面上
        rot=(1.0, 0.0, 0.0, 0.0),  # 保持垂直向上
        lin_vel=(0.0, 0.0, 0.0),  # 初始线性速度为0
        ang_vel=(0.0, 0.0, 0.0),  # 初始角速度为0
        joint_pos={".*": 0.0},    # 初始时所有关节都是放松状态
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 为大拇指2/3/4关节设置更强的驱动器
        "thumb_joints": ImplicitActuatorCfg(
            joint_names_expr=["thumb_Joint_[234]"],  # 匹配大拇指的2/3/4关节
            stiffness=300.0,    # 增加刚性
            damping=50.0,       # 增加阻尼以保持稳定
            effort_limit=5.0,   # 增加驱动力限制
            velocity_limit=1.0,  # 保持速度限制
        ),
        # 其他关节保持原有配置
        "other_joints": ImplicitActuatorCfg(
            joint_names_expr=["(?!thumb_Joint_[234]).*"],  # 匹配除了大拇指2/3/4以外的所有关节
            stiffness=100.0,    # 原有的刚性
            damping=20.0,       # 原有的阻尼
            effort_limit=2.0,   # 原有的驱动力限制
            velocity_limit=1.0,  # 原有的速度限制
        ),
    },
)

hand = Articulation(hand_cfg)

# 设置摄像机位置，便于观察食指竖立手势
sim.set_camera_view(eye=[1.0, -0.8, 0.5], target=[0.0, 0.0, 0.0])

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
print("[INFO] Index finger up demo started!")
print(f"[INFO] Number of joints found: {hand.num_joints}")
print(f"[INFO] Joint names: {hand.joint_names}")

actual_joint_names = hand.joint_names
t = 0.0
cycle_time = 1.0 / args.hz  # 完整周期时间
transition_time = 1.0       # 过渡时间（从放松到食指竖立，或从食指竖立到放松）

print(f"[INFO] Cycle time: {cycle_time:.2f}s, Transition time: {transition_time:.2f}s, Hold time: {args.hold_time:.2f}s")

while app.is_running():
    # 持续保持手部在地面位置（模拟固定在地面效果）
    target_pos = torch.tensor([0.0, 0.0, 0.0], device=sim.device)
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
    if cycle_position < 0.25:  # 前25%时间：从放松状态过渡到食指竖立
        transition_progress = cycle_position / 0.25
        current_pose = interpolate_pose(RELAXED_POSE, INDEX_UP_POSE, transition_progress)
        if cycle_position < 0.01:  # 只在周期开始时打印
            print("[INFO] Transitioning to index up...")
    
    elif cycle_position < 0.75:  # 中间50%时间：保持食指竖立手势
        current_pose = INDEX_UP_POSE.copy()
        if 0.25 <= cycle_position < 0.26:  # 只在开始保持时打印一次
            print("[INFO] Holding index up pose!")
    
    else:  # 后25%时间：从食指竖立过渡到放松状态
        transition_progress = (cycle_position - 0.75) / 0.25
        current_pose = interpolate_pose(INDEX_UP_POSE, RELAXED_POSE, transition_progress)
        if 0.75 <= cycle_position < 0.76:  # 只在开始过渡时打印一次
            print("[INFO] Transitioning to relaxed pose...")

    # 生成关节目标
    joint_targets = get_joint_targets(current_pose, actual_joint_names)
    
    # 检查手的位置是否正常（应该在地面上）
    hand_pos = hand.data.root_pos_w[0]
    target_ground = torch.tensor([0.0, 0.0, 0.0], device=sim.device)
    pos_deviation = torch.norm(hand_pos - target_ground)
    
    if pos_deviation > 1.0:  # 如果偏离地面位置超过1米（异常情况）
        print(f"[WARNING] Hand position abnormal: {hand_pos}, target: {target_ground}, resetting...")
        sim.reset()
        # 重置时间
        t = 0.0
        continue
    
    # 应用关节目标
    if len(joint_targets) == hand.num_joints:
        q_target = torch.tensor([joint_targets], dtype=torch.float32, device=sim.device)
        
        # 验证大拇指关节角度
        thumb_joint_indices = [i for i, name in enumerate(actual_joint_names) if "thumb_Joint" in name]
        if thumb_joint_indices:
            thumb_angles = q_target[0, thumb_joint_indices]
            
            # 计算大拇指总体弯曲度（所有关节角度的绝对值之和）
            total_thumb_bend = float(torch.sum(torch.abs(thumb_angles)).item())
            
            # 验证每个关节的角度是否在合理范围内
            thumb_names = ["根部", "第二关节", "第三关节", "末端"]
            for idx, joint_idx in enumerate(thumb_joint_indices):
                angle = float(q_target[0, joint_idx].item())
                # 只在姿势稳定时（保持阶段）打印角度信息
                if 0.3 <= cycle_position <= 0.7:
                    print(f"[DEBUG] 大拇指{thumb_names[idx]}角度: {angle:.2f} 弧度")
            
            if 0.3 <= cycle_position <= 0.7:
                print(f"[DEBUG] 大拇指总弯曲度: {total_thumb_bend:.2f} 弧度")
                
                # 根据总弯曲度判断是否达到预期效果
                if total_thumb_bend < 1.0:
                    print("[WARNING] 大拇指弯曲度不足，建议增加弯曲角度")
                elif total_thumb_bend > 3.0:
                    print("[WARNING] 大拇指弯曲度过大，建议减小弯曲角度")
        
        hand.set_joint_position_target(q_target)
        hand.write_data_to_sim()
    else:
        print(f"[ERROR] Joint count mismatch: expected {hand.num_joints}, got {len(joint_targets)}")

    # 执行仿真步骤
    sim.step()
    t += args.dt
    hand.update(args.dt)

print("[INFO] Index finger up demo ended.")
