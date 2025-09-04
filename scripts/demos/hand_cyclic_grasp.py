# scripts/demos/hand_cyclic_grasp.py
# 循环抓握动作（正弦开合），适配你给的 MJCF 里那些关节命名与角度范围

import argparse
import math
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cyclic grasp demo for V3_8dof_hand_right.")
parser.add_argument("--hand_usd", type=str, required=True, help="Path to the hand USD (converted from your MJCF).")
parser.add_argument("--prim_path", type=str, default="/World/Hand", help="Prim path to place the hand.")
parser.add_argument("--hz", type=float, default=0.5, help="Grasp frequency in Hz (open->close->open takes 2 seconds at 0.5Hz).")
parser.add_argument("--amp_scale", type=float, default=0.9, help="Use 90% of joint range as amplitude.")
parser.add_argument("--dt", type=float, default=0.01, help="Physics dt.")
parser.add_argument("--phase_thumb", type=float, default=0.25, help="Thumb phase lead (0..1 cycles).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 启动 Omniverse/Isaac
app = AppLauncher(args).app
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ========== 关节表与范围 ==========
# 按你 MJCF 的 actuator ctrlrange（单位: rad）
JOINT_UPPER = {
    # thumb (4 DOF)
    "thumb_Joint_1": 2.00,
    "thumb_Joint_2": 1.07,
    "thumb_Joint_3": 1.03,
    "thumb_Joint_4": 1.03,
    # index (3 DOF)
    "index_Joint_1": 1.57,
    "index_Joint_2": 1.57,
    "index_Joint_3": 1.10,
    # middle (3 DOF)
    "middle_Joint_1": 1.57,
    "middle_Joint_2": 1.57,
    "middle_Joint_3": 1.10,
    # ring (3 DOF)
    "ring_Joint_1": 1.57,
    "ring_Joint_2": 1.57,
    "ring_Joint_3": 1.10,
    # little (3 DOF)
    "little_Joint_1": 1.57,
    "little_Joint_2": 1.57,
    "little_Joint_3": 1.10,
}

# 给每个手指一个轻微相位偏移（0..1 周期）
FINGER_PHASE = {
    "thumb": args.phase_thumb,     # 拇指领先一点点，更像抓握
    "index": 0.0,
    "middle": 0.05,
    "ring": 0.10,
    "little": 0.15,
}

# 便于按手指分组
FINGER_GROUPS = {
    "thumb":  ["thumb_Joint_1",  "thumb_Joint_2",  "thumb_Joint_3",  "thumb_Joint_4"],
    "index":  ["index_Joint_1",  "index_Joint_2",  "index_Joint_3"],
    "middle": ["middle_Joint_1", "middle_Joint_2", "middle_Joint_3"],
    "ring":   ["ring_Joint_1",   "ring_Joint_2",   "ring_Joint_3"],
    "little": ["little_Joint_1", "little_Joint_2", "little_Joint_3"],
}
ALL_JOINTS = [j for g in FINGER_GROUPS.values() for j in g]

# ========== 搭场景 ==========
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
if parent_path != "/World":  # /World 已经存在，别再创建
    prim_utils.create_prim(parent_path, "Xform")

# 不手动加载USD，让ArticulationCfg的spawn来处理

# 创建 Articulation 配置
hand_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=args.hand_usd,
        # 添加关节属性配置，限制关节驱动力
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            max_effort=1.0,    # 限制最大驱动力
            max_velocity=2.0,  # 限制最大速度
        ),
        # 添加刚体属性，增加阻尼
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=1.0,   # 限制线性速度
            max_angular_velocity=5.0,  # 限制角速度
            max_depenetration_velocity=1.0,  # 限制去穿透速度
        ),
    ),
    prim_path=f"{args.prim_path}/Robot",  # 回到Robot路径
    articulation_root_prim_path="/base/base",  # 明确指定articulation root
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),  # 将手稍微抬起
        rot=(1.0, 0.0, 0.0, 0.0),  # 确保正确的四元数方向
        joint_pos={".*": 0.0},  # 所有关节初始位置为0，避免初始时的关节冲突
        joint_vel={".*": 0.0},  # 所有关节初始速度为0
    ),
    actuators={
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # 使用正则表达式匹配所有关节
            stiffness=50.0,   # 进一步降低刚性
            damping=50.0,     # 大幅增加阻尼
            effort_limit=1.0, # 限制关节驱动力
            velocity_limit=1.0, # 限制关节速度
        ),
    },
)

hand = Articulation(hand_cfg)


# 摄像机
sim.set_camera_view(eye=[0.35, -0.45, 0.30], target=[0.0, 0.0, 0.15])

# ========== 抓握轨迹：正弦开合 ==========
# 正弦：q(t) = 0.5*A*(1 - cos(2π f t + φ))，范围[0, A]，A为幅度（<=上限）
f = float(args.hz)
amp_scale = float(args.amp_scale)
two_pi = 2.0 * math.pi

# 为每个关节准备幅度 A（不超过上限的比例）
AMP = {j: JOINT_UPPER[j] * amp_scale for j in ALL_JOINTS}

def phase_of_joint(jname: str) -> float:
    for fname, lst in FINGER_GROUPS.items():
        if jname in lst:
            return FINGER_PHASE[fname]
    return 0.0

# ========== 主循环 ==========
sim.reset()
print("[INFO] Hand cyclic grasp demo started.")
print(f"[INFO] Number of joints found: {hand.num_joints}")
print(f"[INFO] Joint names: {hand.joint_names}")

# 创建与实际关节数量匹配的目标映射
actual_joint_names = hand.joint_names
joint_target_map = {}

# 为实际存在的关节创建目标映射
for joint_name in actual_joint_names:
    if joint_name in JOINT_UPPER:
        joint_target_map[joint_name] = JOINT_UPPER[joint_name] * amp_scale
    else:
        print(f"[WARNING] Joint {joint_name} not found in JOINT_UPPER, using default range 0.5")
        joint_target_map[joint_name] = 0.5  # 默认范围

print(f"[INFO] Joint target mapping: {joint_target_map}")

t = 0.0
reset_counter = 0
last_reset_time = 0.0

while app.is_running():
    current_time = reset_counter * args.dt
    
    # 每2秒或检测到异常位置时重置
    if (current_time - last_reset_time > 2.0) or reset_counter == 0:
        # 检查手的位置是否异常
        hand_pos = hand.data.root_pos_w[0]
        pos_magnitude = torch.norm(hand_pos)
        
        if pos_magnitude > 5.0 or reset_counter == 0:  # 如果位置过远或初始状态
            if reset_counter > 0:
                print(f"[WARNING] Hand position abnormal: {hand_pos} (magnitude: {pos_magnitude:.2f}), performing full reset...")
            
            # 完全重置仿真
            sim.reset()
            
            # 重新设置手的状态
            root_state = hand.data.default_root_state.clone()
            root_state[:, :3] = torch.tensor([0.0, 0.0, 0.15], device=sim.device)
            root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)  # 四元数
            root_state[:, 7:] = 0.0  # 清零速度
            hand.write_root_state_to_sim(root_state)
            
            # 重置关节状态到中性位置
            joint_pos = torch.zeros((hand.num_instances, hand.num_joints), device=sim.device)
            joint_vel = torch.zeros((hand.num_instances, hand.num_joints), device=sim.device)
            hand.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # 重置内部状态
            hand.reset()
            
            # 重置时间
            t = 0.0
            last_reset_time = current_time
            
            print("[INFO] Full reset completed")
            
    reset_counter += 1

    # 生成温和的关节目标
    q_target = []
    for joint_name in actual_joint_names:
        # 获取该关节所属的手指类型以确定相位
        finger_type = None
        for fname, joint_list in FINGER_GROUPS.items():
            if joint_name in joint_list:
                finger_type = fname
                break
        
        phi = FINGER_PHASE.get(finger_type, 0.0) if finger_type else 0.0
        amp = joint_target_map.get(joint_name, 0.5)
        
        # 使用更温和的正弦曲线，限制范围
        arg = two_pi * f * t + two_pi * phi
        q = 0.3 * amp * (1.0 - math.cos(arg))  # 减少到30%的范围
        q = max(0.0, min(q, amp * 0.5))  # 额外限制到50%的最大范围
        q_target.append(q)
    
    # 转换为张量并应用
    if len(q_target) == hand.num_joints:
        q_target_tensor = torch.tensor([q_target], dtype=torch.float32, device=sim.device)
        
        # 温和地设置目标
        hand.set_joint_position_target(q_target_tensor)
        hand.write_data_to_sim()
    else:
        print(f"[ERROR] Joint count mismatch: expected {hand.num_joints}, got {len(q_target)}")

    # 执行仿真步骤
    sim.step()
    t += args.dt

    # 更新手的状态
    hand.update(args.dt)
