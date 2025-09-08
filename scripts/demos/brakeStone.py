# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
演示如何在IsaacLab中创建可破坏的墙体和发射物。

此脚本展示了：
- 使用IsaacLab的资产系统创建刚体
- 创建发射物并设置初始速度
- 运行物理模拟
- 使用InteractiveScene管理多个对象

用法:
./isaaclab.sh -p scripts/demos/brakeStone.py --wall_size 5 --simulation_steps 2000
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="演示可破坏墙体的IsaacLab脚本")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
parser.add_argument("--wall_size", type=lambda s: tuple(map(int, s.split(','))), default=(8, 8, 8), help="墙体尺寸，格式为x,y,z")
parser.add_argument("--cube_size", type=float, default=0.2, help="立方体边长")
parser.add_argument("--cube_mass", type=float, default=0.1, help="立方体质量")
parser.add_argument("--projectile_speed", type=float, default=10.0, help="发射物速度")
parser.add_argument("--simulation_steps", type=int, default=2000, help="模拟步数")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """设计场景，包括地面、灯光和创建原点组。"""
    # 地面平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # 灯光
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # 创建墙体组
    prim_utils.create_prim("/World/Wall", "Xform", translation=[0.0, 0.0, 0.0])
    
    # 创建发射物组
    prim_utils.create_prim("/World/Projectiles", "Xform", translation=[0.0, 0.0, 0.0])


def create_wall_cubes():
    """创建墙体立方体。"""
    wall_size = args_cli.wall_size
    cube_size = args_cli.cube_size
    
    # 为每个立方体创建单独的原点组和立方体对象
    cube_objects = []
    cube_origins = []
    
    for i in range(wall_size[0]):
        for j in range(wall_size[1]):
            for k in range(wall_size[2]):
                # 计算位置
                x = (i - wall_size[0] // 2) * cube_size
                y = (j - wall_size[1] // 2) * cube_size
                z = k * cube_size + cube_size / 2.0
                # print(f"[INFO]: 创建立方体在位置: [{x:.2f}, {y:.2f}, {z:.2f}]")
                # 创建原点组
                prim_path = f"/World/Wall/Cube_{i}_{j}_{k}"
                prim_utils.create_prim(prim_path, "Xform", translation=[x, y, z])
                
                # 创建单个立方体配置
                cube_cfg = RigidObjectCfg(
                    prim_path=f"/World/Wall/Cube_{i}_{j}_{k}/Cube",
                    spawn=sim_utils.CuboidCfg(
                        size=(cube_size, cube_size, cube_size),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.cube_mass),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.2), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z)),
                )
                
                # 创建立方体对象
                cube_object = RigidObject(cfg=cube_cfg)
                cube_objects.append(cube_object)
                cube_origins.append([x, y, z])
    
    return cube_objects, torch.tensor(cube_origins, device="cuda")


def create_projectile():
    """创建发射物。"""
    # Calculate the center of the wall
    wall_center_x = 0.0
    wall_center_y = 0.0
    wall_top_z = args_cli.wall_size[2] * args_cli.cube_size

    # Set the projectile's initial position above the wall
    projectile_initial_pos = (0, 0, 10)

    prim_utils.create_prim("/World/Projectiles/Projectile", "Xform", translation=projectile_initial_pos)

    # Update the projectile configuration
    projectile_cfg = RigidObjectCfg(
        prim_path="/World/Projectiles/Projectile/projectile",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=projectile_initial_pos,
            lin_vel=(0.0, args_cli.projectile_speed, 0.0),
        ),
    )
    
    projectile_object = RigidObject(cfg=projectile_cfg)
    
    return projectile_object


def run_simulator(sim: SimulationContext, entities: dict):
    """运行模拟循环。"""
    # 提取场景实体
    wall_cubes = entities["wall_cubes"]
    projectile = entities["projectile"]
    cube_origins = entities["cube_origins"]
    
    # 定义模拟步进
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    print(f"[INFO]: 开始模拟，墙体大小: {args_cli.wall_size}x{args_cli.wall_size}x{args_cli.wall_size}")
    print(f"[INFO]: 发射物初始速度: {args_cli.projectile_speed} m/s")
    print(f"[INFO]: 墙体立方体数量: {len(wall_cubes)}")
    for i, cube in enumerate(wall_cubes):
        root_state = cube.data.default_root_state.clone()
        root_state[0, :3] = cube_origins[i]
        cube.write_root_pose_to_sim(root_state[:, :7])
        cube.write_root_velocity_to_sim(root_state[:, 7:])
        cube.reset()
    projectile_state = projectile.data.default_root_state.clone()
    projectile_state[0, :3] = torch.tensor([0.0, 0, 7.0], device=sim.device)
    projectile_state[0, 7:10] = torch.tensor([0.0, 0.0, 0.0 - args_cli.projectile_speed], device=sim.device)
    projectile.write_root_pose_to_sim(projectile_state[:, :7])
    projectile.write_root_velocity_to_sim(projectile_state[:, 7:])
    projectile.reset()
    
    # 模拟物理
    # while simulation_app.is_running() and count < args_cli.simulation_steps:
    while simulation_app.is_running():
        # 重置
        if count % 350 == 0 and count > 0:
            # 重置计数器
            sim_time = 0.0
            print(f"[INFO]: 第 {count} 步，重置场景...")
            
            # 重置墙体立方体状态
            for i, cube in enumerate(wall_cubes):
                root_state = cube.data.default_root_state.clone()
                root_state[0, :3] = cube_origins[i]
                cube.write_root_pose_to_sim(root_state[:, :7])
                cube.write_root_velocity_to_sim(root_state[:, 7:])
                cube.reset()
            
            # 重置发射物状态
            projectile_state = projectile.data.default_root_state.clone()
            projectile_state[0, :3] = torch.tensor([0.0, 0.0, 7.0], device=sim.device)
            projectile_state[0, 7:10] = torch.tensor([0.0, 0.0, 0.0 - args_cli.projectile_speed], device=sim.device)
            projectile.write_root_pose_to_sim(projectile_state[:, :7])
            projectile.write_root_velocity_to_sim(projectile_state[:, 7:])
            projectile.reset()
        
        # 应用模拟数据
        for cube in wall_cubes:
            cube.write_data_to_sim()
        projectile.write_data_to_sim()
        
        # 执行步骤
        sim.step()
        
        # 更新模拟时间
        sim_time += sim_dt
        count += 1
        
        # 更新缓冲区
        for cube in wall_cubes:
            cube.update(sim_dt)
        projectile.update(sim_dt)
        
        # 打印进度
        if count % 100 == 0:
            projectile_pos = projectile.data.root_pos_w[0]
            print(f"步数: {count}/{args_cli.simulation_steps}, 发射物位置: [{projectile_pos[0]:.2f}, {projectile_pos[1]:.2f}, {projectile_pos[2]:.2f}]")


def main():
    """主函数。"""
    # 加载工具助手
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # 设置主相机
    sim.set_camera_view(eye=[16.0, 16.0, 10.0], target=[0.0, 0.0, 2.0])
    
    # 设计场景
    design_scene()
    
    # 创建墙体立方体
    wall_cubes, cube_origins = create_wall_cubes()
    
    # 创建发射物
    projectile = create_projectile()
    
    # 场景实体
    scene_entities = {
        "wall_cubes": wall_cubes,
        "projectile": projectile,
        "cube_origins": cube_origins,
    }
    
    # 播放模拟器
    sim.reset()
    
    # 现在我们准备好了！
    print("[INFO]: 设置完成...")
    
    # 运行模拟器
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用
    # simulation_app.close()