# -*- coding: utf-8 -*-
"""
联合模型可视化脚本
评估训练好的 Set Transformer + PCT 模型并进行 3D 可视化
"""
import sys
import os
import torch
import numpy as np
import argparse
import json
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import registration_envs, get_leaf_nodes
from joint_model import JointSelectorPCT
from sliding_window_env import SlidingWindowEnvWrapper, normalize_to_trajectories
import gym
import givenData

# ============================================================
# Plotly 可视化函数
# ============================================================
import plotly.graph_objects as go
import plotly.offline as offline

def hsv_to_rgb(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if h_i == 0: return v, t, p
    elif h_i == 1: return q, v, p
    elif h_i == 2: return p, v, t
    elif h_i == 3: return p, q, v
    elif h_i == 4: return t, p, v
    else: return v, p, q

def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i * (360.0 / max(num_colors, 1))
        rgb = hsv_to_rgb(hue / 360.0, 0.7, 0.9)
        colors.append('rgb({},{},{})'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
    return colors

def plot_cube_plotly(cube_definition, container_def=None, title="", save_path=None):
    """
    绘制 3D 码垛可视化
    
    Args:
        cube_definition: List of [l, w, h, x, y, z] for each box
        container_def: [L, W, H] container dimensions
        title: 图表标题
        save_path: HTML 文件保存路径，None 则自动生成
    """
    fig = go.Figure()

    # 绘制容器
    if container_def is not None:
        x, y, z, l, w, h = 0, 0, 0, container_def[0], container_def[1], container_def[2]
        vertices = [[x,y,z], [x+l,y,z], [x+l,y+w,z], [x,y+w,z],
                    [x,y,z+h], [x+l,y,z+h], [x+l,y+w,z+h], [x,y+w,z+h]]
        
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
            i=[0,0,4,4,1,5,0,4,1,1,2,6], j=[1,2,5,6,2,2,3,3,5,4,3,3], k=[2,3,6,7,5,6,4,7,4,0,6,7],
            color="gray", opacity=0.2, showlegend=False
        ))
        
        edges = [[vertices[i] for i in [0,1,2,3,0]], [vertices[i] for i in [4,5,6,7,4]],
                 [vertices[0],vertices[4]], [vertices[1],vertices[5]], 
                 [vertices[2],vertices[6]], [vertices[3],vertices[7]]]
        for edge in edges:
            ex, ey, ez = zip(*edge)
            fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', 
                                       line=dict(color='black', width=2), showlegend=False))

    annotations_list = []
    color_list = generate_colors(len(cube_definition))
    for i, cube in enumerate(cube_definition):
        l, w, h, x, y, z = cube[:6]
        text_x = x + l / 2
        text_y = y + w / 2
        text_z = z + h / 2
        label = str(cube[6]) if len(cube) > 6 else str(i + 1)
        annotations_list.append([text_x, text_y, text_z, label])
        
        vertices = [[x,y,z], [x+l,y,z], [x+l,y+w,z], [x,y+w,z],
                    [x,y,z+h], [x+l,y,z+h], [x+l,y+w,z+h], [x,y+w,z+h]]
        
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in vertices], y=[v[1] for v in vertices], z=[v[2] for v in vertices],
            i=[0,0,4,4,1,5,0,4,1,1,2,6], j=[1,2,5,6,2,2,3,3,5,4,3,3], k=[2,3,6,7,5,6,4,7,4,0,6,7],
            color=color_list[i], opacity=0.8, showlegend=False
        ))
        
        edges = [[vertices[j] for j in [0,1,2,3,0]], [vertices[j] for j in [4,5,6,7,4]],
                 [vertices[0],vertices[4]], [vertices[1],vertices[5]], 
                 [vertices[2],vertices[6]], [vertices[3],vertices[7]]]
        for edge in edges:
            ex, ey, ez = zip(*edge)
            fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', 
                                       line=dict(color='black', width=1), showlegend=False))

    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            annotations=[
                dict(showarrow=False, x=it[0], y=it[1], z=it[2], text=it[3],
                     font=dict(color="black", size=14), opacity=0.9) for it in annotations_list
            ]
        )
    )
    
    if save_path is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        folder_path = f'./joint_visualization/{date_str}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path = os.path.join(folder_path, f'joint_vis_{time_str}.html')
    
    offline.plot(fig, filename=save_path, auto_open=False)
    print(f"HTML 已保存到: {save_path}")
    return save_path


def get_vis_args():
    """获取可视化参数"""
    parser = argparse.ArgumentParser(description='Joint Model Visualization')
    
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test-dataset', type=str, default='datasets/test.pt', help='Test dataset path')
    parser.add_argument('--num-episodes', type=int, default=-1, help='Number of episodes to visualize (-1 for all)')
    parser.add_argument('--setting', type=int, default=1, help='Experiment setting')
    parser.add_argument('--continuous', action='store_true', help='Use continuous environment')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--window-size', type=int, default=5, help='Window size')
    
    # PCT 参数
    parser.add_argument('--embedding-size', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--gat-layer-num', type=int, default=1)
    parser.add_argument('--internal-node-holder', type=int, default=100)
    parser.add_argument('--leaf-node-holder', type=int, default=50)
    parser.add_argument('--lnes', type=str, default='EMS')
    parser.add_argument('--shuffle', type=bool, default=False)
    
    args = parser.parse_args()
    
    if args.no_cuda:
        args.device = 'cpu'
    
    args.container_size = givenData.container_size
    args.item_size_set = givenData.item_size_set
    
    if args.continuous:
        args.id = 'PctContinuous-v0'
    else:
        args.id = 'PctDiscrete-v0'
    
    if args.setting == 1:
        args.internal_node_length = 6
    elif args.setting == 2:
        args.internal_node_length = 6
    elif args.setting == 3:
        args.internal_node_length = 7
    
    args.normFactor = 1.0 / np.max(args.container_size)
    
    return args


def create_env(args, seed=0):
    """创建评估环境"""
    env = gym.make(
        args.id,
        setting=args.setting,
        container_size=args.container_size,
        item_set=args.item_size_set,
        data_name=None,
        load_test_data=False,
        internal_node_holder=args.internal_node_holder,
        leaf_node_holder=args.leaf_node_holder,
        LNES=args.lnes,
        shuffle=args.shuffle,
        sample_from_distribution=False
    )
    env.seed(seed)
    return env


def visualize(args):
    """运行可视化"""
    # 设置设备
    if isinstance(args.device, int):
        device = torch.device('cuda', args.device) if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 加载测试数据并转换为轨迹格式
    print(f"Loading test dataset from {args.test_dataset}...")
    raw_data = torch.load(args.test_dataset)
    trajectories = normalize_to_trajectories(raw_data)
    total_items = sum(len(t) for t in trajectories)
    if args.num_episodes == -1:
        num_episodes = len(trajectories)
    else:
        num_episodes = min(args.num_episodes, len(trajectories))
    print(f"Loaded {len(trajectories)} trajectories, {total_items} total items")
    print(f"Visualizing {num_episodes} episodes (1 episode = 1 trajectory)")
    
    # 创建模型
    print("Loading model...")
    model = JointSelectorPCT(args, window_size=args.window_size)
    
    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # 创建环境
    env = create_env(args)
    wrapped_env = SlidingWindowEnvWrapper(
        env, trajectories, 
        window_size=args.window_size,
        normFactor=args.normFactor,
        traj_start_idx=0
    )
    
    container_size = list(args.container_size)
    html_files = []
    
    for ep_idx in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {ep_idx + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        obs = wrapped_env.reset(episode_idx=ep_idx)
        done = False
        step_count = 0
        
        # 记录放置的箱子
        placed_boxes = []
        
        while not done:
            # 获取状态
            candidates = torch.FloatTensor(wrapped_env.get_candidates()).unsqueeze(0).to(device)
            container_state = torch.FloatTensor(wrapped_env.get_container_state()).unsqueeze(0).to(device)
            
            # Step 1: Set Transformer 选择候选包裹
            with torch.no_grad():
                selected_idx, selector_log_prob, selector_entropy, selector_value = model.selector(
                    candidates, container_state, deterministic=True
                )
            
            # Step 2: 设置选中的候选并重新生成 observation
            updated_obs = wrapped_env.set_selected_and_get_obs(selected_idx.item())
            updated_pct_obs = torch.FloatTensor(updated_obs).unsqueeze(0).to(device)
            
            # Step 3: PCT 从新的 observation 中选择放置位置
            with torch.no_grad():
                all_nodes, leaf_nodes = get_leaf_nodes(
                    updated_pct_obs,
                    args.internal_node_holder,
                    args.leaf_node_holder
                )
                
                placement_log_prob, placement_idx, pct_entropy, pct_value = model.pct(
                    all_nodes,
                    deterministic=True,
                    normFactor=args.normFactor
                )
            
            # 获取放置动作
            pct_action = leaf_nodes[0, placement_idx.item()].cpu().numpy()
            
            # 记录放置前的箱子信息
            raw_candidates = np.array([list(box)[:3] for box in wrapped_env.candidate_queue])
            selected_box_dims = raw_candidates[selected_idx.item()]  # 真实尺寸
            
            # 执行动作
            obs, reward, done, info = wrapped_env.step(selected_idx.item(), pct_action)
            step_count += 1
            
            # 如果放置成功，从环境中获取实际放置信息
            # Box 类属性: x,y,z = 尺寸(dimensions), lx,ly,lz = 位置(position/lower corner)
            if reward > 0 and hasattr(wrapped_env.base_env, 'space') and len(wrapped_env.base_env.space.boxes) > 0:
                last_box = wrapped_env.base_env.space.boxes[-1]
                placed_boxes.append([
                    float(last_box.x), float(last_box.y), float(last_box.z),     # 尺寸 (l, w, h)
                    float(last_box.lx), float(last_box.ly), float(last_box.lz),  # 位置 (x, y, z)
                    step_count                                                     # 顺序
                ])
                print(f"  Step {step_count}: 放置箱子 尺寸({last_box.x:.1f}, {last_box.y:.1f}, {last_box.z:.1f}) "
                      f"位置({last_box.lx:.1f}, {last_box.ly:.1f}, {last_box.lz:.1f})")
            else:
                print(f"  Step {step_count}: 跳过箱子（放置失败）")
        
        # 记录结果
        ratio = info.get('ratio', 0)
        total_placed = info.get('total_placed', len(placed_boxes))
        total_skipped = info.get('total_skipped', 0)
        
        print(f"\nEpisode {ep_idx + 1} 完成:")
        print(f"  空间利用率: {ratio:.4f} ({ratio*100:.2f}%)")
        print(f"  放置数量: {total_placed}")
        print(f"  跳过数量: {total_skipped}")
        
        # 可视化
        if len(placed_boxes) > 0:
            title = f"Episode {ep_idx + 1} - Ratio: {ratio*100:.2f}% - Placed: {total_placed} - Skipped: {total_skipped}"
            html_path = plot_cube_plotly(placed_boxes, container_size, title=title)
            html_files.append(html_path)
            
            # 保存 JSON 数据
            json_data = {
                'episode': ep_idx,
                'container_size': [float(c) for c in container_size],
                'cube_definition': placed_boxes,  # 已经转成 float 了
                'statistics': {
                    'space_ratio': float(ratio),
                    'packed_count': int(total_placed),
                }
            }
            json_path = html_path.replace('.html', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"  JSON 已保存到: {json_path}")
    
    print(f"\n{'='*50}")
    print(f"可视化完成！共生成 {len(html_files)} 个 HTML 文件")
    for f in html_files:
        print(f"  - {f}")
    print(f"{'='*50}")
    
    return html_files


if __name__ == '__main__':
    registration_envs()
    args = get_vis_args()
    visualize(args)
