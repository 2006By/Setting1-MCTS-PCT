"""
MCTS-PCT 码垛结果可视化辅助函数

这个文件提供了从JSON文件加载MCTS-PCT结果并进行可视化的功能。
可以直接在 Jupyter Notebook 中导入使用。

使用方法:
    from mcts_visualize import load_mcts_result, visualize_mcts_episode, list_all_episodes

    # 1. 可视化单个episode
    visualize_mcts_episode('./mcts_results/2026-01-17/episode_0_12_30_45.json')

    # 2. 列出所有可用的episode文件
    list_all_episodes('./mcts_results')

    # 3. 可视化目录中最新的episode
    visualize_latest_episode('./mcts_results')
"""

import json
import os
import glob
from datetime import datetime

# ============================================================================
# 以下函数用于加载和处理MCTS-PCT的结果
# ============================================================================

def load_mcts_result(json_path):
    """
    从JSON文件加载MCTS-PCT码垛结果
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        dict: 包含 cube_definition, container_size, statistics 的字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def list_all_episodes(result_dir='./mcts_results'):
    """
    列出所有可用的episode结果文件
    
    Args:
        result_dir: 结果目录
        
    Returns:
        list: 所有JSON文件路径列表（按时间排序）
    """
    if not os.path.exists(result_dir):
        print(f"目录不存在: {result_dir}")
        return []
    
    json_files = []
    for root, dirs, files in os.walk(result_dir):
        for f in files:
            if f.endswith('.json'):
                json_files.append(os.path.join(root, f))
    
    # 按修改时间排序
    json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"找到 {len(json_files)} 个episode结果文件:")
    for i, f in enumerate(json_files[:10]):  # 只显示前10个
        data = load_mcts_result(f)
        stats = data.get('statistics', {})
        print(f"  [{i}] {os.path.basename(f)} - "
              f"Episode {data['episode']}, "
              f"{stats.get('packed_count', '?')}/{stats.get('total_items', '?')} items, "
              f"{stats.get('space_ratio', 0)*100:.1f}%")
    
    if len(json_files) > 10:
        print(f"  ... 还有 {len(json_files) - 10} 个文件")
    
    return json_files


def visualize_mcts_episode(json_path, plot_func=None, container_size=None):
    """
    可视化单个MCTS-PCT episode结果
    
    Args:
        json_path: JSON文件路径
        plot_func: 可视化函数 (默认使用 plot_cube_plotly)
        container_size: 容器尺寸 (如果不指定则从JSON读取)
    """
    data = load_mcts_result(json_path)
    
    cube_definition = data['cube_definition']
    if container_size is None:
        container_size = data['container_size']
    
    stats = data.get('statistics', {})
    print(f"\n========== Episode {data['episode'] + 1} ==========")
    print(f"装入物品: {stats.get('packed_count', len(cube_definition))}/{stats.get('total_items', '?')}")
    print(f"空间利用率: {stats.get('space_ratio', 0)*100:.2f}%")
    print(f"总奖励: {stats.get('total_reward', '?')}")
    print(f"时间戳: {stats.get('timestamp', '?')}")
    print(f"容器尺寸: {container_size}")
    print(f"="*40)
    
    if plot_func is not None:
        plot_func(cube_definition, container_size)
    else:
        print("\n提示: 请传入 plot_cube_plotly 函数来进行可视化")
        print("示例: visualize_mcts_episode('path/to/file.json', plot_cube_plotly)")
    
    return cube_definition, container_size


def visualize_latest_episode(result_dir='./mcts_results', plot_func=None):
    """
    可视化最新的episode结果
    
    Args:
        result_dir: 结果目录
        plot_func: 可视化函数
    """
    files = list_all_episodes(result_dir)
    if files:
        return visualize_mcts_episode(files[0], plot_func)
    return None, None


# ============================================================================
# 以下代码可以直接复制到 Jupyter Notebook 中使用
# ============================================================================

NOTEBOOK_CODE = '''
# ==================== MCTS-PCT 可视化代码 ====================
# 将以下代码复制到你的 notebook 的新 cell 中

import json
import os

def load_mcts_result(json_path):
    """从JSON文件加载MCTS-PCT码垛结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def visualize_mcts_episode(json_path, container_size=None):
    """
    可视化MCTS-PCT episode结果
    
    用法:
        visualize_mcts_episode('./mcts_results/2026-01-17/episode_0_12_30_45.json')
    """
    data = load_mcts_result(json_path)
    
    # 直接使用保存的 cube_definition，格式已经是 [l, w, h, x, y, z, label]
    cube_definition = data['cube_definition']
    if container_size is None:
        container_size = data['container_size']
    
    # 打印统计信息
    stats = data.get('statistics', {})
    print(f"\\n========== Episode {data['episode'] + 1} ==========")
    print(f"装入物品: {stats.get('packed_count', len(cube_definition))}/{stats.get('total_items', '?')}")
    print(f"空间利用率: {stats.get('space_ratio', 0)*100:.2f}%")
    print(f"容器尺寸: {container_size}")
    print(f"="*40)
    
    # 调用你的可视化函数
    plot_cube_plotly(cube_definition, container_size)
    
    return cube_definition, container_size

def list_mcts_episodes(result_dir='./mcts_results'):
    """列出所有可用的episode结果"""
    import glob
    files = glob.glob(f'{result_dir}/**/*.json', recursive=True)
    files.sort(key=os.path.getmtime, reverse=True)
    print(f"找到 {len(files)} 个文件:")
    for i, f in enumerate(files[:10]):
        print(f"  [{i}] {f}")
    return files

# ==================== 使用示例 ====================
# 1. 列出所有episode
# files = list_mcts_episodes('./mcts_results')

# 2. 可视化某个episode (修改文件路径)
# visualize_mcts_episode('./mcts_results/2026-01-17/episode_0_12_30_45.json')

# 3. 或者直接手动加载并可视化
# data = load_mcts_result('./mcts_results/2026-01-17/episode_0_12_30_45.json')
# plot_cube_plotly(data['cube_definition'], data['container_size'])
'''

def print_notebook_code():
    """打印可以复制到 Notebook 中的代码"""
    print(NOTEBOOK_CODE)


if __name__ == '__main__':
    print("MCTS-PCT 可视化辅助模块")
    print("\n你可以将以下代码复制到你的 Jupyter Notebook 中使用:")
    print_notebook_code()
