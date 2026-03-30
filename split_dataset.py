# -*- coding: utf-8 -*-
"""
数据集分割脚本
将扁平物品流 all_logs_dataset.pt 分割为训练集、验证集、测试集
"""
import torch
import os
import random
import numpy as np

def split_dataset(source_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    分割数据集为训练、验证、测试集
    
    Args:
        source_path: 原始数据集路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 设置随机种子确保可重复性
    random.seed(seed)
    np.random.seed(seed)
    
    # 加载原始数据集
    print(f"Loading dataset from {source_path}...")
    data = torch.load(source_path)
    total_items = len(data)
    print(f"Total items: {total_items}")
    
    # 打乱数据集
    indices = list(range(total_items))
    random.shuffle(indices)
    
    # 计算分割点
    train_end = int(total_items * train_ratio)
    val_end = train_end + int(total_items * val_ratio)
    
    # 分割数据
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    print(f"Train items: {len(train_data)}")
    print(f"Validation items: {len(val_data)}")
    print(f"Test items: {len(test_data)}")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存分割后的数据集
    train_path = os.path.join(output_dir, 'train.pt')
    val_path = os.path.join(output_dir, 'val.pt')
    test_path = os.path.join(output_dir, 'test.pt')
    
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(test_data, test_path)
    
    print(f"\nDatasets saved to {output_dir}:")
    print(f"  - train.pt: {len(train_data)} items")
    print(f"  - val.pt: {len(val_data)} items")
    print(f"  - test.pt: {len(test_data)} items")
    
    # 统计信息
    all_items = np.array(data)
    
    print(f"\nItem dimension statistics (w, h, d):")
    print(f"  Min: {all_items.min(axis=0)}")
    print(f"  Max: {all_items.max(axis=0)}")
    print(f"  Mean: {all_items.mean(axis=0).round(2)}")
    print(f"  Std: {all_items.std(axis=0).round(2)}")
    
    return train_data, val_data, test_data


if __name__ == '__main__':
    # 分割数据集
    source_path = 'all_logs_dataset.pt'
    output_dir = 'datasets'
    
    split_dataset(source_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
