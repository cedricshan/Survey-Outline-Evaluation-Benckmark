#!/usr/bin/env python3
"""
简单的数据集采样脚本
从 test_prompts.json 中随机采样20条数据
"""

import json
import random

def sample_dataset(input_file, output_file, sample_size=5, random_seed=42):
    """
    从JSON文件中随机采样数据
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        sample_size: 采样数量
        random_seed: 随机种子
    """
    print(f"从 {input_file} 中采样 {sample_size} 条数据...")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据总条数: {len(data)}")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 随机采样
    if sample_size >= len(data):
        print(f"采样数量 {sample_size} 大于等于数据总量 {len(data)}，使用全部数据")
        sampled_data = data
    else:
        sampled_data = random.sample(data, sample_size)
        print(f"成功采样 {len(sampled_data)} 条数据")
    
    # 保存采样结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"采样结果已保存到: {output_file}")
    
    # 显示采样的前几条数据作为预览
    print("\n采样数据预览:")
    for i, item in enumerate(sampled_data[:3]):
        topic = item.get('topic', 'Unknown')
        print(f"  {i+1}. {topic[:50]}...")
    if len(sampled_data) > 3:
        print(f"  ... 还有 {len(sampled_data) - 3} 条数据")

if __name__ == "__main__":
    input_file = "datasets/test_prompts.json"
    output_file = "datasets/test_prompts_sampled.json"
    
    sample_dataset(input_file, output_file, sample_size=5, random_seed=42)
