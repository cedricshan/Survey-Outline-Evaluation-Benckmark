#!/usr/bin/env python3
"""
完整的大纲生成和评测流水线

使用示例:
python run.py \
  --api_url "https://ark.cn-beijing.volces.com/api/v3" \
  --api_key "your-api-key" \
  --model "ep-20250530104326-cc6vk" \
  --num_workers 10 \
  --judge_api_url "https://ark.cn-beijing.volces.com/api/v3" \
  --judge_api_key "your-judge-api-key" \
  --judge_model "ep-20250530104326-cc6vk"
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """运行命令并处理错误"""
    logger.info(f"执行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")
    
    try:
        # 使用实时输出而不是等待完成
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时显示输出
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(f"[{description}] {line}")
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            logger.info(f"✓ {description} 成功完成")
            return True
        else:
            logger.error(f"✗ {description} 失败，返回码: {return_code}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} 失败")
        logger.error(f"错误代码: {e.returncode}")
        if e.stdout:
            logger.error(f"标准输出: {e.stdout}")
        if e.stderr:
            logger.error(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} 执行异常: {e}")
        return False

def run_pipeline(args):
    """
    运行完整的生成和评测流水线
    
    Args:
        args: 命令行参数
    """
    logger.info("=" * 60)
    logger.info("开始运行完整的大纲生成和评测流水线")
    logger.info("=" * 60)
    
    # 创建输出目录
    output_dir = "outputs/final_run"
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 大纲生成
    logger.info("步骤1: 大纲生成")
    logger.info("-" * 40)
    
    generation_cmd = [
        sys.executable, "scripts/genrate_outlines.py",
        "--api_url", args.api_url,
        "--api_key", args.api_key,
        "--model", args.model,
        "--save_dir", output_dir,
        "--dataset_path", "datasets/test_prompts.json",
        "--num_workers", str(args.num_workers),
        "--timeout", "3600"
    ]
    
    if not run_command(generation_cmd, "大纲生成"):
        logger.error("大纲生成失败，终止流水线")
        return False
    
    # 检查生成结果
    generation_output_file = os.path.join(output_dir, "generation.normalized.jsonl")
    if not os.path.exists(generation_output_file):
        logger.error(f"生成输出文件不存在: {generation_output_file}")
        return False
    
    # 步骤2: 数据预处理
    logger.info("步骤2: 数据预处理")
    logger.info("-" * 40)
    
    preprocess_cmd = [
        sys.executable, "scripts/eval_preprocessing.py",
        "--input", generation_output_file,
        "--output", os.path.join(output_dir, "evaluation_input.jsonl")
    ]
    
    if not run_command(preprocess_cmd, "数据预处理"):
        logger.error("数据预处理失败，终止流水线")
        return False
    
    # 检查预处理结果
    evaluation_input_file = os.path.join(output_dir, "evaluation_input.jsonl")
    if not os.path.exists(evaluation_input_file):
        logger.error(f"预处理输出文件不存在: {evaluation_input_file}")
        return False
    
    # 步骤3: 大纲评估
    logger.info("步骤3: 大纲评估")
    logger.info("-" * 40)
    
    evaluation_cmd = [
        sys.executable, "scripts/evaluate_llm.py",
        "--input", evaluation_input_file,
        "--output", os.path.join(output_dir, "evaluation_results.jsonl"),
        "--judge_api_url", args.judge_api_url,
        "--judge_api_key", args.judge_api_key,
        "--judge_model", args.judge_model,
        "--max_workers", str(args.num_workers)
    ]
    
    if not run_command(evaluation_cmd, "大纲评估"):
        logger.error("大纲评估失败")
        return False
    
    # 检查评估结果
    evaluation_output_file = os.path.join(output_dir, "evaluation_results.jsonl")
    if not os.path.exists(evaluation_output_file):
        logger.error(f"评估输出文件不存在: {evaluation_output_file}")
        return False
    
    # 步骤4: 生成最终报告
    logger.info("步骤4: 生成最终报告")
    logger.info("-" * 40)
    
    # 统计生成结果
    generation_count = 0
    with open(generation_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                generation_count += 1
    
    # 统计评估结果
    evaluation_count = 0
    success_count = 0
    with open(evaluation_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    import json
                    data = json.loads(line)
                    # 只统计实际的大纲评估结果，排除统计信息
                    if "scores" in data and "topic" in data and "id" in data:
                        evaluation_count += 1
                        success_count += 1
                except:
                    pass
    
    # 生成报告
    report_file = os.path.join(output_dir, "pipeline_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("大纲生成和评测流水线报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成的大纲数量: {generation_count}\n")
        f.write(f"评估的大纲数量: {evaluation_count}\n")
        f.write(f"成功评估的数量: {success_count}\n")
        f.write(f"评估成功率: {success_count/evaluation_count*100:.1f}% (如果evaluation_count > 0)\n\n")
        f.write("输出文件:\n")
        f.write(f"- 生成结果: {generation_output_file}\n")
        f.write(f"- 评估结果: {evaluation_output_file}\n")
        f.write(f"- 失败响应: {evaluation_output_file.replace('.jsonl', '_failed_responses.json')}\n")
        f.write(f"- 详细统计: outputs/score.json\n")
        f.write(f"- 评估日志: outputs/logs/evaluation_*.log\n")
        f.write(f"- 本报告: {report_file}\n")
    
    logger.info("=" * 60)
    logger.info("流水线完成！")
    logger.info("=" * 60)
    logger.info(f"生成的大纲数量: {generation_count}")
    logger.info(f"评估的大纲数量: {evaluation_count}")
    logger.info(f"成功评估的数量: {success_count}")
    if evaluation_count > 0:
        logger.info(f"评估成功率: {success_count/evaluation_count*100:.1f}%")
    logger.info(f"详细报告: {report_file}")
    logger.info(f"详细统计: outputs/score.json")
    logger.info(f"评估日志: outputs/logs/evaluation_*.log")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="完整的大纲生成和评测流水线")
    parser.add_argument('--api_url', type=str, required=True, help='生成API的URL')
    parser.add_argument('--api_key', type=str, required=True, help='生成API的密钥')
    parser.add_argument('--model', type=str, required=True, help='生成模型名称')
    parser.add_argument('--num_workers', type=int, default=10, help='并发工作线程数')
    parser.add_argument('--judge_api_url', type=str, required=True, help='评估API的URL')
    parser.add_argument('--judge_api_key', type=str, required=True, help='评估API的密钥')
    parser.add_argument('--judge_model', type=str, required=True, help='评估模型名称')
    
    args = parser.parse_args()
    
    # 检查必要文件是否存在
    required_files = [
        "scripts/genrate_outlines.py",
        "scripts/eval_preprocessing.py", 
        "scripts/evaluate_llm.py",
        "datasets/test_prompts.json"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"必要文件不存在: {file_path}")
            return 1
    
    # 运行流水线
    success = run_pipeline(args)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
