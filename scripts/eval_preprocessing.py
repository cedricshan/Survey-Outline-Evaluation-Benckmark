#!/usr/bin/env python3
"""
数据预处理脚本：将生成的大纲数据转换为评估脚本需要的格式

将 outputs 文件夹下生成的大纲数据（包含 outline 数组）转换为
evaluate_llm.py 需要的格式（包含 generated 文本字段）
"""

import os
import json
import argparse
import logging
from pathlib import Path

def setup_logging(log_file=None):
    """设置日志配置"""
    if log_file:
        # 如果指定了日志文件，输出到文件和控制台
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - [Preprocessor] - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
    else:
        # 默认只输出到控制台
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Preprocessor] - %(message)s')
    
    return logging.getLogger(__name__)

# 默认日志配置
logger = setup_logging()

def outline_to_text(outline_items):
    """
    将大纲数组转换为文本格式
    
    Args:
        outline_items: 大纲项目列表，每个项目包含 level, number, title, ref
        
    Returns:
        str: 格式化的大纲文本
    """
    if not outline_items:
        return ""
    
    text_lines = []
    
    for item in outline_items:
        level = item.get("level", 1)
        number = item.get("number", "")
        title = item.get("title", "")
        
        # 根据层级添加缩进
        indent = "  " * (level - 1)
        
        # 格式化标题
        if number:
            formatted_title = f"{indent}{number}. {title}"
        else:
            formatted_title = f"{indent}{title}"
        
        text_lines.append(formatted_title)
    
    return "\n".join(text_lines)

def convert_outline_format(input_file, output_file):
    """
    转换大纲数据格式
    
    Args:
        input_file: 输入文件路径（包含 outline 数组的格式）
        output_file: 输出文件路径（包含 generated 文本的格式）
    """
    converted_count = 0
    error_count = 0
    
    logger.info(f"开始转换文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        try:
            # 尝试解析为JSON数组格式
            data_list = json.load(infile)
            if isinstance(data_list, list):
                logger.info(f"检测到JSON数组格式，包含 {len(data_list)} 个项目")
                
                for item_num, data in enumerate(data_list, 1):
                    try:
                        # 提取必要字段
                        topic = data.get("topic", "Unknown Topic")
                        outline = data.get("outline", [])
                        item_id = data.get("id", str(item_num))
                        
                        # 检查数据完整性
                        if not outline:
                            logger.warning(f"第 {item_num} 项: 缺少大纲数据")
                            error_count += 1
                            continue
                        
                        # 转换大纲为文本格式
                        generated_text = outline_to_text(outline)
                        
                        # 构建新的数据格式
                        converted_data = {
                            "topic": topic,
                            "generated": generated_text,
                            "id": item_id
                        }
                        
                        # 写入转换后的数据
                        outfile.write(json.dumps(converted_data, ensure_ascii=False) + "\n")
                        converted_count += 1
                        
                    except Exception as e:
                        logger.error(f"第 {item_num} 项: 处理失败 - {str(e)}")
                        error_count += 1
                        continue
            else:
                logger.error("输入文件不是JSON数组格式")
                return
                
        except json.JSONDecodeError:
            # 如果不是JSON数组，尝试按行解析JSONL格式
            logger.info("尝试按JSONL格式解析")
            infile.seek(0)  # 重置文件指针
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # 解析原始数据
                    data = json.loads(line)
                    
                    # 提取必要字段
                    topic = data.get("topic", "Unknown Topic")
                    outline = data.get("outline", [])
                    item_id = data.get("id", str(line_num))
                    
                    # 检查数据完整性
                    if not outline:
                        logger.warning(f"第 {line_num} 行: 缺少大纲数据")
                        error_count += 1
                        continue
                    
                    # 转换大纲为文本格式
                    generated_text = outline_to_text(outline)
                    
                    # 构建新的数据格式
                    converted_data = {
                        "topic": topic,
                        "generated": generated_text,
                        "id": item_id
                    }
                    
                    # 写入转换后的数据
                    outfile.write(json.dumps(converted_data, ensure_ascii=False) + "\n")
                    converted_count += 1
                    
                except json.JSONDecodeError as e:
                    logger.error(f"第 {line_num} 行: JSON 解析失败 - {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    logger.error(f"第 {line_num} 行: 处理失败 - {e}")
                    error_count += 1
                    continue
    
    logger.info(f"转换完成: 成功 {converted_count} 条，失败 {error_count} 条")
    return converted_count, error_count

def find_generation_files(outputs_dir):
    """
    在 outputs 目录下查找生成的大纲文件
    
    Args:
        outputs_dir: outputs 目录路径
        
    Returns:
        list: 找到的生成文件列表
    """
    generation_files = []
    
    for root, dirs, files in os.walk(outputs_dir):
        for file in files:
            if file == "generation.normalized.jsonl":
                file_path = os.path.join(root, file)
                generation_files.append(file_path)
    
    return generation_files

def main():
    parser = argparse.ArgumentParser(description="将生成的大纲数据转换为评估格式")
    parser.add_argument('--input', type=str, help='输入文件路径（可选，如果不指定则自动查找）')
    parser.add_argument('--output', type=str, help='输出文件路径（可选，如果不指定则自动生成）')
    parser.add_argument('--outputs_dir', type=str, default='outputs', help='outputs目录路径')
    parser.add_argument('--batch', action='store_true', help='批量处理模式，处理所有找到的生成文件')
    parser.add_argument('--log_file', help='Log file path for unified logging')
    
    args = parser.parse_args()
    
    # 设置日志配置
    global logger
    logger = setup_logging(args.log_file)
    
    if args.batch:
        # 批量处理模式
        logger.info("批量处理模式：查找所有生成文件")
        generation_files = find_generation_files(args.outputs_dir)
        
        if not generation_files:
            logger.warning(f"在 {args.outputs_dir} 目录下未找到任何 generation.normalized.jsonl 文件")
            return
        
        logger.info(f"找到 {len(generation_files)} 个生成文件")
        
        total_converted = 0
        total_errors = 0
        
        for input_file in generation_files:
            # 生成输出文件路径
            relative_path = os.path.relpath(input_file, args.outputs_dir)
            dir_name = os.path.dirname(relative_path)
            output_file = os.path.join(args.outputs_dir, dir_name, "evaluation_input.jsonl")
            
            logger.info(f"处理文件: {input_file} -> {output_file}")
            
            converted, errors = convert_outline_format(input_file, output_file)
            total_converted += converted
            total_errors += errors
        
        logger.info(f"批量处理完成: 总计成功 {total_converted} 条，失败 {total_errors} 条")
        
    else:
        # 单文件处理模式
        if not args.input:
            logger.error("单文件模式需要指定 --input 参数")
            return
        
        if not args.output:
            # 自动生成输出文件路径
            input_dir = os.path.dirname(args.input)
            args.output = os.path.join(input_dir, "evaluation_input.jsonl")
        
        logger.info(f"单文件处理: {args.input} -> {args.output}")
        converted, errors = convert_outline_format(args.input, args.output)
        
        logger.info(f"处理完成: 成功 {converted} 条，失败 {errors} 条")

if __name__ == "__main__":
    main()
