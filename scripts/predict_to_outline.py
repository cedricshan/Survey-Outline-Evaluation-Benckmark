#!/usr/bin/env python3
"""
将 predict 字段转换为 outline 格式的脚本

功能：
1. 去除 predict 字段中的 <think>...</think> 标签
2. 解析 JSON 格式的大纲数据
3. 保留原始的大纲结构（不转换为字符串）
4. 输出包含 topic 和 outline 字段的 JSONL 文件

输入格式：
{
  "prompt": "...",
  "predict": "<think>思考过程...</think>[{\"level\": 1, \"numbering\": \"1\", \"title\": \"引言\", \"ref\": [...]}]"
}

输出格式：
{
  "topic": "主题名称",
  "outline": [
    {
      "level": 1,
      "numbering": "1", 
      "title": "章节标题",
      "ref": ["引用1", "引用2"]
    }
  ]
}
"""

import json
import os
import sys
import re
import ast
import io
import tokenize
import argparse
from pathlib import Path

def process_to_list(s: str):
    """
    处理字符串：
    1. 去掉首尾空白和换行
    2. 删除所有 <think>...</think> 内容（不区分大小写，支持跨行）
    3. 返回清理后的文本
    """
    if not isinstance(s, str):
        print("错误: 输入不是字符串")
        return ""

    # 1. 去掉两端空白和换行
    text = s.strip()

    # 2. 删除所有 <think>...</think>（非贪婪匹配，支持跨行）
    text = re.sub(r'<think\b[^>]*?>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()

    return text

def _convert_string_tokens_to_json_style(s: str) -> str:
    """
    将输入 s 中的所有字符串字面量（无论单引号还是双引号）转换为 JSON 风格（双引号且正确转义）。
    其它 token 原样保留。返回转换后的字符串（应该是有效 JSON）。
    """
    out_parts = []
    # tokenize.tokenize 需要 bytes 的 readline
    b = io.BytesIO(s.encode('utf-8'))
    try:
        tokgen = tokenize.tokenize(b.readline)
    except Exception:
        # 在极端异常下直接返回原始字符串
        return s

    for tok in tokgen:
        toknum = tok.type
        tokval = tok.string

        # 跳过编码声明 token（在开始处），以及末尾的 ENDMARKER
        if toknum == tokenize.ENCODING or toknum == tokenize.ENDMARKER:
            continue

        if toknum == tokenize.STRING:
            # tokval 是比如 "'a\\\"b\\n'" 或 "\"a\\\"b\\n\"" 等字面量
            try:
                # 安全地把字面量评估为 Python 字符串值
                py_str = ast.literal_eval(tokval)
            except Exception:
                # 若失败，保留原 token（降级处理）
                out_parts.append(tokval)
                continue
            # 然后用 json.dumps 生成 JSON 风格的双引号字符串（会正确转义）
            json_str = json.dumps(py_str, ensure_ascii=False)
            out_parts.append(json_str)
        else:
            # 对于其它 token，直接使用原始文本（保持格式，如逗号、方括号、大括号、名字、数值等）
            out_parts.append(tokval)
    return "".join(out_parts)

def parse_list_string(s: str):
    """
    尝试把形如 "[{...}, {...}]" 的字符串解析为 Python 列表）。
    支持：
      - 严格 JSON 字符串（使用双引号） -> json.loads
      - Python 字面量（可用单引号、转义、\\n 等） -> ast.literal_eval
      - 混合或不完全 JSON 的情况下，tokenize -> 转换再 json.loads
    失败时抛出 ValueError。
    """
    if not isinstance(s, str):
        raise TypeError("输入必须是字符串")

    s_strip = s.strip()

    # 1) 先尝试 JSON（最快）
    try:
        return json.loads(s_strip)
    except Exception:
        pass

    # 2) 再尝试 ast.literal_eval（支持单引号、Python 字符串转义等）
    try:
        return ast.literal_eval(s_strip)
    except Exception:
        pass

    # 3) 最后用 tokenize 把所有字符串字面量转换为 JSON 风格，然后 json.loads
    converted = _convert_string_tokens_to_json_style(s_strip)
    try:
        return json.loads(converted)
    except Exception as e:
        # 如果仍失败，给出包含原始错误信息的异常，便于 debug
        raise ValueError(f"无法解析输入字符串为列表：{e}\n原始/转换后字符串：\n{converted}") from e

def extract_topic_from_prompt(prompt: str) -> str:
    """从 prompt 字段中提取主题"""
    if not prompt:
        return "Unknown Topic"
    
    # 使用正则表达式提取 Title: 后的内容
    pattern = r"\nTitle:(.*?)\nReferences:"
    match = re.search(pattern, prompt, re.S)  # re.S 让 . 匹配换行符
    if match:
        return match.group(1).strip()
    else:
        return "Unknown Topic"

def process_predict_to_outline(input_file: str, output_file: str):
    """
    处理 predict 字段，转换为 outline 格式
    
    Args:
        input_file: 输入文件路径（包含 predict 字段的格式）
        output_file: 输出文件路径（包含 outline 数组的格式）
    """
    processed_count = 0
    error_count = 0
    
    print(f"开始处理文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析原始数据
                data = json.loads(line)
                
                # 提取主题
                prompt = data.get('prompt', '')
                topic = extract_topic_from_prompt(prompt)
                
                # 处理 predict 字段
                predict_text = data.get("predict", "")
                if not predict_text:
                    print(f"第 {line_num} 行: 缺少 predict 字段")
                    error_count += 1
                    continue
                
                # 去除 think 标签
                cleaned_text = process_to_list(predict_text)
                
                # 解析为大纲列表
                try:
                    outline = parse_list_string(cleaned_text)
                    
                    # 确保是列表格式
                    if not isinstance(outline, list):
                        print(f"第 {line_num} 行: 解析结果不是列表格式")
                        error_count += 1
                        continue
                    
                    # 验证大纲结构
                    valid_outline = []
                    for item in outline:
                        if isinstance(item, dict) and 'title' in item:
                            # 确保必要字段存在
                            processed_item = {
                                "level": item.get("level", 1),
                                "numbering": item.get("numbering", ""),
                                "title": item.get("title", ""),
                                "ref": item.get("ref", [])
                            }
                            valid_outline.append(processed_item)
                    
                    if not valid_outline:
                        print(f"第 {line_num} 行: 没有有效的大纲项目")
                        error_count += 1
                        continue
                    
                    # 构建输出数据
                    output_data = {
                        "topic": topic,
                        "outline": valid_outline
                    }
                    
                    # 写入转换后的数据
                    outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    processed_count += 1
                    
                except Exception as e:
                    print(f"第 {line_num} 行: 解析大纲失败 - {str(e)}")
                    error_count += 1
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行: JSON 解析失败 - {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"第 {line_num} 行: 处理失败 - {e}")
                error_count += 1
                continue
    
    print(f"处理完成: 成功 {processed_count} 条，失败 {error_count} 条")
    return processed_count, error_count

def main():
    parser = argparse.ArgumentParser(description="将 predict 字段转换为 outline 格式")
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    
    processed, errors = process_predict_to_outline(args.input, args.output)
    
    print(f"\n处理结果:")
    print(f"  成功处理: {processed} 条")
    print(f"  处理失败: {errors} 条")
    print(f"  成功率: {processed/(processed+errors)*100:.1f}%" if (processed+errors) > 0 else "  成功率: 0%")

if __name__ == "__main__":
    main()
