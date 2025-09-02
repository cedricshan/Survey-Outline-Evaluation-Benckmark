#!/usr/bin/env python3
"""
人类大纲评测脚本

从 datasets/test_prompts.json 中提取 "role": "assistant" 的 content，
转换为评估格式，然后使用 evaluate_llm.py 进行评测。
"""

import os
import json
import subprocess
import sys

def outline_to_text(outline_items):
    """
    将大纲数组转换为文本格式
    
    Args:
        outline_items: 大纲项目列表，每个项目包含 level, numbering, title, ref
        
    Returns:
        str: 格式化的大纲文本
    """
    if not outline_items:
        return ""
    
    text_lines = []
    
    for item in outline_items:
        level = item.get("level", 1)
        numbering = item.get("numbering", "")
        title = item.get("title", "")
        
        # 根据层级添加缩进
        indent = "  " * (level - 1)
        
        # 格式化标题
        if numbering:
            formatted_title = f"{indent}{numbering}. {title}"
        else:
            formatted_title = f"{indent}{title}"
        
        text_lines.append(formatted_title)
    
    return "\n".join(text_lines)

def extract_human_outlines(input_file):
    """
    从 test_prompts.json 中提取 assistant 角色的内容
    
    Args:
        input_file: 输入文件路径
        
    Returns:
        list: 包含人类大纲的列表
    """
    print(f"从 {input_file} 中提取 assistant 内容...")
    
    try:
        # 读取文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 参考 genrate_outlines.py 的解析逻辑
        content = content.strip()
        if not content.endswith(']'):
            content += ']'
        
        # 尝试解析为JSON数组
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                print("输入文件不是JSON数组格式")
                return []
        except json.JSONDecodeError as e:
            print(f"Failed to parse as JSON array: {e}")
            # 如果失败，尝试提取有效的JSON对象
            print("Trying to extract valid JSON objects...")
            data = []
            lines = content.split('\n')
            current_obj = ""
            brace_count = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 计算大括号来找到完整的JSON对象
                brace_count += line.count('{') - line.count('}')
                current_obj += line
                
                if brace_count == 0 and current_obj.strip():
                    try:
                        item = json.loads(current_obj.strip())
                        data.append(item)
                        current_obj = ""
                    except json.JSONDecodeError:
                        print(f"Failed to parse object at line {line_num}")
                        current_obj = ""
        
        if not data:
            print("No valid data items found in dataset")
            return []
        
        print(f"成功解析 {len(data)} 个数据项")
        
        human_outlines = []
        extracted_count = 0
        
        for i, item in enumerate(data, 1):
            try:
                # 检查是否有 messages 字段
                if not isinstance(item, dict) or "messages" not in item:
                    continue
                
                messages = item["messages"]
                if not isinstance(messages, list):
                    continue
                
                # 查找 assistant 角色的消息
                for msg in messages:
                    if msg.get("role") == "assistant" and "content" in msg:
                        content = msg["content"]
                        # 不输出具体细节，只显示进度
                        if i % 10 == 0 or i == 1:
                            print(f"处理进度: {i}/{len(data)}")
                        
                        # 从用户消息中提取主题（使用精确的 "Title:\n" 模式）
                        topic = None
                        for user_msg in messages:
                            if user_msg.get("role") == "user" and "content" in user_msg:
                                user_content = user_msg["content"]
                                # 查找 "Title:\n" 模式
                                if "Title:\n" in user_content:
                                    title_start = user_content.find("Title:\n") + len("Title:\n")
                                    # 提取 Title:\n 后面的内容，直到遇到换行符或 References:
                                    title_end = user_content.find("\n", title_start)
                                    if title_end == -1:
                                        title_end = user_content.find("References:", title_start)
                                    if title_end == -1:
                                        title_end = len(user_content)
                                    
                                    topic = user_content[title_start:title_end].strip()
                                    break
                        
                        # 如果没有找到主题，使用默认主题
                        if not topic:
                            topic = f"Human Outline {i}"
                        
                        # 尝试解析内容为大纲结构，然后转换为文本格式
                        try:
                            import ast
                            # 清理内容，移除可能的markdown标记
                            cleaned_content = content.strip()
                            if cleaned_content.startswith('```') and cleaned_content.endswith('```'):
                                cleaned_content = cleaned_content[3:-3].strip()
                            
                            # 尝试使用ast.literal_eval解析Python字典字符串
                            outline_data = ast.literal_eval(cleaned_content)
                            
                            # 使用outline_to_text函数转换为文本格式
                            generated_text = outline_to_text(outline_data)
                            
                            human_outlines.append({
                                "topic": topic,
                                "generated": generated_text,
                                "id": f"human_{i}"
                            })
                            
                        except (ValueError, SyntaxError) as e:
                            print(f"第 {i} 项: 解析失败，使用原始内容 - {e}")
                            # 如果解析失败，直接使用原始内容
                            human_outlines.append({
                                "topic": topic,
                                "generated": content,
                                "id": f"human_{i}"
                            })
                        extracted_count += 1
                        break  # 找到第一个assistant消息就停止
                    
            except Exception as e:
                print(f"处理第 {i} 项时出错: {e}")
                continue
        
        print(f"成功提取 {extracted_count} 个assistant内容")
        return human_outlines
        
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return []

def main():
    input_file = "datasets/test_prompts.json"
    output_dir = "outputs/run_human"
    output_file = os.path.join(output_dir, "evaluation_input_human.jsonl")
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return 1
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 提取人类大纲
    print("开始提取人类大纲...")
    human_outlines = extract_human_outlines(input_file)
    
    if not human_outlines:
        print("没有找到人类大纲")
        return 1
    
    # 保存为评估输入格式
    print(f"保存到: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for outline in human_outlines:
                f.write(json.dumps(outline, ensure_ascii=False) + '\n')
        
        print(f"✓ 成功保存 {len(human_outlines)} 个人类大纲到 {output_file}")
        
        # 运行评测
        print("开始运行评测...")
        if run_evaluation(output_file, output_dir):
            print("✓ 评测完成")
            return 0
        else:
            print("✗ 评测失败")
            return 1
        
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return 1

def run_evaluation(input_file, output_dir):
    """
    运行大纲评估
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录路径
        
    Returns:
        bool: 评估是否成功
    """
    print("运行评测...")
    
    evaluation_cmd = [
        sys.executable, "scripts/evaluate_llm.py",
        "--input", input_file,
        "--output", os.path.join(output_dir, "evaluation_results_human.jsonl"),
        "--judge_api_url", "https://ark.cn-beijing.volces.com/api/v3",
        "--judge_api_key", "30a70266-37d5-4210-b8a2-34d5fb629230",
        "--judge_model", "ep-20250902131956-2glb8",
        "--max_workers", "16"
    ]
    
    print(f"执行评测命令: {' '.join(evaluation_cmd)}")
    
    try:
        # 运行评估命令，不捕获输出以显示进度条
        result = subprocess.run(
            evaluation_cmd,
            check=True
        )
        
        print("✓ 大纲评估成功完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 大纲评估失败，返回码: {e.returncode}")
        if e.stdout:
            print(f"标准输出: {e.stdout}")
        if e.stderr:
            print(f"错误输出: {e.stderr}")
        return False
        
    except Exception as e:
        print(f"✗ 大纲评估执行异常: {e}")
        return False

if __name__ == "__main__":
    exit(main())
