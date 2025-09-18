import json
import os
import glob
import sys
import re
import ast
import io
import tokenize

def process_to_list(s: str):
    """
    处理字符串：
    1. 去掉首尾空白和换行
    2. 删除所有 <think>...</think> 内容（不区分大小写，支持跨行）
    3. 将剩余内容解析为列表（优先 JSON，再尝试 ast.literal_eval）
    返回 (bool, list)。若解析失败或解析结果不是列表，打印行数与报错信息并返回 (False, [])。
    """
    if not isinstance(s, str):
        print("行数: 0, 错误: 输入不是字符串")
        return False, []

    # 1. 去掉两端空白和换行
    text = s.strip()

    # 2. 删除所有 <think>...</think>（非贪婪匹配，支持跨行）
    text = re.sub(r'<think\b[^>]*?>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()

    return text
def extract_outermost_simple(s: str):
    """
    从左向右找到第一个 '['，从右向左找到最后一个 ']'。
    若任一端找不到或位置不合理，打印"无法匹配"并返回空列表。
    若匹配到，尝试将包含方括号的子串用 ast.literal_eval 解析为列表并返回解析结果（解析失败返回空列表）。
    """
    left = s.find('[')
    right = s.rfind(']')
    if left == -1 or right == -1 or left > right:
        print("无法匹配")
        return []
    substring = s[left:right+1]
    try:
        parsed = ast.literal_eval(substring)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        print("匹配到非列表")
        print(substring)
        return []
    
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
    
def delete_think(s: str) -> str:
    # 删除从开头到第一个 </think>（包含 </think>）
    text = re.sub(r'^.*?</think>', '', s, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def process_outline(data):
    """将outline字段转换为字符串格式"""
    
    # 将每个outline条目转换为"[level] numbering title"格式
    processed_items = [
        f"{item['numbering']} {item['title']}" 
        for item in data
    ]
    
    # 使用换行符连接所有条目
    return "\n".join(processed_items)

def preprocess_jsonl(input_file, output_file):
    """
    Preprocess the JSONL file to extract 'topic', 'predict', and 'label' fields.
    """
    processed_data = []

    with open(input_file, 'r', encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line)
            # Extract topic from the 'prompt' field which is a list
            topic = ""
            prompt = data.get('prompt', '')
            pattern = r"\nTitle:(.*?)\nReferences:"
            match = re.search(pattern, prompt, re.S)  # re.S 让 . 匹配换行符
            if match:
                topic = match.group(1).strip()
            else:
                print(f"No topic found in prompt")
            
            text = process_to_list(data.get("predict"))
            #text = text.replace('\\\\', '\\')
            # Create a new dictionary with only the required fields
            try:
                processed_entry = {
                    "topic": topic,
                    #"generated": data.get("predict"),
                    "generated":  process_outline(parse_list_string(delete_think(text))),
                }
                processed_data.append(processed_entry)
            except Exception as e:
                # 如果出现错误，打印错误信息
                print(f"出现错误，错误行: {data.get('predict')}")
                print(f"解析失败，错误信息: {e}")

    # Write the processed data to the output file
    with open(output_file, 'w') as outfile:
        for entry in processed_data:
            outfile.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    # If command line arguments are provided, use them
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
        else:
            # Default output: add _processed before .jsonl
            if input_path.endswith('.jsonl'):
                output_path = input_path[:-6] + '_processed.jsonl'
            else:
                output_path = input_path + '_processed.jsonl'
        print(f"Processing {input_path} -> {output_path}")
        preprocess_jsonl(input_path, output_path)
    else:
        # 自动处理data目录下所有.jsonl文件（不含_processed和非原始数据）
        data_dir = os.path.dirname(os.path.abspath(__file__))
        for file in os.listdir(data_dir):
            if file.endswith(".jsonl") and not file.endswith("_processed.jsonl") and not file.startswith("."):
                input_path = os.path.join(data_dir, file)
                output_file = file.replace(".jsonl", "_processed.jsonl")
                output_path = os.path.join(data_dir, output_file)
                print(f"Processing {input_path} -> {output_path}")
                preprocess_jsonl(input_path, output_path)