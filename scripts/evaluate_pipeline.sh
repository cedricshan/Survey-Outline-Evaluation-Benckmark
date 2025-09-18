#!/bin/bash

# 流水化评测脚本
# 用法: ./evaluate_pipeline.sh <输入文件路径>
# 示例: ./evaluate_pipeline.sh temp/8b_test/8b_rl_lr5e6/result_8b_rl_lr5e6.jsonl

set -e  # 遇到错误立即退出

# 检查参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <输入文件路径>"
    echo "示例: $0 temp/8b_test/8b_rl_lr5e6/result_8b_rl_lr5e6.jsonl"
    exit 1
fi

INPUT_FILE="$1"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件 '$INPUT_FILE' 不存在"
    exit 1
fi

# 获取输入文件的目录和文件名（不含扩展名）
INPUT_DIR=$(dirname "$INPUT_FILE")
INPUT_BASENAME=$(basename "$INPUT_FILE" .jsonl)

echo "=========================================="
echo "开始流水化评测"
echo "输入文件: $INPUT_FILE"
echo "输出目录: $INPUT_DIR"
echo "=========================================="

# 步骤1: 数据预处理（转换为文本格式）
echo "步骤1: 数据预处理..."
PROCESSED_FILE="$INPUT_DIR/${INPUT_BASENAME}_processed.jsonl"
python scripts/generated_preprocessing_new.py "$INPUT_FILE" "$PROCESSED_FILE"
echo "✓ 预处理完成: $PROCESSED_FILE"

# 步骤2: 转换为outline格式（用于与人类大纲比较）
echo "步骤2: 转换为outline格式..."
OUTLINE_FILE="$INPUT_DIR/${INPUT_BASENAME}_outline.jsonl"
python scripts/predict_to_outline.py --input "$INPUT_FILE" --output "$OUTLINE_FILE"
echo "✓ outline格式转换完成: $OUTLINE_FILE"

# 步骤3: 与人类大纲比较
echo "步骤3: 与人类大纲比较..."
REWARDS_FILE="$INPUT_DIR/${INPUT_BASENAME}_vs_human.rewards.jsonl"
python scripts/evaluate_pair_rewards.py \
    --human_file datasets/human_generation.normalized.jsonl \
    --model_file "$OUTLINE_FILE" \
    --output "$REWARDS_FILE" \
    --max_workers 10
echo "✓ 与人类大纲比较完成: $REWARDS_FILE"

# 步骤4: LLM评测（6个维度评分）
echo "步骤4: LLM评测..."
EVALUATION_FILE="$INPUT_DIR/${INPUT_BASENAME}_evaluation_results.jsonl"
LOG_FILE="$INPUT_DIR/evaluation_${INPUT_BASENAME}.log"
python scripts/evaluate_llm.py \
    --input "$PROCESSED_FILE" \
    --output "$EVALUATION_FILE" \
    --judge_api_url "https://ark.cn-beijing.volces.com/api/v3" \
    --judge_api_key "30a70266-37d5-4210-b8a2-34d5fb629230" \
    --judge_model "ep-20250902131956-2glb8" \
    --max_workers 10 \
    --log_file "$LOG_FILE"
echo "✓ LLM评测完成: $EVALUATION_FILE"

echo "=========================================="
echo "流水化评测完成！"
echo "生成的文件："
echo "  - 预处理文件: $PROCESSED_FILE"
echo "  - outline文件: $OUTLINE_FILE"
echo "  - 人类比较结果: $REWARDS_FILE"
echo "  - LLM评测结果: $EVALUATION_FILE"
echo "  - 评测日志: $LOG_FILE"
echo "=========================================="
