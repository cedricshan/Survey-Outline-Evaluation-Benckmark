# Outline Generation and Evaluation Pipeline

A comprehensive pipeline for generating and evaluating academic literature review outlines using Large Language Models (LLMs).

## Overview

This system provides an end-to-end solution for:
1. **Outline Generation**: Automatically generating structured outlines from research prompts using OpenAI-compatible APIs
2. **Outline Evaluation**: Assessing outline quality using a sophisticated multi-dimensional evaluation framework
3. **Statistical Analysis**: Providing detailed performance metrics and statistical insights

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)
- Access to OpenAI-compatible API endpoints

### Installation

```bash
# Clone the repository
git clone https://github.com/cedricshan/Survey-Outline-Evaluation-Benckmark.git
cd Outline

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Run the complete pipeline with a single command:

```bash
python run.py \
  --api_url "your-generation-api-url" \
  --api_key "your-generation-api-key" \
  --model "your-generation-model" \
  --num_workers 16 \
  --judge_api_url "your-evaluation-api-url" \
  --judge_api_key "your-evaluation-api-key" \
  --judge_model "your-evaluation-model"
```

## System Architecture

### Core Components

#### 1. Outline Generator (`scripts/genrate_outlines.py`)
- **Purpose**: Generates structured outlines from research prompts
- **Input**: JSON array of prompts with messages
- **Output**: Normalized outline structures in JSONL format
- **Features**:
  - Multi-threaded processing for high throughput
  - Robust error handling and retry logic
  - Connection pooling for API efficiency
  - Detailed logging and debugging capabilities

#### 2. Data Preprocessor (`scripts/eval_preprocessing.py`)
- **Purpose**: Converts generated outlines to evaluation format
- **Input**: Generated outline JSONL files
- **Output**: Evaluation-ready JSONL format
- **Features**:
  - Supports both JSON array and JSONL input formats
  - Converts outline structures to text format
  - Handles missing data gracefully

#### 3. Outline Evaluator (`scripts/evaluate_llm.py`)
- **Purpose**: Evaluates outline quality using LLM-based assessment
- **Input**: Preprocessed evaluation data
- **Output**: Detailed evaluation scores and statistics
- **Features**:
  - Multi-dimensional evaluation framework
  - Comprehensive statistical analysis
  - Detailed logging and error tracking
  - Robust JSON parsing with fallback mechanisms

#### 4. Pipeline Orchestrator (`run.py`)
- **Purpose**: Coordinates the entire generation and evaluation workflow
- **Features**:
  - Sequential execution of all pipeline stages
  - Real-time progress monitoring
  - Comprehensive reporting
  - Error handling and recovery

## Parameter Reference

### Generation Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `--api_url` | string | Yes | OpenAI-compatible API endpoint URL |
| `--api_key` | string | Yes | API authentication key |
| `--model` | string | Yes | Model name for outline generation |
| `--num_workers` | int | No | Number of concurrent worker threads (default: 8) |
| `--timeout` | int | No | API request timeout in seconds (default: 3600) |
| `--dataset_path` | string | No | Input dataset path (default: datasets/test_prompts.json) |
| `--save_dir` | string | Yes | Output directory for generated results |

### Evaluation Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `--judge_api_url` | string | Yes | Evaluation API endpoint URL |
| `--judge_api_key` | string | Yes | Evaluation API authentication key |
| `--judge_model` | string | Yes | Model name for evaluation |
| `--max_workers` | int | No | Number of concurrent evaluation threads (default: 8) |
| `--sample_size` | int | No | Number of items to sample for evaluation |
| `--random_seed` | int | No | Random seed for reproducible sampling (default: 42) |

### Pipeline Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `--api_url` | string | Yes | Generation API URL |
| `--api_key` | string | Yes | Generation API key |
| `--model` | string | Yes | Generation model name |
| `--num_workers` | int | No | Number of worker threads (default: 10) |
| `--judge_api_url` | string | Yes | Evaluation API URL |
| `--judge_api_key` | string | Yes | Evaluation API key |
| `--judge_model` | string | Yes | Evaluation model name |

## Evaluation Framework

### Assessment Philosophy

The evaluation system is designed around readers' core needs when engaging with academic literature reviews:

1. **Information Retrieval**: Readers need to quickly locate specific knowledge details and related theories within the field
2. **Domain Insight**: Readers need to deeply understand domain insights, grasp development prospects and trends to support subsequent research

In this process, the outline plays a crucial role:
- **For retrieval needs**: The outline serves as the core tool for readers to quickly locate information. The clarity of its main structure, the distinctiveness of each section's responsibilities, and the conciseness and specificity of title expressions directly determine readers' efficiency in targeting information.
- **For insight needs**: The outline assumes the function of cognitive guidance. Whether it helps readers construct progressively deeper cognitive chains, whether it summarizes and integrates domain development (rather than simply listing literature), and whether it covers research gaps and future prospects will directly affect readers' establishment of domain cognitive foundations needed for research and their judgment of potential breakthrough directions.

### Six Evaluation Dimensions

Based on the above principles, literature review outline evaluation can be conducted across three major dimensions: **Structure**, **Content**, and **Pragmatics**. We propose six specific, actionable evaluation dimensions:

#### 1. Structure - Information Quick Location
**Purpose**: Evaluates whether the outline follows domain-standard structures (such as IMRaD, chronological order, methodological classification, etc.), ensuring readers can quickly retrieve target information based on chapter logic.

**Rationale**: Schema theory indicates that structures that meet reader expectations can reduce cognitive load and improve readability.

**Scoring Criteria**:
- 0-3: Outline clearly does not follow any standard structure, or multiple structures are mixed causing logical confusion
- 3-6: Outline generally conforms to a certain paradigm but has additional paragraphs, or sections are functionally clear and independent
- 6-10: Outline follows domain-standard structure with reasonable chapter arrangement and logical consistency

#### 2. Structure - Appropriate Detail Level
**Purpose**: Examines whether the outline differentially arranges content based on the importance and complexity of research topics, highlighting core content and streamlining secondary information through the number of sub-titles and hierarchy depth.

**Rationale**: The University of Houston Writing Guide emphasizes avoiding "equal effort" and focusing on key research.

**Scoring Criteria**:
- 0-3: Highly similar number of sub-titles and hierarchy depth across main chapters (no clear core content focus)
- 3-6: Moderate differentiation in detail level
- 6-10: Clear differentiation with core content prominently featured

#### 3. Content - Chapter Mutual Exclusivity
**Purpose**: Tests whether content at the same level has clear boundaries and no thematic overlap, avoiding reader confusion and inefficient information location due to content crossover.

**Rationale**: Based on MECE (Mutually Exclusive, Collectively Exhaustive) theory.

**Scoring Criteria**:
- 0-3: Significant content overlap between chapters
- 3-6: Minor content crossover that affects information location clarity
- 6-10: Clear content boundaries with no overlap

#### 4. Content - Logical Depth
**Purpose**: Judges whether the outline adopts diverse argumentation logic such as "causal relationships" and "theory-application," constructing deep cognitive frameworks through progressive chains between chapters (rather than simple parallel relationships), and appropriately introducing third-level titles in logically complex or content-rich areas.

**Scoring Criteria**:
- 0-3: Parts only have parallel relationships, reflecting topic listing without progression
- 3-6: Rich argumentation logic patterns but no long logical chain clusters, flat hierarchy without depth
- 6-10: Rich argumentation logic patterns with deep logical progression, appropriate third-level titles for enhanced logical hierarchy

#### 5. Content - Academic Value
**Purpose**: Measures whether the outline clearly presents academic contributions such as "research gaps," "new conceptual frameworks," and "future directions," avoiding mere literature accumulation.

**Scoring Criteria**:
- 0-3: No clear academic value indicators
- 3-6: Mentions at least one academic contribution but lacks detailed elaboration
- 6-10: Clearly presents multiple academic contributions with detailed elaboration

#### 6. Pragmatics - Descriptive and Concise
**Purpose**: Evaluates whether titles use concise expressions to precisely reflect the descriptive objects or themes of each chapter, avoiding broad terms like "Concepts" and "Definitions."

**Rationale**: Effective titles allow readers to predict the problem domain covered in the chapter. This also relates to retrievability: including appropriate keywords can improve article visibility in database searches.

**Scoring Criteria**:
- 0-3: Almost no descriptiveness, unable to know specific objects or information about the topic
- 3-6: Has descriptiveness but complex language, titles exceeding 8 words
- 6-10: Descriptive and concise titles with appropriate keyword inclusion

## Output Files

### Generated Files Structure

```
outputs/
├── final_run/
│   ├── generation.normalized.jsonl          # Generated outline results
│   ├── evaluation_input.jsonl               # Preprocessed evaluation input
│   ├── evaluation_results.jsonl             # Evaluation results
│   ├── evaluation_results_failed_responses.json  # Failed evaluation responses
│   └── pipeline_report.txt                  # Pipeline execution report
├── logs/
│   └── evaluation_YYYYMMDD_HHMMSS.log       # Detailed evaluation logs
└── score.json                               # Comprehensive statistical analysis
```

### Key Output Files

#### `score.json`
Contains comprehensive statistical analysis including:
- **Summary Statistics**: Total items, success rates, evaluation time
- **Average Scores**: Mean scores for each evaluation dimension
- **Dimension Statistics**: Detailed statistics for each dimension (mean, std, min, max, median, quartiles)
- **Overall Statistics**: Combined statistics across all dimensions

#### `evaluation_results.jsonl`
Contains individual evaluation results with:
- Topic and item ID
- Detailed evaluation text
- Scores for each dimension
- Success/failure status

#### `pipeline_report.txt`
Provides a comprehensive overview of the pipeline execution including:
- Generation and evaluation counts
- Success rates
- File locations
- Execution summary

## Performance Optimization

### Recommended Settings

- **num_workers**: 8-16 (depending on API rate limits)
- **timeout**: 3600 seconds (for complex generation tasks)
- **max_retries**: 5 (with exponential backoff)

### Best Practices

1. **API Configuration**: Use dedicated API endpoints for generation and evaluation
2. **Rate Limiting**: Monitor API rate limits and adjust worker counts accordingly
3. **Error Handling**: The system includes robust error handling and retry mechanisms
4. **Logging**: Enable detailed logging for debugging and monitoring

## Troubleshooting

### Common Issues

1. **Connection Errors**: Check API endpoints and network connectivity
2. **Rate Limiting**: Reduce `num_workers` or implement longer delays
3. **JSON Parsing Errors**: Check input data format and API response structure
4. **Timeout Errors**: Increase timeout values for complex tasks

### Debug Information

- Check `outputs/logs/` for detailed execution logs
- Review `evaluation_results_failed_responses.json` for failed evaluations
- Examine `debug_failed_responses_*.json` for generation failures
