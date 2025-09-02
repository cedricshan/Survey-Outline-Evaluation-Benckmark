#!/usr/bin/env python3
"""
Outline Generation Script

This script generates outlines from prompts using OpenAI-compatible APIs.
It processes JSON array input files and outputs standardized outline structures.
"""

import os
import json
import logging
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from tqdm import tqdm

def setup_logging(log_file=None):
    """设置日志配置"""
    if log_file:
        # 如果指定了日志文件，输出到文件和控制台
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - [OutlineGenerator] - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
    else:
        # 默认只输出到控制台
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [OutlineGenerator] - %(message)s')
    
    return logging.getLogger(__name__)

# 默认日志配置
logger = setup_logging()

# Global lock for thread-safe file operations
lock = threading.Lock()

class OutlineGenerator:
    """Main class for generating outlines from prompts."""
    
    def __init__(self, api_url: str, api_key: str, model: str, timeout: int = 3600):
        """
        Initialize the outline generator.
        
        Args:
            api_url: OpenAI-compatible API endpoint
            api_key: API key for authentication
            model: Model name to use for generation
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
        # 统一使用 OpenAI 客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key, 
                base_url=api_url,
                timeout=timeout
            )
            logger.info(f"Using OpenAI client for API: {api_url}")
        except ImportError:
            logger.error("OpenAI library not found. Please install with: pip install openai")
            raise ImportError("OpenAI library required")
    
    def filter_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter messages to only include system and user roles.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Filtered list containing only system and user messages
        """
        filtered = []
        for msg in messages:
            if msg.get("role") in ["system", "user"]:
                filtered.append(msg)
        return filtered
    
    def normalize_outline(self, raw_response: str, item_id: str = "unknown", output_dir: str = ".") -> Dict[str, Any]:
        """
        Normalize the model response to standard outline format.
        
        Args:
            raw_response: Raw response from the model
            item_id: ID of the item being processed (for debugging)
            output_dir: Output directory for debug files
            
        Returns:
            Normalized outline structure
        """
        try:
            # Try to parse as JSON
            if isinstance(raw_response, str):
                # Remove markdown code blocks if present
                raw_response = raw_response.strip()
                if raw_response.startswith('```json'):
                    raw_response = raw_response[7:]
                if raw_response.endswith('```'):
                    raw_response = raw_response[:-3]
                raw_response = raw_response.strip()
                
                data = json.loads(raw_response)
            else:
                data = raw_response
            
            # Handle list responses - extract first element if it's a list
            if isinstance(data, list):
                if len(data) > 0:
                    data = data[0]  # Take the first element
                else:
                    return {
                        "topic": "Unknown Topic",
                        "outline": [],
                        "schema_version": "outline.v1",
                        "format_ok": False,
                        "error": "Empty list response"
                    }
            
            # Extract topic and outline
            topic = data.get("topic", "Unknown Topic")
            outline_data = data.get("outline", [])
            
            # Normalize outline structure
            normalized_outline = []
            for item in outline_data:
                normalized_item = {
                    "level": int(item.get("level", 1)),
                    "number": str(item.get("number", item.get("numbering", "1"))),  # Handle both "number" and "numbering"
                    "title": str(item.get("title", "")),
                    "ref": item.get("ref", []) if isinstance(item.get("ref"), list) else []
                }
                normalized_outline.append(normalized_item)
            
            return {
                "topic": topic,
                "outline": normalized_outline,
                "schema_version": "outline.v1",
                "format_ok": True
            }
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # Log detailed error information for debugging
            logger.error(f"=== PARSING FAILED FOR ITEM {item_id} ===")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {str(e)}")
            logger.error(f"Raw response type: {type(raw_response)}")
            logger.error(f"Raw response length: {len(str(raw_response))}")
            logger.error(f"Raw response content:")
            logger.error(f"{str(raw_response)}")
            logger.error(f"=== END PARSING FAILED FOR ITEM {item_id} ===")
            
            # Save failed response to debug file
            try:
                debug_file = os.path.join(output_dir, f"debug_failed_responses_{item_id}.json")
                debug_data = {
                    "item_id": item_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "raw_response": raw_response,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(debug_file, 'w', encoding='utf-8') as f:
                    json.dump(debug_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Failed response saved to: {debug_file}")
            except Exception as save_error:
                logger.error(f"Failed to save debug file: {save_error}")
            
            logger.warning(f"Failed to normalize response for item {item_id}: {e}")
            return {
                "topic": "Unknown Topic",
                "outline": [],
                "schema_version": "outline.v1",
                "format_ok": False,
                "error": str(e)
            }
    
    def call_api(self, messages: List[Dict[str, str]], max_retries: int = 5) -> Optional[str]:
        """
        Call the API using OpenAI client with retry logic.
        
        Args:
            messages: List of messages to send
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response content or None if failed
        """
        # Add JSON format instruction to the last user message
        modified_messages = messages.copy()
        for i in range(len(modified_messages) - 1, -1, -1):
            if modified_messages[i]["role"] == "user":
                modified_messages[i]["content"] += "\n\nPlease respond in JSON format with the following structure: {\"topic\": \"...\", \"outline\": [{\"level\": 1, \"number\": \"1\", \"title\": \"...\", \"ref\": []}, ...]}"
                break
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=modified_messages,
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # 记录错误详情
                logger.warning(f"API call failed on attempt {attempt + 1}/{max_retries}: {type(e).__name__}: {str(e)}")
                
                # 根据错误类型决定是否重试
                if "rate limit" in error_msg or "429" in error_msg:
                    # Rate limit - 指数退避重试
                    wait_time = min(2 ** attempt, 60)  # 最大等待60秒
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                    
                elif "timeout" in error_msg or "timed out" in error_msg:
                    # 超时错误 - 指数退避重试
                    wait_time = min(2 ** attempt, 30)  # 最大等待30秒
                    logger.warning(f"Request timeout, waiting {wait_time}s before retry")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    continue
                    
                elif "server error" in error_msg or "500" in error_msg or "502" in error_msg or "503" in error_msg or "504" in error_msg:
                    # 服务器错误 - 指数退避重试
                    wait_time = min(2 ** attempt, 60)  # 最大等待60秒
                    logger.warning(f"Server error, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                    
                elif "quota" in error_msg or "billing" in error_msg or "payment" in error_msg:
                    # 配额/计费错误 - 不重试
                    logger.error(f"Quota/billing error, not retrying: {str(e)}")
                    return None
                    
                elif "invalid" in error_msg or "bad request" in error_msg or "400" in error_msg:
                    # 请求参数错误 - 不重试
                    logger.error(f"Invalid request error, not retrying: {str(e)}")
                    return None
                    
                elif "authentication" in error_msg or "unauthorized" in error_msg or "401" in error_msg or "403" in error_msg:
                    # 认证错误 - 不重试
                    logger.error(f"Authentication error, not retrying: {str(e)}")
                    return None
                    
                else:
                    # 其他未知错误 - 指数退避重试
                    wait_time = min(2 ** attempt, 30)  # 最大等待30秒
                    logger.warning(f"Unknown error, waiting {wait_time}s before retry")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    continue
        
        # 所有重试都失败了
        logger.error(f"All {max_retries} attempts failed for API call")
        return None
    
    def process_item(self, item: Dict[str, Any], item_id: str, output_dir: str = ".") -> Dict[str, Any]:
        """
        Process a single data item with improved error handling.
        
        Args:
            item: Data item containing messages
            item_id: ID for the item
            output_dir: Output directory for debug files
            
        Returns:
            Processed result dictionary
        """
        try:
            # Validate item structure
            if not isinstance(item, dict):
                return {
                    "id": item_id,
                    "topic": "Unknown Topic",
                    "outline": [],
                    "schema_version": "outline.v1",
                    "format_ok": False,
                    "error": f"Invalid item type: {type(item)}, expected dict"
                }
            
            # Generate ID if not present
            if "id" not in item:
                item["id"] = item_id
            
            # Filter messages with better error handling
            try:
                messages = self.filter_messages(item.get("messages", []))
            except Exception as e:
                return {
                    "id": item_id,
                    "topic": "Unknown Topic",
                    "outline": [],
                    "schema_version": "outline.v1",
                    "format_ok": False,
                    "error": f"Failed to filter messages: {str(e)}"
                }
            
            if not messages:
                return {
                    "id": item_id,
                    "topic": "Unknown Topic",
                    "outline": [],
                    "schema_version": "outline.v1",
                    "format_ok": False,
                    "error": "No valid messages found"
                }
            
            # Call API
            response_content = self.call_api(messages)
            
            if response_content is None:
                return {
                    "id": item_id,
                    "topic": "Unknown Topic",
                    "outline": [],
                    "schema_version": "outline.v1",
                    "format_ok": False,
                    "error": "API call failed"
                }
            
            # Normalize response
            try:
                normalized = self.normalize_outline(response_content, item_id, output_dir)
                normalized["id"] = item_id
                return normalized
            except Exception as e:
                return {
                    "id": item_id,
                    "topic": "Unknown Topic",
                    "outline": [],
                    "schema_version": "outline.v1",
                    "format_ok": False,
                    "error": f"Failed to normalize response: {str(e)}"
                }
            
        except Exception as e:
            logger.error(f"Error processing item {item_id}: {e}")
            return {
                "id": item_id,
                "topic": "Unknown Topic",
                "outline": [],
                "schema_version": "outline.v1",
                "format_ok": False,
                "error": str(e)
            }
    
    def generate_outlines(self, data_items: List[Dict[str, Any]], 
                         output_file: str, 
                         num_workers: int = 8) -> Dict[str, int]:
        """
        Generate outlines for all data items using parallel processing.
        
        Args:
            data_items: List of data items to process
            output_file: Output file path
            num_workers: Number of worker threads
            
        Returns:
            Dictionary with success and failure counts
        """
        success_count = 0
        fail_count = 0
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Process items in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self.process_item, item, str(i), os.path.dirname(output_file)): i 
                for i, item in enumerate(data_items)
            }
            
            # Process results with progress bar
            with tqdm(total=len(data_items), desc="Generating outlines") as pbar:
                for future in as_completed(future_to_item):
                    result = future.result()
                    
                    # Write result to file (thread-safe)
                    with lock:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                    # Update counters
                    if result.get("format_ok", False):
                        success_count += 1
                    else:
                        fail_count += 1
                    
                    pbar.update(1)
        
        return {"success": success_count, "failed": fail_count}


def main():
    """Main function to handle CLI arguments and execute outline generation."""
    parser = argparse.ArgumentParser(description="Generate outlines from prompts using OpenAI-compatible APIs")
    
    # Required arguments
    parser.add_argument("--api_url", required=True, help="OpenAI-compatible API URL")
    parser.add_argument("--api_key", required=True, help="API key for authentication")
    parser.add_argument("--model", required=True, help="Model name to use for generation")
    parser.add_argument("--save_dir", required=True, help="Output directory (will be created if not exists)")
    
    # Optional arguments
    parser.add_argument("--dataset_path", default="datasets/test_prompts.json", 
                       help="Path to input dataset (default: datasets/test_prompts.json)")
    parser.add_argument("--num_workers", type=int, default=8, 
                       help="Number of concurrent worker threads (default: 8)")
    parser.add_argument("--timeout", type=int, default=3600, 
                       help="API request timeout in seconds (default: 3600)")
    parser.add_argument("--log_file", help="Log file path for unified logging")
    
    args = parser.parse_args()
    
    # 设置日志配置
    global logger
    logger = setup_logging(args.log_file)
    
    # Validate inputs
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset file not found: {args.dataset_path}")
        return 1
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    output_file = os.path.join(args.save_dir, "generation.normalized.jsonl")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    try:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to fix common JSON issues
        content = content.strip()
        if not content.endswith(']'):
            content += ']'
        
        # Try to parse as JSON array first
        try:
            data_items = json.loads(content)
            if not isinstance(data_items, list):
                logger.error("Dataset must be a JSON array")
                return 1
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse as JSON array: {e}")
            # If that fails, try to extract valid JSON objects
            logger.info("Trying to extract valid JSON objects...")
            data_items = []
            lines = content.split('\n')
            current_obj = ""
            brace_count = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Count braces to find complete JSON objects
                brace_count += line.count('{') - line.count('}')
                current_obj += line
                
                if brace_count == 0 and current_obj.strip():
                    try:
                        item = json.loads(current_obj.strip())
                        data_items.append(item)
                        current_obj = ""
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse object at line {line_num}")
                        current_obj = ""
        
        if not data_items:
            logger.error("No valid data items found in dataset")
            return 1
            
        logger.info(f"Loaded {len(data_items)} items from dataset")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
    
    # Initialize generator
    generator = OutlineGenerator(
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout
    )
    
    # Generate outlines
    logger.info(f"Starting outline generation with {args.num_workers} workers")
    logger.info(f"Output will be saved to: {output_file}")
    
    # Clear output file
    with open(output_file, 'w', encoding='utf-8') as f:
        pass  # Create empty file
    
    # Process all items
    start_time = time.time()
    stats = generator.generate_outlines(data_items, output_file, args.num_workers)
    end_time = time.time()
    
    # Print summary
    total_time = end_time - start_time
    logger.info(f"\nGeneration completed in {total_time:.2f} seconds")
    logger.info(f"Total items processed: {len(data_items)}")
    logger.info(f"Successful generations: {stats['success']}")
    logger.info(f"Failed generations: {stats['failed']}")
    logger.info(f"Success rate: {stats['success']/len(data_items)*100:.1f}%")
    logger.info(f"Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
