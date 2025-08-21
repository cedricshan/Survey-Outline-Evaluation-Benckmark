#!/usr/bin/env python3
"""
Test script for outline generation

This script demonstrates how to use the genrate_outlines.py script
with the provided API information.
"""

import subprocess
import sys
import os

def test_outline_generation():
    """Test the outline generation script with sample parameters."""
    
    # API configuration from the user's specifications
    api_url = "https://ark.cn-beijing.volces.com/api/v3"
    api_key = "your-api-key-here"  # Replace with your actual API key
    model = "ep-20250530104326-cc6vk"
    
    # Output directory
    save_dir = "outputs/test_run"
    
    # Dataset path
    dataset_path = "datasets/test_prompts.json"
    
    # Number of workers (reduced for testing)
    num_workers = 4
    
    # Build the command
    cmd = [
        "python3", "scripts/genrate_outlines.py",
        "--api_url", api_url,
        "--api_key", api_key,
        "--model", model,
        "--save_dir", save_dir,
        "--dataset_path", dataset_path,
        "--num_workers", str(num_workers),
        "--timeout", "120"
    ]
    
    print("Testing outline generation script...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
        # Check if output file was created
        output_file = os.path.join(save_dir, "generation.normalized.jsonl")
        if os.path.exists(output_file):
            print(f"\nOutput file created: {output_file}")
            
            # Show first few lines of output
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"Generated {len(lines)} outline entries")
                if lines:
                    print("First entry:")
                    print(lines[0][:200] + "..." if len(lines[0]) > 200 else lines[0])
        else:
            print(f"\nOutput file not found: {output_file}")
            
    except subprocess.TimeoutExpired:
        print("Script timed out after 5 minutes")
    except Exception as e:
        print(f"Error running script: {e}")

if __name__ == "__main__":
    test_outline_generation()
