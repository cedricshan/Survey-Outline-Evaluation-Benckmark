#!/usr/bin/env python3
"""
Example usage of the outline generator

This script demonstrates how to use the OutlineGenerator class directly
in Python code, bypassing the command-line interface.
"""

import json
import os
from scripts.genrate_outlines import OutlineGenerator

def create_sample_dataset():
    """Create a small sample dataset for testing."""
    sample_data = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a scientific writing assistant. Produce a structured outline based on the article metadata and references provided."
                },
                {
                    "role": "user",
                    "content": "Create an outline for a survey paper on machine learning in healthcare."
                },
                {
                    "role": "assistant",
                    "content": "This is a human-generated outline for evaluation purposes."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a scientific writing assistant. Produce a structured outline based on the article metadata and references provided."
                },
                {
                    "role": "user",
                    "content": "Create an outline for a review paper on renewable energy technologies."
                }
            ]
        }
    ]
    
    # Save sample dataset
    os.makedirs("datasets", exist_ok=True)
    with open("datasets/sample_prompts.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("Created sample dataset: datasets/sample_prompts.json")
    return sample_data

def example_usage():
    """Example of using the OutlineGenerator class directly."""
    
    # Create sample dataset
    sample_data = create_sample_dataset()
    
    # Initialize generator with your API credentials
    generator = OutlineGenerator(
        api_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key="30a70266-37d5-4210-b8a2-34d5fb629230",
        model="ep-20250530104326-cc6vk",
        timeout=120
    )
    
    # Process a single item
    print("\nProcessing single item...")
    result = generator.process_item(sample_data[0], "0")
    print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # Process all items
    print("\nProcessing all items...")
    output_file = "outputs/example_run/generation.normalized.jsonl"
    stats = generator.generate_outlines(sample_data, output_file, num_workers=2)
    
    print(f"\nProcessing complete!")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output saved to: {output_file}")
    
    # Show output
    if os.path.exists(output_file):
        print("\nGenerated output:")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                print(f"ID: {data['id']}")
                print(f"Topic: {data['topic']}")
                print(f"Format OK: {data['format_ok']}")
                if data.get('error'):
                    print(f"Error: {data['error']}")
                print(f"Outline items: {len(data['outline'])}")
                print("---")

if __name__ == "__main__":
    example_usage()
