"""
Integration test for Python-AST-RAG using the 'requests' library.
"""

import os
import subprocess
import time
import random
import pytest
from src.parser import ASTParser
from src.utils import get_all_python_files, read_file


def clone_requests():
    """Clones the requests library for testing if not already present."""
    target_dir = os.path.join("tests", "data", "requests")
    if not os.path.exists(target_dir):
        print(f"\nCloning requests library to {target_dir}...")
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/psf/requests.git", target_dir],
            check=True,
            capture_output=True
        )
    return target_dir


def test_requests_integration():
    """
    Stress test: Parse the entire requests library and collect metrics.
    """
    repo_path = clone_requests()
    python_files = get_all_python_files(repo_path)
    
    parser = ASTParser()
    
    total_files = len(python_files)
    success_files = 0
    failed_files = 0
    all_chunks = []
    
    start_time = time.time()
    
    for file_path in python_files:
        try:
            content = read_file(file_path)
            chunks = parser.parse_source(content, file_path)
            all_chunks.extend(chunks)
            success_files += 1
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            failed_files += 1
            
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 统计指标打印
    print("\n" + "="*50)
    print("BATTLE TESTING REPORT: REQUESTS LIBRARY")
    print("="*50)
    print(f"Total Files Found:    {total_files}")
    print(f"Successfully Parsed:  {success_files}")
    print(f"Failed Files:         {failed_files}")
    print(f"Total Chunks Created: {len(all_chunks)}")
    print(f"Total Time Taken:     {total_duration:.2f} seconds")
    print("="*50)
    
    # Assertions
    assert failed_files == 0, f"Expected 0 failures, but found {failed_files}"
    assert len(all_chunks) > 0, "No chunks were generated from the requests library"
    
    # 抽样打印 (Random Sampling)
    if all_chunks:
        print("\nRANDOM SAMPLES FOR MANUAL VERIFICATION:")
        samples = random.sample(all_chunks, min(3, len(all_chunks)))
        for i, chunk in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"  File:        {os.path.relpath(chunk.metadata.file_path, repo_path)}")
            print(f"  Name:        {chunk.metadata.name}")
            print(f"  Type:        {chunk.metadata.node_type}")
            print(f"  Parent:      {chunk.metadata.parent_name}")
            print(f"  Deps:        {chunk.metadata.dependencies}")
            # print(f"  Content Line Count: {len(chunk.content.splitlines())}")
    print("="*50 + "\n")
