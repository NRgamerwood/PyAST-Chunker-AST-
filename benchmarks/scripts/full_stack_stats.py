"""
Full-Stack Benchmark: Comparing multiple chunking strategies on complex Python files.
Strategies: Simple Character, LangChain Python-Aware, LlamaIndex Tree-sitter, and PyAST-RAG.
"""

import os
import ast
import json
import time
import textwrap
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from llama_index.core.node_parser import CodeSplitter
from llama_index.core import Document

from src.parser import ASTParser
from src.utils import read_file

# 1. 实验配置
TEST_FILES = [
    "tests/data/requests/src/requests/sessions.py",
    "tests/data/requests/src/requests/models.py",
    "tests/data/requests/src/requests/adapters.py",
    "tests/data/requests/src/requests/utils.py",
    "tests/data/requests/src/requests/cookies.py"
]
CHUNK_SIZE = 800
RESULTS_DIR = "benchmarks/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_chunks(chunks: List[str], method_name: str, meta_per_chunk: float = 1.0) -> Dict[str, Any]:
    """Calculates metrics for a list of string chunks."""
    total_chunks = len(chunks)
    syntax_errors = 0
    fragmented_functions = 0
    total_def_chunks = 0

    for chunk in chunks:
        # A. Syntax Validity
        try:
            ast.parse(textwrap.dedent(chunk))
        except SyntaxError:
            syntax_errors += 1

        # B. Function Integrity
        if "def " in chunk:
            total_def_chunks += 1
            try:
                tree = ast.parse(textwrap.dedent(chunk))
                # If it contains 'def' but no FunctionDef node is found in the top-level 
                # (or it's partial), it's fragmented.
                has_func = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
                if not has_func:
                    fragmented_functions += 1
            except SyntaxError:
                fragmented_functions += 1

    return {
        "method": method_name,
        "total_chunks": total_chunks,
        "syntax_validity": (1 - syntax_errors / total_chunks) * 100 if total_chunks > 0 else 0,
        "function_integrity": (1 - fragmented_functions / total_def_chunks) * 100 if total_def_chunks > 0 else 100,
        "metadata_density": meta_per_chunk
    }


def run_benchmark():
    """Main benchmark execution logic."""
    print(f"🚀 Starting Full-Stack Benchmark on {len(TEST_FILES)} complex files...")
    
    # Initialize Parsers
    lc_simple = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    lc_pro = RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=0)
    li_splitter = CodeSplitter(language="python", chunk_lines=40, max_chars=CHUNK_SIZE)
    pyast_parser = ASTParser()

    all_results = []
    
    # Data containers for samples
    samples = {
        "baseline_simple": [],
        "baseline_pro": [],
        "llama_index": [],
        "pyast_rag": []
    }

    # Timing and Processing
    for name, splitter_fn in [
        ("Baseline 1 (Simple)", lambda c: lc_simple.split_text(c)),
        ("Baseline 2 (LangChain Pro)", lambda c: lc_pro.split_text(c)),
        ("Baseline 3 (LlamaIndex)", lambda c: [n.text for n in li_splitter.get_nodes_from_documents([Document(text=c)])]),
        ("PyAST-RAG (Ours)", lambda c, p: pyast_parser.parse_source(c, p))
    ]:
        print(f"  Processing method: {name}...")
        start_time = time.perf_counter()
        
        method_chunks = []
        meta_counts = []
        
        for file_path in TEST_FILES:
            content = read_file(file_path)
            if name == "PyAST-RAG (Ours)":
                chunks_obj = pyast_parser.parse_source(content, file_path)
                method_chunks.extend([c.content for c in chunks_obj])
                meta_counts.extend([len([v for v in c.metadata.model_dump().values() if v is not None]) for c in chunks_obj])
            else:
                chunks = splitter_fn(content)
                method_chunks.extend(chunks)
                meta_counts.extend([1.0] * len(chunks)) # Standard splitters have 1 (text)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Save samples
        sample_key = name.lower().replace(" ", "_").split("(")[0].strip()
        with open(os.path.join(RESULTS_DIR, f"samples_{sample_key}.json"), "w", encoding="utf-8") as f:
            json.dump(method_chunks[:10], f, indent=2, ensure_ascii=False)

        # Evaluate
        avg_meta = sum(meta_counts) / len(meta_counts) if meta_counts else 0
        eval_res = evaluate_chunks(method_chunks, name, avg_meta)
        eval_res["speed_ms"] = (duration / len(TEST_FILES)) * 1000
        all_results.append(eval_res)

    # 2. 输出报告表格
    print("\n" + "="*95)
    header = f"{'METHOD':<30} | {'SYNTAX %':<10} | {'FUNC INTG %':<12} | {'META DENSITY':<12} | {'SPEED/FILE (ms)':<15}"
    print(header)
    print("-" * 95)
    for r in all_results:
        print(f"{r['method']:<30} | {r['syntax_validity']:>8.1f}% | {r['function_integrity']:>10.1f}% | {r['metadata_density']:>12.1f} | {r['speed_ms']:>15.2f}")
    print("="*95)

    # 3. 生成对比图
    generate_chart(all_results)


def generate_chart(results):
    """Generates the final comparison bar chart."""
    methods = [r['method'] for r in results]
    syntax = [r['syntax_validity'] for r in results]
    integrity = [r['function_integrity'] for r in results]
    
    x = range(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar([i - width/2 for i in x], syntax, width, label='Syntax Validity %', color='#66b3ff')
    ax.bar([i + width/2 for i in x], integrity, width, label='Function Integrity %', color='#ff9999')

    ax.set_ylabel('Score %')
    ax.set_title('Full-Stack Benchmark: PyAST-RAG vs Industrial Solutions')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()
    ax.set_ylim(0, 110)

    output_path = os.path.join(RESULTS_DIR, "final_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n✅ Benchmark result chart saved to: {output_path}")


if __name__ == "__main__":
    run_benchmark()
