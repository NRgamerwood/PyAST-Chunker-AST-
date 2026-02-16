"""
Unit tests for the vector store module.
"""

import pytest
import shutil
import os
from src.vector_store import CodeBaseStore
from src.parser import CodeChunk, ChunkMetadata


def test_vector_store_add_and_search():
    """
    Test adding chunks to the vector store and searching for them.
    Verifies metadata serialization and restoration (especially lists and tuples).
    """
    # Setup: Use a temporary directory for the test database
    test_db_dir = "./test_chroma_db"
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)
    
    try:
        store = CodeBaseStore(collection_name="test_collection", persist_directory=test_db_dir)
        
        # 1. Create dummy chunks
        chunk1 = CodeChunk(
            content="def add(a, b):\n    return a + b",
            metadata=ChunkMetadata(
                file_path="math_utils.py",
                node_type="function",
                name="add",
                line_range=(1, 2),
                parent_name=None,
                dependencies=["int", "float"]
            )
        )
        
        chunk2 = CodeChunk(
            content="class Calculator:\n    def subtract(self, a, b):\n        return a - b",
            metadata=ChunkMetadata(
                file_path="calc.py",
                node_type="function",
                name="subtract",
                line_range=(2, 3),
                parent_name="Calculator",
                dependencies=["int"]
            )
        )
        
        # 2. Add chunks
        store.add_chunks([chunk1, chunk2])
        
        # 3. Search
        # Searching for "subtract" should return the second chunk
        results = store.search("Calculator subtract", n_results=1)
        
        # 4. Assertions
        assert len(results) == 1
        found_chunk = results[0]
        assert found_chunk.metadata.name == "subtract"
        assert found_chunk.metadata.parent_name == "Calculator"
        
        # Check if dependencies list was correctly restored
        assert isinstance(found_chunk.metadata.dependencies, list)
        assert "int" in found_chunk.metadata.dependencies
        
        # Check if line_range tuple was correctly restored
        assert found_chunk.metadata.line_range == (2, 3)
        assert isinstance(found_chunk.metadata.line_range, tuple)
        
        # Search for "add"
        results_add = store.search("add a b", n_results=1)
        assert results_add[0].metadata.name == "add"
        assert "float" in results_add[0].metadata.dependencies
        assert results_add[0].metadata.parent_name is None
        
    finally:
        # Cleanup
        if os.path.exists(test_db_dir):
            shutil.rmtree(test_db_dir)
