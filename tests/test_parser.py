"""
Unit tests for the AST parser module.
"""

import pytest
from src.parser import ASTParser


def test_parse_simple_function():
    """Test parsing a simple top-level function."""
    source = "def hello():\n    print('world')"
    parser = ASTParser()
    chunks = parser.parse_source(source, "test.py")

    assert len(chunks) == 1
    assert chunks[0].metadata.name == "hello"
    assert chunks[0].metadata.node_type == "function"
    assert chunks[0].metadata.parent_name is None
    assert "print" in chunks[0].metadata.dependencies


def test_parse_class_with_methods():
    """Test parsing a class with methods and verify parent_name and dependencies."""
    source = """
class MyClass:
    def method_one(self):
        pass
    
    def method_two(self):
        self.method_one()
"""
    parser = ASTParser()
    chunks = parser.parse_source(source, "test.py")

    # Should have 3 chunks: 1 class, 2 functions
    assert len(chunks) == 3

    # Verify Class chunk
    class_chunk = next(c for c in chunks if c.metadata.node_type == "class")
    assert class_chunk.metadata.name == "MyClass"
    assert class_chunk.metadata.parent_name is None

    # Verify Method chunks
    method_chunks = [c for c in chunks if c.metadata.node_type == "function"]
    assert len(method_chunks) == 2
    for chunk in method_chunks:
        assert chunk.metadata.parent_name == "MyClass"

    # Verify dependency tracking in method_two
    m2 = next(c for c in method_chunks if c.metadata.name == "method_two")
    assert "method_one" in m2.metadata.dependencies


def test_async_function():
    """Test parsing an async function."""
    source = "async def fetch_data():\n    await asyncio.sleep(1)"
    parser = ASTParser()
    chunks = parser.parse_source(source, "test.py")

    assert len(chunks) == 1
    assert chunks[0].metadata.name == "fetch_data"
    assert chunks[0].metadata.node_type == "function"
    assert "sleep" in chunks[0].metadata.dependencies


def test_nested_classes_and_functions():
    """Test nested structure: class inside function or function inside function."""
    source = """
def outer_func():
    class InnerClass:
        def inner_method(self):
            pass
    def inner_func():
        pass
"""
    parser = ASTParser()
    chunks = parser.parse_source(source, "test.py")

    # outer_func (function)
    # InnerClass (class, parent=None in current implementation because we don't push functions to parent_stack)
    # inner_method (function, parent=InnerClass)
    # inner_func (function, parent=None)
    
    # Let's check names
    names = [c.metadata.name for c in chunks]
    assert "outer_func" in names
    assert "InnerClass" in names
    assert "inner_method" in names
    assert "inner_func" in names

    inner_method = next(c for c in chunks if c.metadata.name == "inner_method")
    assert inner_method.metadata.parent_name == "InnerClass"
