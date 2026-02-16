"""
Vector store module for persistent storage and retrieval of code chunks using ChromaDB.
"""

import chromadb
from typing import List, Optional, Dict, Any
from src.parser import CodeChunk, ChunkMetadata


class CodeBaseStore:
    """
    Encapsulates ChromaDB operations for storing and searching code chunks.
    """

    def __init__(self, collection_name: str = "code_rag", persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB client and collection.

        Args:
            collection_name (str): Name of the collection.
            persist_directory (str): Directory for data persistence.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: List[CodeChunk]):
        """
        Adds a list of code chunks to the vector store.
        Handles serialization of metadata fields not supported by ChromaDB.

        Args:
            chunks (List[CodeChunk]): List of code chunks to add.
        """
        if not chunks:
            return

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk.content)
            
            # Serialize metadata from the Pydantic model
            meta_dict = chunk.metadata.model_dump()
            
            # 1. 处理 dependencies: List[str] -> str
            if "dependencies" in meta_dict:
                meta_dict["dependencies"] = ",".join(meta_dict["dependencies"])
            
            # 2. 处理 line_range: Tuple[int, int] -> start/end line fields
            if "line_range" in meta_dict:
                line_range = meta_dict.pop("line_range")
                meta_dict["start_line"] = line_range[0]
                meta_dict["end_line"] = line_range[1]
            
            # 3. 确保没有 None 值 (ChromaDB 不支持 None)
            meta_dict = {k: v for k, v in meta_dict.items() if v is not None}
                
            metadatas.append(meta_dict)
            
            # Create a unique ID for each chunk
            chunk_id = f"{chunk.metadata.file_path}:{chunk.metadata.parent_name or ''}:{chunk.metadata.name}"
            ids.append(chunk_id)

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, n_results: int = 5) -> List[CodeChunk]:
        """
        Searches for the most similar code chunks and restores metadata.

        Args:
            query (str): The search query.
            n_results (int): Number of results to return.

        Returns:
            List[CodeChunk]: List of matched code chunks with restored metadata.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        chunks = []
        if results["documents"] and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                content = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                
                # 1. Restore dependencies: str -> List[str]
                if "dependencies" in meta and isinstance(meta["dependencies"], str):
                    if meta["dependencies"]:
                        meta["dependencies"] = meta["dependencies"].split(",")
                    else:
                        meta["dependencies"] = []
                
                # 2. Restore line_range: start/end -> Tuple[int, int]
                if "start_line" in meta and "end_line" in meta:
                    start = meta.pop("start_line")
                    end = meta.pop("end_line")
                    meta["line_range"] = (start, end)
                
                # parent_name if missing will be handled by ChunkMetadata default (None)
                chunks.append(CodeChunk(content=content, metadata=ChunkMetadata(**meta)))
        
        return chunks
