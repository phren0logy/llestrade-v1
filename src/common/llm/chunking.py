"""
Markdown-aware document chunking strategies using langchain-text-splitters.
"""

import logging
from typing import List, Optional, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Configure logging
logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """Provides various strategies for chunking documents."""
    
    @staticmethod
    def markdown_headers(
        text: str,
        max_tokens: int,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        strip_headers: bool = True,
        overlap: int = 200,
        chars_per_token: int = 4,
    ) -> List[str]:
        """
        Split markdown text by headers, respecting token limits.
        
        Args:
            text: The markdown text to split
            max_tokens: Maximum tokens per chunk
            headers_to_split_on: List of (header_level, header_name) tuples
            strip_headers: Whether to strip headers from chunk content
            overlap: Number of characters to overlap between chunks
            chars_per_token: Character heuristic used to derive the initial hard cap
            
        Returns:
            List of text chunks
        """
        # Default headers to split on if not provided
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        
        # Create the markdown splitter
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=strip_headers
        )
        
        # Split the text by headers
        header_splits = splitter.split_text(text)
        
        # Convert max_tokens to approximate character limit
        max_chars = max(max_tokens * max(chars_per_token, 1), 1)
        
        # Process the splits to respect token limits
        chunks = []
        current_chunk = ""
        current_metadata = {}
        
        for doc in header_splits:
            # Extract content and metadata
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            # Build header context from metadata
            header_context = ""
            for level, name in headers_to_split_on:
                if name in metadata:
                    header_context += f"{level} {metadata[name]}\n\n"
            
            # Full content with headers
            full_content = header_context + content
            
            # Check if adding this would exceed the limit
            if current_chunk and len(current_chunk) + len(full_content) + 2 > max_chars:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    # Take last 'overlap' characters from previous chunk
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + "\n\n" + full_content
                else:
                    current_chunk = full_content
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + full_content
                else:
                    current_chunk = full_content
            
            current_metadata = metadata
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If no chunks were created, return the original text as a single chunk
        if not chunks:
            chunks = [text]
        
        logger.info(f"Split document into {len(chunks)} chunks using markdown headers")
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i+1}: {len(chunk)} characters")
        
        return chunks
    
    @staticmethod
    def simple_overlap(
        text: str,
        max_tokens: int,
        overlap: int = 200,
        chars_per_token: int = 4,
    ) -> List[str]:
        """
        Simple overlapping chunk strategy (character-based).
        
        Args:
            text: The text to chunk
            max_tokens: Maximum tokens per chunk
            overlap: Number of characters to overlap
            
        Returns:
            List of text chunks
        """
        # Convert tokens to characters (rough estimate)
        max_chars = max(max_tokens * max(chars_per_token, 1), 1)
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + max_chars, len(text))
            
            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(end - 200, start)
                sentence_breaks = []
                
                # Find sentence endings
                for i in range(search_start, end):
                    if text[i] in '.!?':
                        # Check if it's followed by whitespace
                        if (i + 1 < len(text) and 
                            text[i + 1].isspace()):
                            sentence_breaks.append(i + 1)
                
                # Use the last sentence break if available
                if sentence_breaks:
                    end = sentence_breaks[-1]
                
                # If no sentence breaks, try paragraph breaks
                elif '\n\n' in text[search_start:end]:
                    para_pos = text.rfind('\n\n', search_start, end)
                    if para_pos > start:
                        end = para_pos + 2
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        
        logger.info(f"Split document into {len(chunks)} chunks using simple overlap")
        return chunks if chunks else [text]
