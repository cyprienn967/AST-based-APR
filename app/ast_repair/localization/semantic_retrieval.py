"""
semantic_retrieval.py

Embedding-based semantic retrieval for bug localization.

This module provides a signal that captures semantic similarity between
the issue description and code units (functions/methods), independent of
execution traces (SBFL) or syntactic structure (AST).

Architecture:
    1. Code Embedding: Each function/method is embedded using a code-focused
       encoder (UniXcoder, CodeBERT, or similar). The embedding captures:
       - Function signature (name, parameters, return type hints)
       - Docstring (if present)
       - First N lines of body (semantic intent)

    2. Issue Embedding: The issue text is embedded once at localization time.

    3. Similarity: Cosine similarity between issue embedding and each code
       embedding produces a relevance score.

Key Design Choices:
    - Uses sentence-transformers for efficient batched encoding
    - Embeddings are L2-normalized at creation time for fast dot-product similarity
    - Function-level granularity (not file-level) for precise localization
    - Caching strategy: embeddings can be precomputed and stored per-project
    - Graceful degradation: falls back to empty scores if model unavailable

Usage in localization pipeline:
    semantic_scores = compute_semantic_scores(md, source_code, issue_text)
    # Returns: Dict[int, float] mapping node_id -> similarity score
"""

from __future__ import annotations

import ast
import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from loguru import logger

from app.ast_repair.metadata import ASTMetadata


# ============================================================================
# Configuration
# ============================================================================

# Model selection - UniXcoder is optimal for code-NL matching
# Fallback chain: unixcoder -> codebert -> minilm (general purpose)
DEFAULT_MODEL_NAME = "microsoft/unixcoder-base"
FALLBACK_MODELS = [
    "microsoft/codebert-base",
    "sentence-transformers/all-MiniLM-L6-v2",  # Fast fallback
]

# Embedding dimensions (768 for most transformer models)
EMBEDDING_DIM = 768

# Maximum tokens for code representation
MAX_CODE_TOKENS = 256
MAX_ISSUE_TOKENS = 512

# How many lines of function body to include (beyond signature + docstring)
MAX_BODY_LINES = 15

# Cache directory for precomputed embeddings
CACHE_DIR = Path(".ast_cache/embeddings")


# ============================================================================
# Lazy model loading (singleton pattern)
# ============================================================================

_model_instance: Optional[Any] = None
_tokenizer_instance: Optional[Any] = None
_model_load_attempted: bool = False


def _get_model_and_tokenizer() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Lazy-load the embedding model and tokenizer.
    Returns (model, tokenizer) or (None, None) if unavailable.
    """
    global _model_instance, _tokenizer_instance, _model_load_attempted
    
    if _model_load_attempted:
        return _model_instance, _tokenizer_instance
    
    _model_load_attempted = True
    
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        # Try models in order of preference
        models_to_try = [DEFAULT_MODEL_NAME] + FALLBACK_MODELS
        
        for model_name in models_to_try:
            try:
                logger.info(f"Loading embedding model: {model_name}")
                _tokenizer_instance = AutoTokenizer.from_pretrained(model_name)
                _model_instance = AutoModel.from_pretrained(model_name)
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    _model_instance = _model_instance.cuda()
                    logger.info("Using GPU for embeddings")
                
                # Set to eval mode
                _model_instance.eval()
                logger.info(f"Successfully loaded: {model_name}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if _model_instance is None:
            logger.warning("No embedding model available - semantic retrieval disabled")
            
    except ImportError as e:
        logger.warning(f"transformers library not available: {e}")
        logger.warning("Install with: pip install transformers torch")
    
    return _model_instance, _tokenizer_instance


# ============================================================================
# Code representation extraction
# ============================================================================

@dataclass
class FunctionInfo:
    """Extracted information about a function for embedding."""
    node_id: int
    name: str
    signature: str
    docstring: Optional[str]
    body_preview: str
    full_text: str  # Combined text for embedding
    start_line: int
    end_line: int


def extract_function_info(
    node: ast.AST,
    node_id: int,
    source_lines: List[str],
    md: ASTMetadata
) -> Optional[FunctionInfo]:
    """
    Extract embeddable text representation from a function/method node.
    
    Returns None for non-function nodes.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    
    start_line, end_line = md.line_map.get(node_id, (None, None))
    if start_line is None or end_line is None:
        return None
    
    # Extract function name
    name = node.name
    
    # Build signature string
    signature = _build_signature(node)
    
    # Extract docstring
    docstring = ast.get_docstring(node)
    
    # Extract body preview (first N lines after signature/docstring)
    body_start = start_line
    if node.body:
        first_body_node = node.body[0]
        if hasattr(first_body_node, 'lineno'):
            body_start = first_body_node.lineno
            # Skip docstring node if present
            if isinstance(first_body_node, ast.Expr) and isinstance(first_body_node.value, ast.Constant):
                if len(node.body) > 1 and hasattr(node.body[1], 'lineno'):
                    body_start = node.body[1].lineno
    
    # Get body lines (capped at MAX_BODY_LINES)
    body_end = min(body_start + MAX_BODY_LINES - 1, end_line)
    body_lines = source_lines[body_start - 1:body_end] if body_start <= len(source_lines) else []
    body_preview = '\n'.join(body_lines)
    
    # Compose full text for embedding
    parts = [signature]
    if docstring:
        parts.append(f'"""{docstring}"""')
    if body_preview.strip():
        parts.append(body_preview)
    
    full_text = '\n'.join(parts)
    
    return FunctionInfo(
        node_id=node_id,
        name=name,
        signature=signature,
        docstring=docstring,
        body_preview=body_preview,
        full_text=full_text,
        start_line=start_line,
        end_line=end_line,
    )


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a readable signature string from function AST node."""
    parts = []
    
    # Async prefix
    if isinstance(node, ast.AsyncFunctionDef):
        parts.append("async ")
    
    parts.append(f"def {node.name}(")
    
    # Arguments
    args = node.args
    arg_strs = []
    
    # Regular args
    num_defaults = len(args.defaults)
    num_args = len(args.args)
    
    for i, arg in enumerate(args.args):
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except:
                pass
        
        # Check if this arg has a default
        default_idx = i - (num_args - num_defaults)
        if default_idx >= 0 and default_idx < len(args.defaults):
            try:
                arg_str += f"={ast.unparse(args.defaults[default_idx])}"
            except:
                arg_str += "=..."
        
        arg_strs.append(arg_str)
    
    # *args
    if args.vararg:
        arg_strs.append(f"*{args.vararg.arg}")
    
    # **kwargs
    if args.kwarg:
        arg_strs.append(f"**{args.kwarg.arg}")
    
    parts.append(", ".join(arg_strs))
    parts.append(")")
    
    # Return annotation
    if node.returns:
        try:
            parts.append(f" -> {ast.unparse(node.returns)}")
        except:
            pass
    
    return "".join(parts)


def extract_all_functions(
    md: ASTMetadata,
    source_code: str
) -> List[FunctionInfo]:
    """
    Extract FunctionInfo for all functions/methods in the AST.
    """
    source_lines = source_code.split('\n')
    functions = []
    
    for node_id, node in md.node_index.items():
        info = extract_function_info(node, node_id, source_lines, md)
        if info is not None:
            functions.append(info)
    
    return functions


# ============================================================================
# Embedding computation
# ============================================================================

def compute_embedding(
    text: str,
    model: Any,
    tokenizer: Any,
    max_length: int = MAX_CODE_TOKENS
) -> Optional[np.ndarray]:
    """
    Compute a single embedding vector for the given text.
    
    Returns L2-normalized embedding or None on failure.
    """
    try:
        import torch
        
        # Tokenize with truncation
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling over sequence dimension
        # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
        attention_mask = inputs['attention_mask']
        hidden_states = outputs.last_hidden_state
        
        # Expand attention mask for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum and normalize by actual sequence length
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embedding = (sum_embeddings / sum_mask).squeeze(0)
        
        # L2 normalize for cosine similarity via dot product
        embedding = embedding / embedding.norm()
        
        return embedding.cpu().numpy()
        
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        return None


def compute_embeddings_batch(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    max_length: int = MAX_CODE_TOKENS,
    batch_size: int = 16
) -> List[Optional[np.ndarray]]:
    """
    Compute embeddings for multiple texts efficiently in batches.
    
    Returns list of L2-normalized embeddings (None for failures).
    """
    try:
        import torch
        
        all_embeddings = []
        device = next(model.parameters()).device
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # L2 normalize each embedding
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            
            # Convert to numpy
            batch_embeddings = embeddings.cpu().numpy()
            all_embeddings.extend([batch_embeddings[j] for j in range(len(batch_texts))])
        
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Batch embedding computation failed: {e}")
        return [None] * len(texts)


# ============================================================================
# Similarity computation
# ============================================================================

def compute_similarity(
    query_embedding: np.ndarray,
    code_embeddings: Dict[int, np.ndarray]
) -> Dict[int, float]:
    """
    Compute cosine similarity between query and all code embeddings.
    
    Since embeddings are L2-normalized, this is just dot product.
    
    Args:
        query_embedding: L2-normalized issue embedding
        code_embeddings: Dict mapping node_id -> L2-normalized embedding
    
    Returns:
        Dict mapping node_id -> similarity score in [0, 1]
    """
    if len(code_embeddings) == 0:
        return {}
    
    similarities = {}
    
    for node_id, code_emb in code_embeddings.items():
        # Dot product of normalized vectors = cosine similarity
        sim = float(np.dot(query_embedding, code_emb))
        # Clamp to [0, 1] (negative similarities are not useful for ranking)
        similarities[node_id] = max(0.0, sim)
    
    return similarities


def compute_similarity_batch(
    query_embedding: np.ndarray,
    node_ids: List[int],
    embeddings_matrix: np.ndarray
) -> Dict[int, float]:
    """
    Vectorized similarity computation for better performance.
    
    Args:
        query_embedding: L2-normalized query vector (d,)
        node_ids: List of node_ids corresponding to rows
        embeddings_matrix: (n, d) matrix of L2-normalized embeddings
    
    Returns:
        Dict mapping node_id -> similarity score
    """
    # Matrix-vector multiply: (n, d) @ (d,) -> (n,)
    similarities = embeddings_matrix @ query_embedding
    
    # Convert to dict with non-negative values
    return {
        nid: max(0.0, float(sim))
        for nid, sim in zip(node_ids, similarities)
    }


# ============================================================================
# Caching utilities
# ============================================================================

def _compute_source_hash(source_code: str) -> str:
    """Compute a hash of source code for cache invalidation."""
    return hashlib.md5(source_code.encode()).hexdigest()[:12]


def _get_cache_path(project_path: str, file_path: str) -> Path:
    """Get cache file path for a given source file."""
    cache_dir = CACHE_DIR / hashlib.md5(project_path.encode()).hexdigest()[:8]
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
    return cache_dir / f"{file_hash}.pkl"


def save_embeddings_cache(
    cache_path: Path,
    source_hash: str,
    embeddings: Dict[int, np.ndarray],
    function_names: Dict[int, str]
):
    """Save computed embeddings to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'source_hash': source_hash,
                'embeddings': embeddings,
                'function_names': function_names,
            }, f)
        logger.debug(f"Saved embeddings cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save embeddings cache: {e}")


def load_embeddings_cache(
    cache_path: Path,
    source_hash: str
) -> Optional[Tuple[Dict[int, np.ndarray], Dict[int, str]]]:
    """Load cached embeddings if valid."""
    try:
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        if data.get('source_hash') != source_hash:
            logger.debug("Cache invalidated: source changed")
            return None
        
        return data['embeddings'], data['function_names']
        
    except Exception as e:
        logger.warning(f"Failed to load embeddings cache: {e}")
        return None


# ============================================================================
# Main API
# ============================================================================

def compute_semantic_scores(
    md: ASTMetadata,
    source_code: str,
    issue_text: str,
    project_path: str = "",
    file_path: str = "",
    use_cache: bool = True,
) -> Dict[int, float]:
    """
    Compute semantic similarity scores between issue text and all functions.
    
    This is the main entry point for semantic retrieval in the localization pipeline.
    
    Args:
        md: AST metadata with node_index
        source_code: The source code of the file being localized
        issue_text: The bug report / issue description
        project_path: Optional project root for caching
        file_path: Optional file path for caching
        use_cache: Whether to use embedding cache
    
    Returns:
        Dict mapping node_id -> similarity score in [0, 1]
        Empty dict if model unavailable or on error.
    
    Example:
        scores = compute_semantic_scores(md, source, issue_text)
        # scores = {42: 0.85, 67: 0.72, 89: 0.45, ...}
    """
    if not issue_text or not source_code:
        return {}
    
    # Get model (lazy loading)
    model, tokenizer = _get_model_and_tokenizer()
    if model is None or tokenizer is None:
        logger.debug("Semantic retrieval skipped: no model available")
        return {}
    
    # Extract functions from AST
    functions = extract_all_functions(md, source_code)
    if not functions:
        logger.debug("No functions found for semantic analysis")
        return {}
    
    logger.info(f"Computing semantic embeddings for {len(functions)} functions")
    
    # Try to load cached code embeddings
    code_embeddings: Dict[int, np.ndarray] = {}
    function_names: Dict[int, str] = {}
    source_hash = _compute_source_hash(source_code)
    
    cache_path = None
    if use_cache and project_path and file_path:
        cache_path = _get_cache_path(project_path, file_path)
        cached = load_embeddings_cache(cache_path, source_hash)
        if cached is not None:
            code_embeddings, function_names = cached
            logger.debug(f"Loaded {len(code_embeddings)} cached embeddings")
    
    # Compute missing embeddings
    if not code_embeddings:
        # Batch compute all function embeddings
        texts = [f.full_text for f in functions]
        embeddings = compute_embeddings_batch(texts, model, tokenizer, MAX_CODE_TOKENS)
        
        for func_info, emb in zip(functions, embeddings):
            if emb is not None:
                code_embeddings[func_info.node_id] = emb
                function_names[func_info.node_id] = func_info.name
        
        # Save to cache
        if cache_path and code_embeddings:
            save_embeddings_cache(cache_path, source_hash, code_embeddings, function_names)
    
    if not code_embeddings:
        logger.warning("No code embeddings computed")
        return {}
    
    # Compute issue embedding
    issue_embedding = compute_embedding(issue_text, model, tokenizer, MAX_ISSUE_TOKENS)
    if issue_embedding is None:
        logger.warning("Failed to compute issue embedding")
        return {}
    
    # Compute similarities
    if len(code_embeddings) > 10:
        # Use vectorized computation for larger sets
        node_ids = list(code_embeddings.keys())
        embeddings_matrix = np.stack([code_embeddings[nid] for nid in node_ids])
        similarities = compute_similarity_batch(issue_embedding, node_ids, embeddings_matrix)
    else:
        similarities = compute_similarity(issue_embedding, code_embeddings)
    
    # Log top matches for debugging
    if similarities:
        top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
        for nid, score in top_matches:
            func_name = function_names.get(nid, f"node_{nid}")
            logger.debug(f"Semantic match: {func_name} = {score:.3f}")
    
    return similarities


def compute_semantic_scores_multi_file(
    files: List[Tuple[ASTMetadata, str, str]],  # (md, source_code, file_path)
    issue_text: str,
    project_path: str = "",
    use_cache: bool = True,
) -> Dict[str, Dict[int, float]]:
    """
    Compute semantic scores across multiple files efficiently.
    
    This batches the issue embedding computation and allows for
    cross-file similarity comparison.
    
    Args:
        files: List of (metadata, source_code, file_path) tuples
        issue_text: The bug report / issue description
        project_path: Project root for caching
        use_cache: Whether to use embedding cache
    
    Returns:
        Dict mapping file_path -> {node_id -> similarity score}
    """
    if not issue_text or not files:
        return {}
    
    model, tokenizer = _get_model_and_tokenizer()
    if model is None:
        return {}
    
    # Compute issue embedding once
    issue_embedding = compute_embedding(issue_text, model, tokenizer, MAX_ISSUE_TOKENS)
    if issue_embedding is None:
        return {}
    
    results = {}
    
    for md, source_code, file_path in files:
        functions = extract_all_functions(md, source_code)
        if not functions:
            continue
        
        # Check cache
        code_embeddings: Dict[int, np.ndarray] = {}
        source_hash = _compute_source_hash(source_code)
        cache_path = _get_cache_path(project_path, file_path) if use_cache else None
        
        if cache_path:
            cached = load_embeddings_cache(cache_path, source_hash)
            if cached:
                code_embeddings = cached[0]
        
        if not code_embeddings:
            texts = [f.full_text for f in functions]
            embeddings = compute_embeddings_batch(texts, model, tokenizer, MAX_CODE_TOKENS)
            
            for func_info, emb in zip(functions, embeddings):
                if emb is not None:
                    code_embeddings[func_info.node_id] = emb
            
            if cache_path and code_embeddings:
                func_names = {f.node_id: f.name for f in functions}
                save_embeddings_cache(cache_path, source_hash, code_embeddings, func_names)
        
        if code_embeddings:
            similarities = compute_similarity(issue_embedding, code_embeddings)
            results[file_path] = similarities
    
    return results


# ============================================================================
# Convenience wrapper for localize.py integration
# ============================================================================

def semantic_retrieval_scores(
    md: ASTMetadata,
    source_code: str,
    issue_text: str,
) -> Dict[int, float]:
    """
    Simplified wrapper for use in localize_fault().
    
    Returns node_id -> similarity score mapping, or empty dict on error.
    """
    return compute_semantic_scores(
        md=md,
        source_code=source_code,
        issue_text=issue_text,
        use_cache=True,
    )

