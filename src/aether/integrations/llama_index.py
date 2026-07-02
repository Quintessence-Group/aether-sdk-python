import time
from typing import Any, List

try:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.schema import BaseNode, TextNode
except ImportError:
    raise ImportError(
        "Could not import llama_index_core. Please install it with `pip install llama-index-core` "
        "to use the Aether LlamaIndex integration."
    )

from aether.client import AetherClient
from aether.models import BatchInsertItem

class AetherVectorStore(BasePydanticVectorStore):
    """Aether VectorStore integration for LlamaIndex."""

    _client: Any = PrivateAttr()

    def __init__(self, client: AetherClient, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = client

    @property
    def client(self) -> Any:
        return self._client

    @property
    def flat_metadata(self) -> bool:
        return True

    def stores_text(self) -> bool:
        return True

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to index."""
        if not nodes:
            return []
            
        items = []
        for node in nodes:
            metadata = node.metadata or {}
            tags = metadata.get("tags")
            # If tags is a string, wrap it
            if isinstance(tags, str):
                tags = [tags]
                
            content = node.get_content(metadata_mode="all")
            filename = f"llamaindex_{int(time.time()*1000)}_{node.node_id}.txt"
            
            items.append(BatchInsertItem(filename=filename, content=content, tags=tags))
            
        chunk_size = add_kwargs.get("chunk_size")
        overlap = add_kwargs.get("overlap")
        
        # Execute batch upload
        results = self._client.batch_insert(items, chunk_size=chunk_size, overlap=overlap)
        
        # Aether mints its own doc_ids (which represent vectors/chunks)
        # We will return the aether sequence of doc IDs
        return [r.doc_id for r in results]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete from index."""
        self._client.delete(ref_doc_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        k = query.similarity_top_k
        tags = []
        
        if query.filters:
            # simple filter parsing (best effort mapped to tags)
            for filter in query.filters.legacy_filters():
                if filter.key == "tags":
                    tags.append(filter.value)
        
        tags_kw = kwargs.get("tags") or []
        tags.extend(tags_kw)
        
        tags_param = tags if tags else None
        
        if query.query_embedding:
            results = self._client.search_by_vector(
                embedding=query.query_embedding, 
                k=k, 
                include_content=True,
                tags=tags_param
            )
        else:
            results = self._client.search(
                query.query_str or "", 
                k=k, 
                include_content=True,
                tags=tags_param
            )

        nodes = []
        similarities = []
        ids = []
        
        for r in results:
            content = r.content or r.passage or ""
            # Reconstruct node object that LlamaIndex expects
            node = TextNode(
                id_=r.doc_id,
                text=content,
                metadata={"title": r.title, "content_type": r.content_type, "score": r.score}
            )
            nodes.append(node)
            similarities.append(r.score / 100.0)  # score is 0-100; normalize to 0-1
            ids.append(r.doc_id)
            
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
