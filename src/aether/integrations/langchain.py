import time
from typing import Any, Iterable, List, Optional, Type

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError:
    raise ImportError(
        "Could not import langchain_core. Please install it with `pip install langchain-core` "
        "to use the Aether LangChain integration."
    )

from aether.client import AetherClient
from aether.models import BatchInsertItem


class AetherVectorStore(VectorStore):
    """Aether VectorStore integration for LangChain.
    
    Aether is unique because it handles text parsing and embeddings natively. 
    If you don't provide an embedding model, Aether will generate the embeddings 
    on the server automatically using its optimized internal models.
    """

    def __init__(
        self,
        client: AetherClient,
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ):
        """Initialize the Aether vector store.
        
        Args:
            client: An authenticated AetherClient instance.
            embedding: Optional LangChain embeddings model. If not provided, 
                       Aether will compute embeddings natively server-side.
        """
        self.client = client
        self.embedding = embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run texts through the embeddings and add to the vector store."""
        texts = list(texts)
        if not texts:
            return []
            
        metadatas = metadatas or [{} for _ in texts]
        
        # We can extract a 'tags' list from the metadata dictionary
        doc_ids = []
        if self.embedding is not None:
            # Generate embeddings locally using standard Langchain setup
            embeddings = self.embedding.embed_documents(texts)
            for text, meta, emb in zip(texts, metadatas, embeddings):
                tags = meta.get("tags")
                res = self.client.insert_with_embeddings(
                    content=text,
                    embedding=emb,
                    tags=tags,
                )
                doc_ids.append(res.doc_id)
        else:
            # Let Aether do the embeddings natively using Batch APIs for max throughput
            items = []
            for i, (text, meta) in enumerate(zip(texts, metadatas)):
                tags = meta.get("tags")
                # Aether requires a filename, so we generate a unique one
                filename = f"langchain_{int(time.time()*1000)}_{i}.txt"
                items.append(BatchInsertItem(filename=filename, content=text, tags=tags))
            
            chunk_size = kwargs.get("chunk_size")
            overlap = kwargs.get("overlap")
            results = self.client.batch_insert(items, chunk_size=chunk_size, overlap=overlap)
            doc_ids.extend([r.doc_id for r in results])
            
        return doc_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        tags = kwargs.get("tags")
        
        if self.embedding is not None:
            emb = self.embedding.embed_query(query)
            results = self.client.search_by_vector(emb, k=k, include_content=True, tags=tags)
        else:
            results = self.client.search(query, k=k, include_content=True, tags=tags)
            
        docs = []
        for r in results:
            content = r.content or r.passage or ""
            metadata = {
                "doc_id": r.doc_id, 
                "score": r.score,
                "title": r.title
            }
            if r.content_type:
                metadata["content_type"] = r.content_type
            docs.append(Document(page_content=content, metadata=metadata))
            
        return docs

    @classmethod
    def from_texts(
        cls: Type["AetherVectorStore"],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        client: Optional[AetherClient] = None,
        **kwargs: Any,
    ) -> "AetherVectorStore":
        """Return VectorStore initialized from texts."""
        if client is None:
            raise ValueError("An aether.AetherClient instance must be provided via the 'client' arg.")
        
        store = cls(client=client, embedding=embedding)
        store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store
