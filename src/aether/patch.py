import re

path = "client.py"
with open(path, "r") as f:
    text = f.read()

# Update imports
text = text.replace("from .models import (", "from .models import (\n    BatchInsertItem,\n    BatchSearchQuery,\n    BatchSearchResponse,")

# Add chunk_size & overlap
sig_pattern_1 = r"def insert\(\s*self,\s*file_path: str \| Path,\s*content_type: str \| None = None,\s*tags: list\[str\] \| None = None,\s*\) -> DocumentRecord:"
sig_repl_1 = """def insert(
        self,
        file_path: str | Path,
        content_type: str | None = None,
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> DocumentRecord:"""
text = re.sub(sig_pattern_1, sig_repl_1, text)

sig_pattern_2 = r"def insert_text\(\s*self,\s*text: str,\s*filename: str = \"text.txt\",\s*tags: list\[str\] \| None = None,\s*\) -> DocumentRecord:"
sig_repl_2 = """def insert_text(
        self,
        text: str,
        filename: str = "text.txt",
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> DocumentRecord:"""
text = re.sub(sig_pattern_2, sig_repl_2, text)

sig_pattern_3 = r"def insert_stream\(\s*self,\s*stream,\s*filename: str = \"upload.bin\",\s*content_type: str = \"application/octet-stream\",\s*tags: list\[str\] \| None = None,\s*\) -> DocumentRecord:"
sig_repl_3 = """def insert_stream(
        self,
        stream,
        filename: str = "upload.bin",
        content_type: str = "application/octet-stream",
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> DocumentRecord:"""
text = re.sub(sig_pattern_3, sig_repl_3, text)

sig_pattern_4 = r"def update\(\s*self,\s*doc_id: str,\s*file_path: str \| Path,\s*content_type: str = \"text/plain\",\s*tags: list\[str\] \| None = None,\s*\) -> DocumentRecord:"
sig_repl_4 = """def update(
        self,
        doc_id: str,
        file_path: str | Path,
        content_type: str = "text/plain",
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> DocumentRecord:"""
text = re.sub(sig_pattern_4, sig_repl_4, text)

sig_pattern_5 = r"def insert_async\(\s*self,\s*file_path: str \| Path,\s*content_type: str \| None = None,\s*tags: list\[str\] \| None = None,\s*\) -> dict:"
sig_repl_5 = """def insert_async(
        self,
        file_path: str | Path,
        content_type: str | None = None,
        tags: list[str] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> dict:"""
text = re.sub(sig_pattern_5, sig_repl_5, text)

# Add URL logic. All the methods use `url += f"&tags={quote(','.join(tags))}"`
tag_pattern = r'        if tags:\s*url \+= f"&tags=\{quote\('"','"'\.join\(tags\)\)\}"'
tag_repl = """        if tags:
            url += f"&tags={quote(','.join(tags))}"
        if chunk_size is not None:
            url += f"&chunk_size={chunk_size}"
        if overlap is not None:
            url += f"&overlap={overlap}\""""
text = re.sub(tag_pattern, tag_repl, text)

# Append batches
if "def batch_insert" not in text:
    text += """
    # ── Batch APIs ────────────────────────────────────────────────────

    def batch_insert(
        self,
        documents: list[BatchInsertItem],
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[DocumentRecord]:
        \"\"\"Insert multiple text documents in a single batch request.\"\"\"
        payload = {
            "documents": [{"filename": d.filename, "content": d.content, "tags": d.tags} for d in documents]
        }
        if chunk_size is not None:
            payload["chunk_size"] = chunk_size
        if overlap is not None:
            payload["overlap"] = overlap
            
        resp = self._request_with_retry("POST", "/documents/batch", json=payload)
        self._raise_for_status(resp)
        results = resp.json().get("results", [])
        return [
            DocumentRecord(
                doc_id=r["doc_id"], cid=r["cid"], chunks=r["chunks"],
                vectors=r["vectors"], version=r["version"],
            ) for r in results
        ]

    def batch_search(
        self,
        queries: list[BatchSearchQuery],
    ) -> list[BatchSearchResponse]:
        \"\"\"Run multiple search queries in a single batch request.\"\"\"
        payload = {
            "queries": [
                {
                    "q": q.q, "k": q.k, "tags": q.tags,
                    "include_content": q.include_content
                } for q in queries
            ]
        }
        resp = self._request_with_retry("POST", "/search/batch", json=payload)
        self._raise_for_status(resp)
        batch_results = resp.json().get("results", [])
        
        parsed = []
        for br in batch_results:
            results = [
                SearchResult(
                    doc_id=sr["doc_id"], distance=sr["distance"],
                    title=sr.get("title"), content_type=sr.get("content_type", ""),
                    content=sr.get("content"), passage=sr.get("passage"),
                ) for sr in br.get("results", [])
            ]
            parsed.append(BatchSearchResponse(query=br["query"], results=results))
        return parsed
"""

with open(path, "w") as f:
    f.write(text)

print("Patch applied")
