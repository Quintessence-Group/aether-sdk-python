# aether-ai

Python SDK for the [Aether](https://aetherdb.ai) decentralized RAG API.

## Installation

```bash
pip install aether-ai
```

## Memory — the fastest way to build agent memory

For per-user or per-agent memory, reach for the `Memory` facade. Construct it once
with an entity id and every call is automatically scoped to that entity — no tags or
filters to manage:

```python
from aether import Memory

mem = Memory("patient-john", api_key="aether_your_key_here")

# Store a memory
mem.remember("Anxious about flying; uses 4-7-8 breathing")

# Recall the most relevant memories for this entity
for item in mem.recall("anxiety coping"):
    print(item.score, item.text)

# Newest-first history, or wipe the slate
mem.list(limit=20)
mem.forget_all()
```

- `recall(query, k=5, recency_weight=0.0, since=..., until=..., filter=...)` blends
  relevance (a `score` in `(0, 1]`, higher is better) with optional exponential
  recency decay, and can filter on the metadata you stored.
- `remember(text, metadata={...})` stores the memory and writes `metadata` as
  searchable `key:value` tags.
- `AsyncMemory` mirrors the same surface with `await` on every call.

The raw `AetherClient` below is the lower-level API — use it when you need direct
control over documents, search, and batch operations rather than entity-scoped memory.

## Quick Start

```python
from aether import AetherClient

client = AetherClient(api_key="aether_your_key_here")

# Insert a file — content type is auto-detected from the extension
doc = client.insert("report.pdf")
print(f"Inserted: {doc.doc_id}")

# Insert raw text
doc = client.insert_text("Some text content to index")

# Search
results = client.search("machine learning", k=5)
for r in results:
    print(f"  {r.doc_id} (score: {r.score}) - {r.passage}")

# List documents
for doc in client.list():
    print(f"  {doc.doc_id}: {doc.title}")
```

## Supported File Formats

Aether automatically extracts clean text from binary documents before embedding. No need to specify `content_type` -- it's guessed from the file extension.

| Format | Extensions |
|--------|-----------|
| PDF | .pdf |
| Word | .docx, .doc |
| PowerPoint | .pptx, .ppt |
| Excel | .xlsx, .xls |
| HTML | .html, .htm |
| CSV | .csv |
| Plain text | .txt, .md, .json, .xml |

Binary-format parsing is handled automatically server-side — no setup required.

## RAG Quick Start

Use `retrieve()` to search and get document content in a single call -- ready to pass into any LLM:

```python
from aether import AetherClient

client = AetherClient(api_key="your_key")

# Insert documents (PDF, DOCX, XLSX — all auto-detected)
client.insert("company-handbook.pdf")
client.insert("benefits-guide.docx")
client.insert_text("Remote work is allowed 3 days per week...")

# Retrieve relevant documents with content
results = client.retrieve("How much PTO do I get?", k=3)
for r in results:
    print(f"{r.title}: {r.content[:100]}...")
```

For complete RAG examples with Anthropic, OpenAI, Azure, and more, see [examples/](../../examples/).

## Async Processing

For large files or frontend integrations that need progress feedback, use the async REST endpoint:

```python
import httpx, time

resp = httpx.post(
    "https://api.aetherdb.ai/documents/async?filename=report.pdf",
    content=open("report.pdf", "rb").read(),
    headers={"Authorization": "Bearer aether_..."},
)
job = resp.json()  # {"job_id": "...", "poll_url": "/documents/jobs/..."}

while True:
    status = httpx.get(
        f"https://api.aetherdb.ai{job['poll_url']}",
        headers={"Authorization": "Bearer aether_..."},
    ).json()
    print(f"{status['progress']:.0%} - {status['message']}")
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(0.5)
```

## CLI

```bash
aether-py status
aether-py insert document.pdf
aether-py search "your query"
aether-py list
```

## License

MIT
