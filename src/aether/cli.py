"""Aether Python CLI (typer-based)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from .client import AetherClient

app = typer.Typer(name="aether-py", help="Aether dRAG Python CLI")

DEFAULT_URL = os.environ.get("AETHER_BASE_URL", "https://api.aetherdb.ai")


def get_client(node: str) -> AetherClient:
    return AetherClient(node)


@app.command()
def insert(
    file: Path,
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Insert a document."""
    client = get_client(node)
    doc = client.insert(file)
    typer.echo(f"Inserted: {doc.doc_id} (CID: {doc.cid}, v{doc.version})")


@app.command()
def search(
    query: str,
    k: int = typer.Option(10, "-k", help="Number of results"),
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Search for similar documents."""
    client = get_client(node)
    results = client.search(query, k=k)
    if not results:
        typer.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        typer.echo(f"  {i}. {r.doc_id} (score: {r.score}) - {r.title or 'untitled'}")


@app.command("list")
def list_docs(
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """List all active documents."""
    client = get_client(node)
    docs = client.list()
    typer.echo(f"Documents ({len(docs)}):")
    for d in docs:
        typer.echo(f"  {d.doc_id} v{d.version} - {d.title or 'untitled'} ({d.size_bytes} bytes)")


@app.command()
def get(
    doc_id: str,
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Get document metadata."""
    client = get_client(node)
    doc = client.get(doc_id)
    typer.echo(f"Document {doc.doc_id}:")
    typer.echo(f"  CID:     {doc.cid}")
    typer.echo(f"  Title:   {doc.title}")
    typer.echo(f"  Size:    {doc.size_bytes} bytes")
    typer.echo(f"  Version: {doc.version}")


@app.command()
def download(
    doc_id: str,
    output: Path = typer.Option(..., "-o", help="Output path"),
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Download a document."""
    client = get_client(node)
    n = client.download(doc_id, output)
    typer.echo(f"Downloaded {n} bytes to {output}")


@app.command()
def delete(
    doc_id: str,
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Tombstone a document."""
    client = get_client(node)
    client.delete(doc_id)
    typer.echo(f"Tombstoned {doc_id}")


@app.command()
def status(
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Show node status."""
    client = get_client(node)
    s = client.status()
    typer.echo(f"Node {s.node_id} ({'cluster' if s.cluster_mode else 'standalone'}):")
    typer.echo(f"  Documents: {s.documents} (+{s.tombstoned} tombstoned)")
    typer.echo(f"  Vectors:   {s.vectors}")
    typer.echo(f"  Shards:    {s.shards}")
    typer.echo(f"  Events:    {s.events}")
    typer.echo(f"  Tokens:    {s.token_balance}")


@app.command()
def cluster(
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Show cluster status."""
    client = get_client(node)
    c = client.cluster_status()
    typer.echo(f"Cluster: {'active' if c.cluster_mode else 'standalone'}")
    typer.echo(f"  Healthy: {c.healthy_nodes}/{c.total_nodes}")
    for p in c.peers:
        icon = "\u2713" if p.healthy else "\u2717"
        typer.echo(f"  [{icon}] Node {p.node_id} - {p.api} (events: {p.event_count})")


@app.command()
def validate(
    golden_path: str = typer.Option("tests/golden/queries.json", help="Golden queries path"),
    k: int = typer.Option(5, "-k"),
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Run golden query validation."""
    client = get_client(node)
    report = client.validate(golden_path, k)
    typer.echo(f"Validation: {report.passed}/{report.total_queries} passed ({report.accuracy:.0%})")
    for r in report.results:
        status = "\u2713" if r.get("passed") else "\u2717"
        typer.echo(f"  [{status}] {r.get('query', '')}")


@app.command()
def recover(
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Run full disaster recovery."""
    client = get_client(node)
    r = client.recover()
    typer.echo("Recovery complete:")
    typer.echo(f"  Events synced: {r.events_synced}")
    typer.echo(f"  Shards: {r.recovered_shards} recovered, {r.unrecoverable_shards} lost")
    typer.echo(f"  Vectors: {r.recovered_vectors} recovered, {r.unrecoverable_vectors} lost")


@app.command("metrics")
def show_metrics(
    node: str = typer.Option(DEFAULT_URL, help="Aether API URL"),
):
    """Show node metrics."""
    client = get_client(node)
    m = client.metrics()
    typer.echo(f"Metrics (uptime: {m.uptime_secs}s):")
    typer.echo(f"  Queries:  {m.queries_total}")
    typer.echo(f"  Inserts:  {m.inserts_total}")
    typer.echo(f"  Updates:  {m.updates_total}")
    typer.echo(f"  Deletes:  {m.deletes_total}")
    typer.echo(f"  Docs:     {m.documents_active}")
    typer.echo(f"  Vectors:  {m.vectors_count}")
    typer.echo(f"  Tokens:   {m.token_balance}")


if __name__ == "__main__":
    app()
