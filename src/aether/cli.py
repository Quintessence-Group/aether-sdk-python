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
        typer.echo(f"  {i}. {r.doc_id} (distance: {r.distance:.4f}) - {r.title or 'untitled'}")


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
    typer.echo(f"Node {s.node_id}:")
    typer.echo(f"  Documents: {s.documents}")
    typer.echo(f"  Vectors:   {s.vectors}")
    typer.echo(f"  Version:   {s.version or 'unknown'}")


# Note: cluster, validate, recover, and metrics are admin-only operations that
# are not exposed in the public SDK (see AetherClient). Use the REST API directly
# with an admin API key for those operational tasks.


if __name__ == "__main__":
    app()
