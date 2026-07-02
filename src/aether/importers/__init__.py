"""Importers that migrate data from other memory stores into Aether.

Each importer maps a foreign export into Aether's memory model **through the
public ``Memory`` facade**, so imported data is immediately queryable via
``recall`` / ``list`` / the memory-graph read methods. See
``docs/importers/`` in the monorepo for the per-source mapping documents.
"""

from .mem0 import Mem0ImportReport, import_mem0

__all__ = [
    "Mem0ImportReport",
    "import_mem0",
]
