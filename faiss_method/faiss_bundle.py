"""Utilities for building and persisting FAISS indices with custom IDs."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import json
import os

import faiss
import numpy as np

MyID = Union[int, str]


# ---------- core helpers ----------
def _build_base_index(dim: int, metric: str = "cosine") -> faiss.Index:
    if metric == "l2":
        return faiss.IndexFlatL2(dim)
    if metric == "cosine":
        # cosine via inner product on L2-normalized vectors
        return faiss.IndexFlatIP(dim)
    raise ValueError("metric must be 'l2' or 'cosine'")


def _normalize_if_cosine(x: np.ndarray, metric: str) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if metric == "cosine":
        faiss.normalize_L2(x)
    return x


# ---------- build ----------
def build_index_with_my_ids(
    id_to_emb: Dict[MyID, np.ndarray],
    metric: str = "cosine",
) -> Tuple[faiss.Index, Dict[int, MyID], Dict[MyID, int]]:
    """Build an ID-mapped FAISS index from a mapping of IDs to embeddings."""
    ids: List[MyID] = []
    vecs: List[np.ndarray] = []
    for key, value in id_to_emb.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise ValueError(f"embedding for id={key} must be a single vector; got {arr.shape}")
        ids.append(key)
        vecs.append(arr)

    xb = np.stack(vecs, axis=0)
    xb = _normalize_if_cosine(xb, metric)
    dim = xb.shape[1]

    base = _build_base_index(dim, metric)
    index = faiss.IndexIDMap2(base)

    myid_to_id64: Dict[MyID, int] = {}
    id64_to_myid: Dict[int, MyID] = {}
    if all(isinstance(x, int) for x in ids):
        id64 = np.asarray(ids, dtype=np.int64)
        for value in id64:
            myid_to_id64[int(value)] = int(value)
            id64_to_myid[int(value)] = int(value)
    else:
        id64 = np.arange(len(ids), dtype=np.int64)
        for i, key in enumerate(ids):
            myid = int(id64[i])
            myid_to_id64[key] = myid
            id64_to_myid[myid] = key

    index.add_with_ids(xb, id64)
    return index, id64_to_myid, myid_to_id64


# ---------- save / load ----------
def save_bundle(index: faiss.Index, id64_to_myid: Dict[int, MyID], path_index: str) -> None:
    """Save a FAISS index and its accompanying ID mapping to disk."""
    faiss.write_index(index, path_index)
    sidecar = f"{path_index}.idmap.json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump({str(key): value for key, value in id64_to_myid.items()}, handle, ensure_ascii=False)


def load_bundle(path_index: str) -> Tuple[faiss.Index, Dict[int, MyID]]:
    """Load a FAISS index and optional ID mapping from disk."""
    index = faiss.read_index(path_index)
    sidecar = f"{path_index}.idmap.json"
    id64_to_myid: Dict[int, MyID] = {}
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        id64_to_myid = {int(key): value for key, value in raw.items()}
    return index, id64_to_myid


# ---------- orchestrator ----------
def faiss_handler(
    *,
    id_to_emb: Optional[Dict[MyID, np.ndarray]] = None,
    metric: str = "cosine",
    path_index: Optional[str] = None,
    save_index: bool = False,
    load_index: bool = False,
) -> Tuple[faiss.Index, Dict[int, MyID], Dict[MyID, int]]:
    """Create, save, or load a FAISS bundle in a single call.

    Parameters
    ----------
    id_to_emb:
        Mapping from your identifiers to embedding vectors. Required when
        building a new index.
    metric:
        Distance metric to use (``"cosine"`` or ``"l2"``).
    path_index:
        File path for saving or loading the FAISS index.
    save_index:
        When ``True`` and an index is built, the index and ID map are
        written to ``path_index``.
    load_index:
        When ``True``, an existing index is loaded from ``path_index`` and the
        ``id_to_emb`` argument is ignored.

    Returns
    -------
    (index, id64_to_myid, myid_to_id64)
        The FAISS index alongside bidirectional mappings between FAISS int64
        identifiers and the original IDs.
    """

    if load_index:
        if not path_index:
            raise ValueError("path_index must be provided when load_index=True")
        index, id64_to_myid = load_bundle(path_index)
        if not id64_to_myid:
            # Attempt to reconstruct the mapping from the FAISS ID map.
            reconstructed: Dict[int, MyID] = {}
            if isinstance(index, faiss.IndexIDMap2):
                try:
                    stored_ids = faiss.vector_to_array(index.id_map)
                    reconstructed = {int(v): int(v) for v in stored_ids}
                except Exception:  # pragma: no cover - best effort
                    reconstructed = {}
            if not reconstructed:
                reconstructed = {int(i): int(i) for i in range(index.ntotal)}
            id64_to_myid = reconstructed
        myid_to_id64 = {value: key for key, value in id64_to_myid.items()}
        return index, id64_to_myid, myid_to_id64

    if id_to_emb is None:
        raise ValueError("id_to_emb must be provided when load_index=False")

    index, id64_to_myid, myid_to_id64 = build_index_with_my_ids(id_to_emb, metric=metric)

    if save_index:
        if not path_index:
            raise ValueError("path_index must be provided when save_index=True")
        save_bundle(index, id64_to_myid, path_index)

    return index, id64_to_myid, myid_to_id64


# ---------- search (maps back to your IDs) ----------
def search_with_my_ids(
    index: faiss.Index,
    id64_to_myid: Dict[int, MyID],
    xq: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> Tuple[np.ndarray, List[List[MyID]]]:
    """Run a FAISS search and map back to the original identifiers."""
    xq = np.asarray(xq, dtype=np.float32)
    if xq.ndim == 1:
        xq = xq.reshape(1, -1)
    xq = _normalize_if_cosine(xq, metric)

    distances, indices = index.search(xq, k)
    mapped = [[id64_to_myid.get(int(item), int(item)) for item in row] for row in indices]
    return distances, mapped
