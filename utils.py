# utils.py
import hashlib
import shelve
import os
from typing import List, Tuple
import torch
import numpy as np
from Bio import SeqIO
import io
import logging
import yaml

logger = logging.getLogger(__name__)

def load_config(path='config.yaml'):
    import yaml
    with open(path, 'r') as fh:
        cfg = yaml.safe_load(fh)
    return cfg

def parse_fasta_bytes(content: bytes) -> List[str]:
    """
    Parse FASTA content bytes and return list of sequences (strings).
    If the provided content is raw newline-separated sequences, attempt that fallback.
    """
    seqs = []
    try:
        s = content.decode('utf-8')
    except UnicodeDecodeError:
        logger.exception("Failed to decode file content as utf-8")
        return []

    # Try FASTA first
    try:
        fh = io.StringIO(s)
        for rec in SeqIO.parse(fh, "fasta"):
            seqs.append(str(rec.seq).strip())
    except Exception:
        seqs = []

    if not seqs:
        # fallback to newline-separated sequences (simple)
        for line in s.splitlines():
            line = line.strip()
            if line and not line.startswith('>'):
                seqs.append(line)
    return seqs

def seq_hash(sequence: str) -> str:
    return hashlib.md5(sequence.encode('utf-8')).hexdigest()

# Simple shelve-based cache for embeddings
class EmbeddingCache:
    def __init__(self, path='cache/embeddings.db'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.path = path
        self._db = None

    def __enter__(self):
        self._db = shelve.open(self.path)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._db.close()
        except Exception:
            pass

    def get(self, key):
        return self._db.get(key)

    def set(self, key, val):
        self._db[key] = val

    def close(self):
        if self._db is not None:
            self._db.close()

def sequence_to_tokens(sequence: str, max_len: int = 1000) -> torch.LongTensor:
    """
    Convert DNA sequence to integer tokens (A,T,G,C,N => 0..4), pad/truncate to max_len.
    Returns tensor shape (max_len,) as LongTensor.
    """
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    tokens = [mapping.get(base.upper(), 4) for base in sequence]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [4] * (max_len - len(tokens))
    return torch.LongTensor(tokens)

def batchify_sequences(seqs: List[str], batch_size: int):
    for i in range(0, len(seqs), batch_size):
        yield seqs[i:i+batch_size]

def compute_gc(seq: str) -> float:
    s = seq.upper()
    gc = s.count('G') + s.count('C')
    if len(s) == 0:
        return 0.0
    return gc / len(s)
