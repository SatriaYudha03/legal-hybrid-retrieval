from pathlib import Path
import json
import sys

try:
    import faiss
except Exception as e:
    print("faiss tidak tersedia:", e)
    sys.exit(1)

proj_root = Path(__file__).resolve().parent.parent
meta_path = proj_root / "data" / "index" / "faiss.meta.json"
index_path = proj_root / "data" / "index" / "faiss.faiss"

if not meta_path.exists() or not index_path.exists():
    print("Index atau metadata tidak ditemukan:", meta_path, index_path)
    sys.exit(1)

meta = json.loads(meta_path.read_text(encoding='utf-8'))
doc_ids = meta.get('doc_ids', [])

index = faiss.read_index(str(index_path))
print(f"Index ntotal: {index.ntotal}, dim: {index.d}")

n = min(3, index.ntotal)
for i in range(n):
    try:
        v = index.reconstruct(i)
    except Exception:
        # fallback: use index.reconstruct_n if available
        arr = faiss.vector_to_array(index.reconstruct(i))
        v = arr
    print(f"-- doc_id[{i}] =", doc_ids[i] if i < len(doc_ids) else '(no-id)')
    # show first 10 dimensions for brevity
    print("vector (first 10 dims):", list(map(float, v[:10])))
    print("vector dtype/len:", getattr(v, 'dtype', None), len(v))
    print()
