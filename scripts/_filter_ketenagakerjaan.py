import json, shutil
from pathlib import Path

eval_dir = Path('data/eval')

# backup
shutil.copy(eval_dir / 'queries.json', eval_dir / 'queries.json.bak')
shutil.copy(eval_dir / 'qrels.json',   eval_dir / 'qrels.json.bak')

# filter queries
queries = json.loads((eval_dir / 'queries.json').read_text(encoding='utf-8'))
queries_filtered = [q for q in queries if q['domain'] != 'ketenagakerjaan']
(eval_dir / 'queries.json').write_text(
    json.dumps(queries_filtered, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'queries: {len(queries)} -> {len(queries_filtered)}')

# filter qrels
ketenaga_ids = {q['id'] for q in queries if q['domain'] == 'ketenagakerjaan'}
qrels = json.loads((eval_dir / 'qrels.json').read_text(encoding='utf-8'))
qrels_filtered = {k: v for k, v in qrels.items()
                  if k.startswith('_') or k not in ketenaga_ids}
(eval_dir / 'qrels.json').write_text(
    json.dumps(qrels_filtered, ensure_ascii=False, indent=2), encoding='utf-8')

total_before = sum(1 for k in qrels if not k.startswith('_'))
total_after  = sum(1 for k in qrels_filtered if not k.startswith('_'))
print(f'qrels: {total_before} -> {total_after} query')
print(f'Query dihapus: {sorted(ketenaga_ids)}')
