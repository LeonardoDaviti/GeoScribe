#!/usr/bin/env python3
"""Convert a '# %%'-cell-marked .py into a Colab-ready .ipynb (stdlib only)."""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
cells, cur, kind = [], [], None

def flush():
    if kind is None or not any(l.strip() for l in cur):
        return
    lines = cur[:]
    while lines and not lines[0].strip(): lines.pop(0)
    while lines and not lines[-1].strip(): lines.pop()
    if kind == 'markdown':
        lines = [l[2:] if l.startswith('# ') else (l[1:] if l.startswith('#') else l) for l in lines]
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': [l + '\n' for l in lines]})
    else:
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': [l + '\n' for l in lines]})

for line in open(src, encoding='utf-8').read().splitlines():
    if line.startswith('# %%'):
        flush()
        cur, kind = [], ('markdown' if '[markdown]' in line else 'code')
    else:
        cur.append(line)
flush()

nb = {
    'nbformat': 4, 'nbformat_minor': 0,
    'metadata': {
        'colab': {'provenance': [], 'gpuType': 'T4'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'accelerator': 'GPU',
    },
    'cells': cells,
}
json.dump(nb, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'{dst}: {len(cells)} cells')
