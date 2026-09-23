"""Verify public per-node judgments against archived aggregate counts."""
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
public = json.loads((root/'public-evaluation.json').read_text())
audit = json.loads((root/'evaluation-audit.json').read_text())
counts = Counter()
seen = set()
for row in audit['nodes']:
    key = (row['row_index'], row['message_index'])
    assert key not in seen, key
    seen.add(key)
    for name, flag in row['counts'].items():
        assert flag in (0, 1)
        counts[name] += flag
for name, value in counts.items():
    assert value == public['metrics'][name], (name, value, public['metrics'][name])
assert len(seen) == 378 and len({x[0] for x in seen}) == 200
print('PASS: 378 unique nodes / 200 conversations; all 10 counts match.')
