"""Read-only completion/integrity check; Python standard library only."""
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if not summary.get('complete') or summary['manifest'] != sha(root/'manifest.json'):
        raise ValueError('Missing completion marker or changed manifest')
    if summary['events'] != manifest['events'] or summary['heads'] != len(manifest['heads']):
        raise ValueError('Completion counts do not match manifest')
    for name,expected in {**manifest['prepared'],**summary['outputs']}.items():
        if sha(root/name) != expected:
            raise ValueError('Missing/changed result: '+name)
    for head,expected in summary['shards'].items():
        if sha(root/'heads'/(head+'.jsonl.gz')) != expected:
            raise ValueError('Missing/changed OV shard: '+head)
        meta=json.loads((root/'heads'/(head+'.json')).read_text(encoding='utf-8'))
        if meta['manifest'] != summary['manifest'] or meta['sha256'] != expected:
            raise ValueError('Changed shard metadata: '+head)
    print(f"Verified complete: {summary['events']} events, {summary['heads']} heads, "
          f"{summary['candidates']} descriptive candidates.")


if __name__=='__main__':
    verify(Path(sys.argv[1]))
