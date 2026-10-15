#!/usr/bin/env python3
"""Export active-control time windows; never delete or concatenate raw time."""
import argparse
import json
from pathlib import Path
from recording import atomic_json


def segments(rows):
    start = last = None
    result = []
    for row in rows:
        now = row['received_ns']
        if row['data']['engaged']:
            if start is None:
                start = now
            last = now
        elif start is not None:
            result.append({'start_ns': start, 'end_ns_exclusive': now, 'last_active_sample_ns': last})
            start = last = None
    if start is not None:
        result.append({'start_ns': start, 'end_ns_exclusive': last+1, 'last_active_sample_ns': last})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists')
    episodes = []
    for path in sorted(args.session.glob('episode_*')):
        if not (path / 'control.jsonl').exists():
            continue
        meta = json.loads((path / 'episode.json').read_text())
        with (path / 'control.jsonl').open() as f:
            windows = segments(json.loads(line) for line in f)
        episodes.append({'episode': path.name, 'outcome': meta['outcome'], 'segments': windows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {'session': str(args.session.resolve()), 'criterion': 'engaged',
                            'clock': 'host monotonic_ns', 'episodes': episodes})


if __name__ == '__main__':
    main()
