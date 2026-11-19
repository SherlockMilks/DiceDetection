"""Validate data and produce JSON/Markdown reports without changing inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from dataset_tools import ROOT, image_files, group_id, read_labels


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(raw, split):
    errors, warnings, summaries, records = [], [], {}, {}
    review_items = []
    locations = {'original': (raw/'images', raw/'labels')}
    locations.update({s: (split/'images'/s, split/'labels'/s) for s in ('train', 'val', 'test')})
    for name, (images_dir, labels_dir) in locations.items():
        counts, object_counts, groups, items = Counter(), Counter(), set(), {}
        try:
            images = image_files(images_dir)
        except FileNotFoundError:
            errors.append(f'{name}: missing image directory {images_dir}')
            images = []
        if not images:
            errors.append(f'{name}: no images')
        if not labels_dir.is_dir():
            errors.append(f'{name}: missing label directory {labels_dir}')
        stems = Counter(p.stem for p in images)
        for stem, count in stems.items():
            if count > 1:
                errors.append(f'{name}: ambiguous image stem {stem}')
        for p in images:
            error_start, warning_start = len(errors), len(warnings)
            groups.add(group_id(p))
            try:
                with Image.open(p) as im:
                    im.load()
                    if min(im.size) < 1:
                        raise ValueError('empty image')
            except Exception as exc:
                errors.append(f'{name}/{p.name}: unreadable image: {exc}')
            label = labels_dir/(p.stem+'.txt')
            try:
                boxes = read_labels(label, check_bounds=False)
                counts.update(str(b[0]+1) for b in boxes)
                object_counts[len(boxes)] += 1
                if len(boxes) != 2:
                    warnings.append(f'{name}/{p.name}: {len(boxes)} dice instead of 2')
                read_labels(label)
            except (OSError, ValueError) as exc:
                errors.append(f'{name}: {exc}')
            items[p.name] = {'image_hash': digest(p), 'label_hash': digest(label) if label.is_file() else None}
            if len(errors) > error_start or len(warnings) > warning_start:
                review_items.append({'split': name, 'filename': p.name,
                                     'group': group_id(p),
                                     'issues': errors[error_start:] + warnings[warning_start:]})
        for p in labels_dir.glob('*.txt'):
            if p.stem not in stems:
                errors.append(f'{name}: orphan label {p.name}')
        summaries[name] = {'images': len(images), 'groups': len(groups), 'dice': sum(counts.values()),
                           'values': {str(i): counts[str(i)] for i in range(1, 7)},
                           'dice_per_image': dict(sorted(object_counts.items()))}
        for value in range(1, 7):
            if not counts[str(value)]:
                warnings.append(f'{name}: value {value} absent')
        records[name] = items
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlap = {group_id(p) for p in records[a]} & {group_id(p) for p in records[b]}
        if overlap:
            errors.append(f'{a}/{b}: shared groups {sorted(overlap)}')
        hashes = {v['image_hash'] for v in records[a].values()} & {v['image_hash'] for v in records[b].values()}
        if hashes:
            errors.append(f'{a}/{b}: {len(hashes)} identical image hashes')
    membership = defaultdict(list)
    for name in ('train', 'val', 'test'):
        for filename, item in records[name].items():
            membership[filename].append(name)
            if filename not in records['original']:
                errors.append(f'{name}: unknown image {filename}')
            elif item != records['original'][filename]:
                errors.append(f'{name}: image/label differs from original: {filename}')
    for filename in records['original']:
        if len(membership[filename]) != 1:
            errors.append(f'{filename}: expected exactly one split, got {membership[filename]}')
    return {'generated_at': datetime.now(timezone.utc).isoformat(),
            'raw_directory': str(raw.resolve()), 'split_directory': str(split.resolve()),
            'summary': summaries, 'errors': errors, 'warnings': warnings,
            'review_items': review_items,
            'limitations': 'Filename groups and byte hashes do not detect near-duplicates. Semantic label correctness requires visual review.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/original')
    parser.add_argument('--split', type=Path, default=ROOT/'data/dice_yolo')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/data_audit')
    args = parser.parse_args()
    result = audit(args.raw, args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Adatellenőrzési eredmények', '', '| Halmaz | Kép | Csoport | Kocka | 1 | 2 | 3 | 4 | 5 | 6 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, s in result['summary'].items():
        lines.append('| '+ ' | '.join(map(str, [name, s['images'], s['groups'], s['dice'], *s['values'].values()]))+' |')
    lines += ['', f"Hibák: {len(result['errors'])}. Figyelmeztetések: {len(result['warnings'])}.", '',
              'Az osztályszámok az átalakított képek címkéit is tartalmazzák, nem független dobások számai.', '',
              'A fájlnév- és bájtszintű ellenőrzés nem zárja ki a közel azonos sorozatképeket. A címkék tartalmi helyessége vizuális ellenőrzést igényel.', '']
    lines += ['- '+e for e in result['errors']+result['warnings']]
    (args.output/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))
    print(f"Errors: {len(result['errors'])}; warnings: {len(result['warnings'])}")
    return bool(result['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
