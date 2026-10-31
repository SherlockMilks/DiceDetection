"""Draw ground truth labels, not model predictions, in an offline HTML gallery."""
import argparse
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dataset_tools import ROOT, image_files, group_id, read_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=ROOT/'data/original/images')
    parser.add_argument('--labels', type=Path, default=ROOT/'data/original/labels')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/label_preview')
    parser.add_argument('--limit', type=int, default=0, help='0 means all images')
    parser.add_argument('--originals-only', action='store_true')
    parser.add_argument('--review-json', type=Path,
                        help='Only show original images flagged by audit_dataset.py')
    args = parser.parse_args()
    files = image_files(args.images)
    review = {}
    if args.review_json:
        report = json.loads(args.review_json.read_text(encoding='utf-8'))
        review = {item['filename']: item['issues'] for item in report['review_items']
                  if item['split'] == 'original'}
        files = [p for p in files if p.name in review]
    if args.originals_only:
        files = [p for p in files if p.stem == group_id(p)]
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error('No matching images')
    args.output.mkdir(parents=True, exist_ok=True)
    cards = []
    font = ImageFont.load_default(size=20)
    for number, p in enumerate(files):
        issue = '; '.join(review.get(p.name, []))
        try:
            boxes = read_labels(args.labels/(p.stem+'.txt'))
        except (OSError, ValueError) as exc:
            issue = issue or str(exc)
            try:
                boxes = read_labels(args.labels/(p.stem+'.txt'), check_bounds=False)
            except (OSError, ValueError):
                boxes = []
        try:
            with Image.open(p) as src:
                im = src.convert('RGB')
        except (OSError, ValueError) as exc:
            cards.append(f'<figure><figcaption>{html.escape(p.name)}<br>'
                         f'<strong>Nem olvasható kép: {html.escape(str(exc))}</strong></figcaption></figure>')
            continue
        im.thumbnail((1000, 750))
        w, h = im.size
        draw = ImageDraw.Draw(im)
        for cls, x, y, bw, bh in boxes:
            rect = ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)
            draw.rectangle(rect, outline='#00ff66', width=3)
            pos = (max(0, rect[0]), max(0, rect[1]-25))
            label = f'{cls+1} (id {cls})'
            draw.rectangle(draw.textbbox(pos, label, font=font), fill='black')
            draw.text(pos, label, font=font, fill='white')
        target = f'{number:04d}.jpg'
        im.save(args.output/target, quality=90)
        cards.append(f'<figure><img loading="lazy" src="{target}"><figcaption>{html.escape(p.name)}<br><strong>{html.escape(issue)}</strong></figcaption></figure>')
    page = '<!doctype html><html lang="hu"><meta charset="utf-8"><title>Címkeellenőrzés</title><style>body{font:16px sans-serif;margin:24px;background:#eee}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}figure{margin:0;background:white;padding:12px}img{width:100%}figcaption{overflow-wrap:anywhere}</style><h1>Címkeellenőrzés</h1><p>A megadott címkék láthatók, nem modelljóslatok. Ellenőrizd a kereteket és a felső lapok pontértékét.</p><main>'+''.join(cards)+'</main></html>'
    (args.output/'index.html').write_text(page, encoding='utf-8')
    print(f'{len(files)} previews: {args.output / "index.html"}')


if __name__ == '__main__':
    main()
