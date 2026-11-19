import tempfile
import unittest
from pathlib import Path
from dataset_tools import read_labels, group_id
from create_split import create_split
from audit_dataset import audit
from PIL import Image


class DatasetTests(unittest.TestCase):
    def test_audit_flags_bad_boxes_and_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, split = root/'raw', root/'split'
            for folder in [raw/'images', raw/'labels']:
                folder.mkdir(parents=True)
            for index, part in enumerate(('train', 'val', 'test')):
                for kind in ('images', 'labels'):
                    (split/kind/part).mkdir(parents=True)
                name = f'{index}.png'
                Image.new('RGB', (20, 20), (index * 50, 0, 0)).save(raw/'images'/name)
                label = '0 0.5 0.5 0.2 0.2\n1 0.8 0.8 0.1 0.1'
                if part == 'train':
                    label = '0 0.01 0.5 0.2 0.2'
                (raw/'labels'/f'{index}.txt').write_text(label)
                (split/'images'/part/name).write_bytes((raw/'images'/name).read_bytes())
                (split/'labels'/part/f'{index}.txt').write_text(label)
            # Different filename, identical image bytes: must still detect leakage.
            (split/'images'/'val'/'1.png').write_bytes((raw/'images'/'0.png').read_bytes())
            result = audit(raw, split)
            self.assertTrue(any('identical image hashes' in e for e in result['errors']))
            self.assertTrue(any('differs from original' in e for e in result['errors']))
            original_issues = [i for i in result['review_items'] if i['split'] == 'original']
            self.assertEqual(len(original_issues), 1)
            self.assertEqual(original_issues[0]['filename'], '0.png')
            self.assertEqual(len(original_issues[0]['issues']), 2)

    def test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'label.txt'
            p.write_text('0 0.5 0.5 0.2 0.2\n5 0.8 0.8 0.1 0.1')
            self.assertEqual([b[0] for b in read_labels(p)], [0, 5])
            for bad in ['', '6 0.5 0.5 0.2 0.2', '0 nan 0.5 0.2 0.2',
                        '0 0.5 0.5 -0.2 0.2', '0 0.01 0.5 0.2 0.2', '0 0.5 0.5']:
                p.write_text(bad)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    read_labels(p)

    def test_grouped_split_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'raw'
            (raw/'images').mkdir(parents=True)
            (raw/'labels').mkdir()
            for i in range(20):
                for suffix in ['', '__fliph']:
                    name = f'{i:03d}{suffix}'
                    (raw/'images'/f'{name}.jpg').write_bytes(b'fixture')
                    (raw/'labels'/f'{name}.txt').write_text('0 0.5 0.5 0.2 0.2')
            out = root/'out'
            create_split(raw, out)
            sets = [{group_id(p) for p in (out/'images'/s).iterdir()} for s in ('train', 'val', 'test')]
            self.assertEqual(list(map(len, sets)), [16, 2, 2])
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            before = (out/'split_manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                create_split(raw, out)
            self.assertEqual(before, (out/'split_manifest.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
