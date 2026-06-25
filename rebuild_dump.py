import pathlib

files = [
    ('ofp/data_schema.py',        'ofp/data_schema.py'),
    ('ofp/api_streamer.py',       'ofp/api_streamer.py'),
    ('ofp/book_reconstructor.py', 'ofp/book_reconstructor.py'),
    ('ofp/feature_extractor.py',  'ofp/feature_extractor.py'),
    ('ofp/grid_sweeper.py',       'ofp/grid_sweeper.py'),
    ('run_research.py',            'run_research.py'),
    ('verify_dataset.py',          'verify_dataset.py'),
    ('train_model.py',             'train_model.py'),
]

lines = []
for label, p in files:
    lines.append(f'# --- {label} ---')
    lines.append(pathlib.Path(p).read_text(encoding='utf-8'))

pathlib.Path('raw_audit_dump.txt').write_text('\n'.join(lines), encoding='utf-8')
print(f'Wrote {len(lines)} lines, {pathlib.Path("raw_audit_dump.txt").stat().st_size} bytes')
