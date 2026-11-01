"""Freeze non-whitelisted 261030 crops through frame 430 for offline analysis.

No acquisition files are modified. Copy completion is recorded per channel after
SHA256 comparison; interruptions resume at the last completed channel.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import time

REPO = Path(__file__).resolve().parents[1]
OUT = Path('C:/QPI_analysis/261030')
SOURCE = Path('D:/AquisitionData/Kitagishi/261030/online_crop_sub_YE_5min_31d')
REL = Path('output_phase/channels/crop_sub_rawraw/z000')


def main():
    inputs = OUT / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for name in ('channel_whitelist.json', 'drift_config_zstack.json'):
        dest = inputs / name
        if not dest.exists():
            shutil.copy2(REPO / 'drift_session_261030_YE' / name, dest)
    whitelist = json.loads((inputs / 'channel_whitelist.json').read_text())['whitelist']
    inventory = json.loads((OUT / 'inventory.json').read_text())
    selected = inventory['channels']
    (inputs / 'selected_channels.txt').write_text(''.join(f"{r['pos']} {r['ch']}\n" for r in selected))
    (inputs / 'pilot_channels.txt').write_text(''.join(f"{r['pos']} {r['ch']}\n" for r in selected[:2]))
    start = time.monotonic()
    for i, row in enumerate(selected, 1):
        assert int(row['ch'][2:]) not in whitelist.get(row['pos'], [])
        base = Path(row['pos']) / REL / row['ch']
        dest = OUT / 'phase_f0000-0430' / base
        marker = inputs / 'copy_checks' / (row['pos'] + '_' + row['ch'] + '.json')
        if marker.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in (SOURCE / base).glob('img_*_ph_000.tif')
                       if 0 <= int(p.name.split('_')[1]) <= 430)
        assert len(files) == row['n'], (base, len(files), row['n'])
        def copy_verified(src):
            data = src.read_bytes()
            target = dest / src.name
            target.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, target
            return src.name, digest
        with ThreadPoolExecutor(max_workers=8) as pool:
            hashes = dict(pool.map(copy_verified, files))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(dict(source=str(SOURCE / base), sha256=hashes), indent=2))
        print(f"copy {i}/{len(selected)} {row['pos']} {row['ch']} files={len(files)} elapsed={time.monotonic()-start:.0f}s", flush=True)
    (inputs / 'SNAPSHOT_COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S'))
    print('SNAPSHOT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
