"""Segment verified snapshot channels as the copy finishes; resume using _DONE."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import json
import time
import seg_omnipose as seg

ROOT = Path('C:/QPI_analysis/261030')
REL = Path('output_phase/channels/crop_sub_rawraw/z000')


def main():
    rows = json.loads((ROOT / 'inventory.json').read_text())['channels']
    seg._ensure_cuda_dlls_on_path()
    initargs = (str(Path(__file__).resolve().parents[1] / 'models/omni_model_d20_2026_09_07_12_41_31.782047'),
                seg.EVAL_DEFAULT, 0.7, 40, str(ROOT / 'phase_f0000-0430'),
                str(ROOT / 'seg'), 'img_*_ph_000.tif', None)
    pending = list(rows)
    active = {}
    completed = 0
    errors = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=6, initializer=seg._init, initargs=initargs) as pool:
        while pending or active:
            for row in list(pending):
                if len(active) >= 6:
                    break
                key = row['pos'] + '_' + row['ch']
                if not (ROOT / 'inputs/copy_checks' / (key + '.json')).exists():
                    continue
                ch = ROOT / 'phase_f0000-0430' / row['pos'] / REL / row['ch']
                # Every file must be inside the frozen window, even on resume.
                assert all(0 <= seg._frame_no(p) <= 430 for p in ch.glob('img_*.tif'))
                f = pool.submit(seg.process_channel, (int(row['pos'][3:]), str(ch)))
                active[f] = key
                pending.remove(row)
            if not active:
                time.sleep(5)
                continue
            finished, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for f in finished:
                key = active.pop(f)
                result = f.result()
                completed += 1
                print(f'{completed}/{len(rows)} {key} {result} elapsed={time.monotonic()-started:.0f}s', flush=True)
                if result[2] == 'ERROR':
                    errors.append(result)
    (ROOT / 'seg_summary.json').write_text(json.dumps(dict(channels=completed, errors=errors,
        elapsed_s=time.monotonic()-started), indent=2))
    if errors:
        raise RuntimeError(f'{len(errors)} failed channels; inspect seg_summary.json')
    (ROOT / 'SEG_COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S'))


if __name__ == '__main__':
    main()
