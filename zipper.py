import numpy as np
from pathlib import Path

TGT = Path('./npz')
COMBINE_SIZE = 2
KEYS = ['imgs', 'labels', 'ids']

paths = sorted(
    [p for p in TGT.iterdir() if p.is_file() and p.suffix == '.npz'],
    key=lambda p: int(p.stem.split('-')[0])
)

if not paths:
    print(f'No .npz files found in {TGT.resolve()}')
    raise SystemExit(0)


def combine_batch(batch):
    combined = {}
    for index, path in enumerate(batch):
        with np.load(path) as data:
            for key in KEYS:
                if index == 0:
                    combined[key] = data[key]
                else:
                    combined[key] = np.concatenate((combined[key], data[key]), axis=0)

    start = batch[0].stem.split('-')[0]
    end = batch[-1].stem.split('-')[-1]
    target_path = TGT / f'{start}-{end}.npz'
    np.savez(target_path, **combined)
    return target_path


def main():
    saved_files = []
    processed_files = []
    batch = []

    for path in paths:
        batch.append(path)
        if len(batch) == COMBINE_SIZE:
            saved_files.append(combine_batch(batch))
            processed_files.extend(batch)
            batch = []

    if batch:
        saved_files.append(combine_batch(batch))
        processed_files.extend(batch)

    for path in processed_files:
        try:
            path.unlink()
        except OSError as exc:
            print(f'Failed to remove {path}: {exc}')

    print(f'Combined {len(processed_files)} files into {len(saved_files)} .npz files.')


if __name__ == '__main__':
    main()
