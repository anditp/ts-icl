"""
Convert a GiftEvalPretrain directory from Arrow IPC *stream* format to
Arrow IPC *file* (random-access) format, splitting each file into
multiple record batches of at most ``--batch-rows`` rows.

Background
----------
The source files each contain a single huge record batch.  Arrow IPC
*file* format stores a footer that records every record-batch offset,
enabling O(1) random access to any individual batch — but only if there
*are* multiple batches.  By splitting at write time we give
``BaseFileFormatGiftEvalDataset`` the granularity it needs to read only
the batches that contain sampled rows, rather than the whole file.

Usage
-----
    python scripts/stream_to_file.py \
        --src  /path/to/GiftEvalPretrain \
        --dst  /path/to/GiftEvalPretrain_nostream \
        [--workers 8] \
        [--batch-rows 512]

The destination mirrors the source directory tree exactly.
Files that already exist at the destination are skipped, so the script
is safe to re-run after an interruption.
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_stream_to_file(
    src: Path, dst: Path, batch_rows: int = 512
) -> tuple[Path, str | None]:
    """Convert one Arrow IPC stream file to IPC file format.

    The source typically contains a single large record batch.  We split
    it into chunks of at most *batch_rows* rows so that
    ``BaseFileFormatGiftEvalDataset`` can later read only the specific
    batches that contain sampled rows.

    Returns (src, error_message) where error_message is None on success.
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return src, None  # already done — skip

        with pa.OSFile(str(src), "rb") as source:
            table = pa_ipc.open_stream(source).read_all()

        with pa.OSFile(str(dst), "wb") as sink:
            with pa_ipc.new_file(sink, table.schema) as writer:
                for start in range(0, max(table.num_rows, 1), batch_rows):
                    writer.write_batch(
                        table.slice(start, batch_rows).to_batches()[0]
                    )

        return src, None
    except Exception as exc:
        return src, str(exc)


def _worker(args: tuple[Path, Path, int]) -> tuple[Path, str | None]:
    return convert_stream_to_file(*args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GiftEvalPretrain from Arrow stream to Arrow file format."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/data/parietal/store4/data/tsmixup"),
        help="Source tsmixup root directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/data/parietal/store4/data/tsmixup_nostream"),
        help="Destination root directory (created if it does not exist).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, mp.cpu_count()),
        help="Number of parallel worker processes (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=512,
        help=(
            "Number of rows per record batch in the output file "
            "(default: 512).  Smaller values give finer random-access "
            "granularity at the cost of a slightly larger footer."
        ),
    )
    args = parser.parse_args()

    src_root: Path = args.src.resolve()
    dst_root: Path = args.dst.resolve()
    batch_rows: int = args.batch_rows

    if not src_root.is_dir():
        print(f"ERROR: source directory does not exist: {src_root}", file=sys.stderr)
        sys.exit(1)

    # Collect all data-*.arrow files and their destination paths
    jobs: list[tuple[Path, Path, int]] = []
    for src_file in sorted(src_root.rglob("data-*.arrow")):
        rel = src_file.relative_to(src_root)
        dst_file = dst_root / rel
        jobs.append((src_file, dst_file, batch_rows))

    if not jobs:
        print(f"No 'data-*.arrow' files found under {src_root}.")
        sys.exit(0)

    n_skip = sum(1 for _, d, _ in jobs if d.exists())
    n_todo = len(jobs) - n_skip
    print(
        f"Found {len(jobs)} files  "
        f"({n_skip} already converted, {n_todo} to do)  "
        f"using {args.workers} workers, {batch_rows} rows/batch."
    )

    if n_todo == 0:
        print("Nothing to do.")
        return

    errors: list[tuple[Path, str]] = []
    done = 0

    with mp.Pool(processes=args.workers) as pool:
        for src_file, err in pool.imap_unordered(_worker, jobs):
            done += 1
            if err:
                errors.append((src_file, err))
                print(f"  ERROR  {src_file.relative_to(src_root)}: {err}")
            else:
                print(f"  [{done}/{len(jobs)}]  {src_file.relative_to(src_root)}", end="\r")

    print()  # newline after \r progress
    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for p, e in errors:
            print(f"  {p}: {e}")
        sys.exit(1)
    else:
        print(f"Done. Converted files are in {dst_root}")


if __name__ == "__main__":
    main()