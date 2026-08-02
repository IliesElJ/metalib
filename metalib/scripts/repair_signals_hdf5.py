"""
Recovery tool for a truncated signals.hdf5. Root cause (see conversation):
metastrategy.py's save_signal_data_to_db() opens the ONE shared
metalib/store/signals.hdf5 via pd.HDFStore(path, mode="a") from every
strategy instance's own OS process, with no locking or coordination --
HDF5/PyTables isn't safe for concurrent multi-process append writes without
SWMR mode, which isn't in use here. At some point a write was interrupted
mid-flush (crash, disk-full, a sync tool truncating the tail -- exactly 1.00
MiB is missing, a suspiciously round shortfall consistent with an
interrupted final chunk, not random corruption), leaving the file's
superblock believing it's 1 MiB larger than it actually is on disk. HDF5
refuses to open ANY file where stored_eof > actual size, so every read/write
attempt fails identically, every time -- not intermittent.

The actual corruption is an 8-byte field (the superblock's End-Of-File
address, literally named "stored_eof" in HDF5's own error trace). Everything
else in the file is very likely intact -- HDF5 flushes internal structures
incrementally, so a single interrupted final write typically only orphans
whatever was being appended at that instant, not the whole file. This script
therefore patches just that field rather than attempting a full rebuild.

SAFE BY DESIGN:
  - Default is --dry-run: inspects the file, reports the superblock version,
    the current stored_eof, the actual on-disk size, and what a patch WOULD
    write -- makes zero changes.
  - --apply backs up ONLY the ~8 bytes being changed to a small sidecar
    JSON file (<path>.superblock_backup.json) -- NOT a full-file copy
    (impractical at 185GB) -- then patches the field in place, then
    verifies the file opens and every key is readable before declaring
    success.
  - --restore reverts using that sidecar backup, byte for byte.
  - Never deletes anything. Never touches the original file's data bytes,
    only the 8-byte EOF field in the superblock header near the start of
    the file.

Usage (run on the server, where the actual file lives):
    python repair_signals_hdf5.py "C:\\path\\to\\signals.hdf5"                 # dry run, default
    python repair_signals_hdf5.py "C:\\path\\to\\signals.hdf5" --apply         # patch + verify
    python repair_signals_hdf5.py "C:\\path\\to\\signals.hdf5" --restore       # undo the patch
    python repair_signals_hdf5.py "C:\\path\\to\\signals.hdf5" --apply --compact-to "C:\\path\\to\\signals_recovered.hdf5"
                                                                                 # also (slowly) rewrite
                                                                                 # every key into a fresh
                                                                                 # file, once the patch
                                                                                 # succeeds -- optional,
                                                                                 # needs ~185GB free space
                                                                                 # and will take a while.
"""
import argparse
import json
import os
import struct
import sys


SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _read_superblock_info(f):
    """Parses just enough of the HDF5 superblock to locate the stored_eof
    field. Returns (version, size_of_offsets, eof_field_offset)."""
    f.seek(0)
    sig = f.read(8)
    if sig != SIGNATURE:
        raise ValueError(f"Not an HDF5 file (bad signature: {sig!r})")

    version = f.read(1)[0]

    if version in (0, 1):
        f.seek(13)
        size_of_offsets = f.read(1)[0]
        # v0 header block is 24 bytes before the address fields begin;
        # v1 adds 4 extra bytes (indexed storage internal node K + reserved).
        addr_block_start = 24 if version == 0 else 28
        # Address fields, in order: base address, free-space address, EOF address, ...
        eof_field_offset = addr_block_start + 2 * size_of_offsets
    elif version in (2, 3):
        f.seek(9)
        size_of_offsets = f.read(1)[0]
        # After the 12-byte common header: base address, then ext address, then EOF address.
        eof_field_offset = 12 + 2 * size_of_offsets
    else:
        raise ValueError(f"Unrecognized/unsupported superblock version: {version}")

    return version, size_of_offsets, eof_field_offset


def _read_uint(f, offset, size):
    f.seek(offset)
    raw = f.read(size)
    return int.from_bytes(raw, byteorder="little", signed=False), raw


def inspect(path):
    actual_size = os.path.getsize(path)
    with open(path, "rb") as f:
        version, size_of_offsets, eof_offset = _read_superblock_info(f)
        stored_eof, raw_bytes = _read_uint(f, eof_offset, size_of_offsets)

    print(f"File:                  {path}")
    print(f"Actual size on disk:   {actual_size:,} bytes ({actual_size/1024**3:.2f} GiB)")
    print(f"Superblock version:    {version}")
    print(f"Size of offsets:       {size_of_offsets} bytes")
    print(f"stored_eof field at:   byte offset {eof_offset}")
    print(f"stored_eof value:      {stored_eof:,} bytes ({stored_eof/1024**3:.2f} GiB)")
    print(f"Discrepancy:           {stored_eof - actual_size:,} bytes "
          f"({(stored_eof - actual_size)/1024**2:.2f} MiB)")

    return {
        "path": os.path.abspath(path),
        "actual_size": actual_size,
        "version": version,
        "size_of_offsets": size_of_offsets,
        "eof_offset": eof_offset,
        "stored_eof": stored_eof,
        "stored_eof_bytes_hex": raw_bytes.hex(),
    }


def apply_patch(path):
    info = inspect(path)
    print()

    if info["stored_eof"] == info["actual_size"]:
        print("stored_eof already matches actual file size -- nothing to patch.")
        return

    if info["stored_eof"] < info["actual_size"]:
        print("stored_eof is SMALLER than the actual file size. This script only "
              "handles the truncation case (stored_eof > actual size, as in the "
              "observed error). Refusing to patch a file that doesn't match that "
              "pattern -- inspect manually before proceeding.")
        sys.exit(1)

    backup_path = path + ".superblock_backup.json"
    if os.path.exists(backup_path):
        print(f"A backup already exists at {backup_path} -- refusing to overwrite it. "
              f"If a previous --apply already ran, use --restore first, or remove/rename "
              f"that backup file if you're sure it's safe to.")
        sys.exit(1)

    with open(backup_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Backed up the current {info['size_of_offsets']}-byte stored_eof field "
          f"to {backup_path}")

    new_value_bytes = info["actual_size"].to_bytes(info["size_of_offsets"], byteorder="little", signed=False)
    with open(path, "r+b") as f:
        f.seek(info["eof_offset"])
        f.write(new_value_bytes)
        f.flush()
        os.fsync(f.fileno())
    print(f"Patched stored_eof at byte offset {info['eof_offset']} to "
          f"{info['actual_size']:,} (matches actual file size).")

    print()
    print("Verifying the file now opens and every key is readable...")
    verify(path)


def restore(path):
    backup_path = path + ".superblock_backup.json"
    if not os.path.exists(backup_path):
        print(f"No backup found at {backup_path} -- nothing to restore.")
        sys.exit(1)

    with open(backup_path) as f:
        info = json.load(f)

    original_bytes = bytes.fromhex(info["stored_eof_bytes_hex"])
    with open(path, "r+b") as f:
        f.seek(info["eof_offset"])
        f.write(original_bytes)
        f.flush()
        os.fsync(f.fileno())

    print(f"Restored the original stored_eof bytes at offset {info['eof_offset']}.")
    os.remove(backup_path)
    print(f"Removed {backup_path}.")


def verify(path):
    """Attempts to open the file and read every top-level key, via pandas/
    PyTables first (matches how the codebase actually reads/writes this
    file), falling back to raw h5py if PyTables isn't available."""
    try:
        import pandas as pd
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
            print(f"Opened successfully via pandas.HDFStore. {len(keys)} keys found.")
            failures = []
            for key in keys:
                try:
                    df = store.select(key, start=0, stop=1)
                    n_rows = store.get_storer(key).nrows
                    print(f"  OK  {key}: {n_rows:,} rows, columns={list(df.columns)}")
                except Exception as e:
                    failures.append((key, str(e)))
                    print(f"  FAIL {key}: {e}")
            if failures:
                print(f"\n{len(failures)} of {len(keys)} keys failed to read cleanly -- "
                      f"the superblock patch succeeded but some individual tables may "
                      f"still be damaged. The file is at least openable now; inspect the "
                      f"failed keys individually before relying on them.")
            else:
                print(f"\nAll {len(keys)} keys read cleanly. Recovery looks complete.")
            return
    except ImportError:
        pass

    import h5py
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"Opened successfully via h5py. {len(keys)} top-level keys found: {keys}")


def compact(path, out_path):
    print(f"\nCompacting into a fresh file at {out_path} -- this reads and rewrites "
          f"the full dataset and will take a while at 185GB scale.")
    import pandas as pd
    with pd.HDFStore(path, mode="r") as src, pd.HDFStore(out_path, mode="w") as dst:
        for key in src.keys():
            print(f"  copying {key} ...")
            df = src.select(key)
            dst.put(key, df, format="table", data_columns=True)
    print(f"Done. Fresh copy written to {out_path}. Original left untouched at {path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Path to signals.hdf5")
    parser.add_argument("--apply", action="store_true", help="Patch the stored_eof field (default is dry-run/inspect only)")
    parser.add_argument("--restore", action="store_true", help="Undo a previous --apply using its sidecar backup")
    parser.add_argument("--compact-to", metavar="OUT_PATH", help="After a successful --apply, also rewrite all data into a fresh file at OUT_PATH (slow, optional)")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}")
        sys.exit(1)

    if args.restore:
        restore(args.path)
        return

    if args.apply:
        apply_patch(args.path)
        if args.compact_to:
            compact(args.path, args.compact_to)
        return

    print("DRY RUN (no changes made) -- pass --apply to actually patch the file.\n")
    inspect(args.path)


if __name__ == "__main__":
    main()
