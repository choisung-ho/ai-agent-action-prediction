"""Build the fixed-format submission ZIP from an explicit artifact directory.

The submission contract is deliberately small and stable.  Regardless of the
source directory, the resulting archive contains *only* these four POSIX-path
members:

    script.py
    requirements.txt
    model/best_model.pkl
    model/best_model_metadata.json

Use ``--source-dir`` for archived or staged candidates so an unrelated model
in the workspace root can never be packaged by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path


REQUIRED_MEMBERS = (
    "script.py",
    "requirements.txt",
    "model/best_model.pkl",
    "model/best_model_metadata.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed four-file submission ZIP from a verified artifact directory."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Verified directory containing exactly script.py, requirements.txt and model/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output ZIP path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after the new archive has passed all packaging checks.",
    )
    return parser.parse_args()


def resolve_required_files(source_dir: Path) -> dict[str, Path]:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")

    files = {member: source_dir / Path(member) for member in REQUIRED_MEMBERS}
    missing = [member for member, path in files.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Required submission artifacts are missing or empty: {missing}")

    actual_members = {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file()
    }
    expected_members = set(REQUIRED_MEMBERS)
    if actual_members != expected_members:
        raise RuntimeError(
            "Source directory must contain exactly the fixed four submission artifacts: "
            f"expected={sorted(expected_members)}, actual={sorted(actual_members)}"
        )
    return files


def verify_archive(output_path: Path) -> None:
    expected = set(REQUIRED_MEMBERS)
    with zipfile.ZipFile(output_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"CRC validation failed for ZIP member: {corrupt}")
        members = archive.namelist()

    if len(members) != len(set(members)):
        raise RuntimeError("Submission ZIP contains duplicate members.")
    if members != list(REQUIRED_MEMBERS):
        raise RuntimeError(
            "Submission ZIP member order is not exactly the fixed contract: "
            f"expected={list(REQUIRED_MEMBERS)}, actual={members}"
        )
    if set(members) != expected:
        raise RuntimeError("Submission ZIP member set differs from the fixed contract.")
    if any("\\" in member or member.startswith("/") or ".." in Path(member).parts for member in members):
        raise RuntimeError(f"Submission ZIP contains a non-portable member path: {members}")


def build_submission_zip(source_dir: Path, output_path: Path, *, overwrite: bool) -> None:
    source_dir = source_dir.resolve()
    output_path = output_path.resolve()
    files = resolve_required_files(source_dir)

    # The output file must never replace one of the four source artifacts.
    if output_path in {path.resolve() for path in files.values()}:
        raise ValueError("Output ZIP path overlaps a required source artifact.")
    if source_dir in output_path.parents:
        raise ValueError("Output ZIP must be outside the source directory.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists and was not touched: {output_path}. Use --overwrite after validation."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.", suffix=".tmp", dir=output_path.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
        # Match the established B42 submission container convention.  The
        # payload contract is what matters to the scorer, but keeping level-6
        # deflate and zero external attributes also makes a rebuild of the
        # archived B42 source byte-for-byte reproducible.
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for member in REQUIRED_MEMBERS:
                # ``member`` is intentionally a literal POSIX path, independent
                # of the operating system that creates the archive.
                archive.write(files[member], arcname=member)
                # ``ZipFile.write`` carries Windows file-mode bits into the
                # central directory.  The known-good archive uses no external
                # attributes, and this mutation is safe before close() writes
                # the central directory.
                archive.filelist[-1].external_attr = 0

        verify_archive(temp_path)
        temp_path.replace(output_path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    print("[PASS] fixed submission ZIP created")
    print(f"[INFO] source_dir={source_dir}")
    print(f"[INFO] output={output_path}")
    for member in REQUIRED_MEMBERS:
        path = files[member]
        print(f"[INFO] {member}: bytes={path.stat().st_size}, sha256={sha256(path)}")


def main() -> int:
    args = parse_args()
    try:
        build_submission_zip(Path(args.source_dir), Path(args.output), overwrite=bool(args.overwrite))
    except Exception as exc:  # Make packaging failures visible to automation.
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
