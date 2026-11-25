import argparse
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from zipfile import ZipFile


COMMIT = "9a18e129f6f7db3018b509677ef426c67a616e0b"
UPSTREAM_CONFIG = Path(
    "upstream/mmsegmentation/local_config/MCANet/main/"
    "MCANet_mscan_t_isic2018_512x512_80k.py"
)


def main():
    parser = argparse.ArgumentParser(description="Unpack the pinned MCANet source archive.")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    if Path("upstream").exists():
        parser.error("upstream/ already exists. Use the existing verified checkout.")

    with ZipFile(args.archive) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        root = f"medical_seg-{COMMIT}"
        paths = []
        for entry in entries:
            parts = PurePosixPath(entry.filename).parts
            if parts[0] != root or ".." in parts or any("\\" in part or ":" in part for part in parts):
                parser.error(f"Unexpected entry in the pinned archive: {entry.filename}")
            # A literal asterisk in upstream config filenames is invalid on Windows.
            paths.append(Path("upstream", *(part.replace("*", "x") for part in parts[1:])))
        if UPSTREAM_CONFIG not in paths or len(paths) != len(set(paths)):
            parser.error("The archive is missing the tiny ISIC config or has conflicting filenames.")
        for entry, path in zip(entries, paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, path.open("xb") as destination:
                copyfileobj(source, destination)
    Path("upstream/UPSTREAM_COMMIT").write_text(COMMIT + "\n", encoding="utf-8")
    print(f"Upstream source installed at commit {COMMIT}.")


if __name__ == "__main__":
    main()
