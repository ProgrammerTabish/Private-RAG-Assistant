"""
Set metadata on PDFs that are ALREADY in the blob container (no re-upload).

Usage (from D:\Private-RAG-Assistant):
  1) python set_blob_metadata.py init    -> creates documents\manifest.csv listing every blob
  2) fill in the columns in manifest.csv (Excel is fine, save as CSV UTF-8)
  3) python set_blob_metadata.py apply   -> writes the metadata to each blob
"""
import csv
import sys
import pathlib
import unicodedata

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# ---- adjust if your names differ ------------------------------------------
ACCOUNT = "strragdevswc01"
CONTAINER = "spg-compliance"
MANIFEST = pathlib.Path(r"D:\Private-RAG-Assistant\documents\manifest.csv")
# ---------------------------------------------------------------------------

COLUMNS = ["file", "regulator", "doc_type", "title", "publication_date", "language"]


def ascii_only(value: str) -> str:
    """Blob metadata must be ASCII: turn umlauts into ae/oe/ue/ss, drop other accents."""
    for src, dst in {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}.items():
        value = value.replace(src, dst)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return value.strip()


def container_client():
    svc = BlobServiceClient(f"https://{ACCOUNT}.blob.core.windows.net",
                            credential=DefaultAzureCredential())
    return svc.get_container_client(CONTAINER)


def init():
    if MANIFEST.exists():
        sys.exit(f"{MANIFEST} already exists - delete it first if you want a fresh one.")
    names = [b.name for b in container_client().list_blobs() if b.name.lower().endswith(".pdf")]
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for n in sorted(names):
            w.writerow([n, "", "", pathlib.Path(n).stem, "", ""])
    print(f"Wrote {len(names)} rows to {MANIFEST}. Fill in the columns, then run: apply")


def apply():
    cc = container_client()
    existing = {b.name for b in cc.list_blobs()}
    ok = missing = 0
    with MANIFEST.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row.pop("file").strip()
            if name not in existing:
                print(f"SKIP (not in container): {name}")
                missing += 1
                continue
            meta = {k: ascii_only(v) for k, v in row.items() if v and v.strip()}
            cc.get_blob_client(name).set_blob_metadata(meta)
            print(f"OK   {name}: {meta}")
            ok += 1
    print(f"\nDone: {ok} updated, {missing} skipped.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("init", "apply"):
        sys.exit("Usage: python set_blob_metadata.py init | apply")
    init() if sys.argv[1] == "init" else apply()
