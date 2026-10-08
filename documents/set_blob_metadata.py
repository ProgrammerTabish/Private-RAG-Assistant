r"""
Apply manifest.csv metadata to the PDFs already in blob storage, then verify.
Uses your Azure CLI login (az login). Run from D:\Private-RAG-Assistant\documents:

  python set_blob_metadata.py apply    -> writes metadata to all 50 blobs
  python set_blob_metadata.py verify   -> checks every blob against manifest.csv
"""
import csv, sys, pathlib, unicodedata
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient

ACCOUNT   = ""   # leave empty: found automatically via your az login
CONTAINER = ""   # leave empty: found automatically (the container holding the PDFs)
MANIFEST  = pathlib.Path(__file__).with_name("manifest.csv")
KEYS      = ["regulator", "doc_type", "title", "publication_date", "language"]

def ascii_only(v):
    for a, b in {"ä":"ae","ö":"oe","ü":"ue","Ä":"Ae","Ö":"Oe","Ü":"Ue","ß":"ss"}.items():
        v = v.replace(a, b)
    return unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode().strip()

def az(*args):
    import json, shutil, subprocess
    exe = shutil.which("az") or shutil.which("az.cmd")
    if not exe:
        sys.exit("Azure CLI (az) not found in PATH. Open a new PowerShell window after installing it.")
    out = subprocess.run([exe, *args, "-o", "json"], capture_output=True, text=True)
    if out.returncode:
        sys.exit("az failed: " + out.stderr.strip())
    return json.loads(out.stdout or "null")

def az_try(*args):
    import json, shutil, subprocess
    exe = shutil.which("az") or shutil.which("az.cmd")
    out = subprocess.run([exe, *args, "-o", "json"], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else None

def client():
    cred = AzureCliCredential()
    wanted = set(manifest())
    accounts = [ACCOUNT] if ACCOUNT else [a["name"] for a in az("storage", "account", "list", "--query", "[].{name:name}")]
    if not accounts:
        sys.exit("No storage accounts visible to your az login. Check: az account show")
    print("Storage accounts found:", ", ".join(accounts))
    for acc in accounts:
      for mode in ("login", "key"):
        if mode == "login":
            svc = BlobServiceClient(f"https://{acc}.blob.core.windows.net", credential=cred)
        else:
            keys = az_try("storage", "account", "keys", "list", "--account-name", acc)
            if not keys:
                print(f"  no account key available for {acc} (needs Contributor/Owner on the storage account)")
                break
            print(f"  retrying {acc} with the account key")
            svc = BlobServiceClient(f"https://{acc}.blob.core.windows.net", credential=keys[0]["value"])
        try:
            names = [CONTAINER] if CONTAINER else [c.name for c in svc.list_containers()]
            for cont in names:
                cc = svc.get_container_client(cont)
                hits = sum(1 for b in cc.list_blobs() if b.name in wanted)
                if hits:
                    print(f"Using {acc} / {cont} ({hits} of {len(wanted)} manifest PDFs found)\n")
                    return cc
            break
        except Exception as e:
            print(f"  {acc} via {mode}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            if "not authorized" not in str(e) and "AuthorizationPermissionMismatch" not in str(e):
                break
    sys.exit("Could not find a container with the manifest PDFs. See the 'skipped' lines above.")

def manifest():
    with MANIFEST.open(encoding="utf-8-sig") as f:
        return {r.pop("file").strip(): {k: ascii_only(r[k]) for k in KEYS if r.get(k, "").strip()} for r in csv.DictReader(f)}

def apply():
    cc, m = client(), manifest()
    existing = {b.name for b in cc.list_blobs()}
    ok = 0
    for name, meta in m.items():
        if name not in existing:
            print(f"SKIP not in container: {name}"); continue
        cc.get_blob_client(name).set_blob_metadata(meta)
        print(f"OK   {name}"); ok += 1
    print(f"\nApplied metadata to {ok} of {len(m)} blobs. Now run: python set_blob_metadata.py verify")

def verify():
    cc, m = client(), manifest()
    blobs = {b.name: (b.metadata or {}) for b in cc.list_blobs(include=["metadata"])}
    problems = 0
    for name, want in m.items():
        got = blobs.get(name)
        if got is None:
            print(f"MISSING blob: {name}"); problems += 1; continue
        diff = [k for k in want if got.get(k) != want[k]]
        if diff:
            print(f"MISMATCH {name}: {diff}"); problems += 1
    extra = [n for n in blobs if n.lower().endswith(".pdf") and n not in m]
    for n in extra: print(f"NOT IN MANIFEST: {n}")
    tagged = sum(1 for n in m if blobs.get(n) and all(k in blobs[n] for k in ("regulator","doc_type","title","language")))
    print(f"\nBlobs in container: {len(blobs)} | in manifest: {len(m)} | fully tagged: {tagged} | problems: {problems + len(extra)}")
    print("VERIFIED OK" if problems + len(extra) == 0 else "CHECK THE LINES ABOVE")

if __name__ == "__main__":
    {"apply": apply, "verify": verify}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit("Usage: python set_blob_metadata.py apply | verify"))()
