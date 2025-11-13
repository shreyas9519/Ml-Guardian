# scripts/patch_feature_paths.py
import io, sys, pathlib

FILES = [
    pathlib.Path("core/train_xgb.py"),
    pathlib.Path("core/train_mlp_bert.py"),
    pathlib.Path("core/train_mlp.py"),  # optional; add any other training files you use
]

OLD = "data/processed/bert_features.parquet"
NEW = "data/processed/bert_claim_features.parquet"

def patch_file(p: pathlib.Path):
    if not p.exists():
        print(f"[skip] {p} does not exist")
        return
    text = p.read_text(encoding="utf-8")
    if OLD not in text:
        print(f"[noop] {p} does not reference {OLD}")
        return
    # backup
    bak = p.with_suffix(p.suffix + ".bak")
    bak.write_text(text, encoding="utf-8")
    new_text = text.replace(OLD, NEW)
    p.write_text(new_text, encoding="utf-8")
    print(f"[patched] {p} -> backup {bak.name}")

def main():
    for f in FILES:
        patch_file(f)

if __name__ == "__main__":
    main()
