import json

file_path = "data/raw/fever_data/train.jsonl"  # or the path where you have train.jsonl
label_counts = {}

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        label = obj.get("label", "UNKNOWN")
        label_counts[label] = label_counts.get(label, 0) + 1

print("Label distribution:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

total = sum(label_counts.values())
for label, count in label_counts.items():
    print(f"{label}: {count/total:.2%}")
