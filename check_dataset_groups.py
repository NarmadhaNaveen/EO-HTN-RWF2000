import os
import re
from collections import defaultdict

root = "RWF-2000"
splits = ["train", "val"]
classes = ["NonFight", "Fight"]

records = defaultdict(list)

for split in splits:
    for class_name in classes:
        folder = os.path.join(root, split, class_name)

        for filename in os.listdir(folder):
            if not filename.lower().endswith((".avi", ".mp4", ".mov")):
                continue

            stem = os.path.splitext(filename)[0]
            source_id = re.sub(r"_\d+$", "", stem)

            records[source_id].append({
                "split": split,
                "class": class_name,
                "filename": filename
            })

cross_split = []
repeated = []

for source_id, items in records.items():
    present_splits = {item["split"] for item in items}
    present_classes = {item["class"] for item in items}

    if len(present_splits) > 1:
        cross_split.append((source_id, items))

    if len(items) > 1:
        repeated.append((source_id, items))

print("\n========== CROSS-SPLIT SOURCE IDS ==========")
print("Number of source IDs appearing in train and val:", len(cross_split))

for source_id, items in cross_split[:30]:
    print("\n", source_id)
    for item in items:
        print(" ", item["split"], item["class"], item["filename"])

print("\n========== REPEATED SOURCE IDS ==========")
print("Number of source IDs with multiple clips:", len(repeated))

for source_id, items in repeated[:20]:
    print("\n", source_id)
    for item in items:
        print(" ", item["split"], item["class"], item["filename"])

print("\n========== CLASS CONFLICTS ==========")
conflicts = [
    (source_id, items)
    for source_id, items in records.items()
    if len({item["class"] for item in items}) > 1
]

print("Source IDs appearing under both class labels:", len(conflicts))

for source_id, items in conflicts[:20]:
    print("\n", source_id)
    for item in items:
        print(" ", item["split"], item["class"], item["filename"])
