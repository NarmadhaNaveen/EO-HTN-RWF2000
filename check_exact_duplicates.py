import os
import hashlib

root = "RWF-2000"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for split in ["train", "val"]:
    fight_dir = os.path.join(root, split, "Fight")
    normal_dir = os.path.join(root, split, "NonFight")

    fight_files = set(os.listdir(fight_dir))
    normal_files = set(os.listdir(normal_dir))

    common = sorted(
        f for f in fight_files.intersection(normal_files)
        if f.lower().endswith((".avi", ".mp4", ".mov"))
    )

    identical = []
    different = []

    print(f"\n========== {split.upper()} ==========")
    print("Identical filenames in both classes:", len(common))

    for filename in common:
        fight_path = os.path.join(fight_dir, filename)
        normal_path = os.path.join(normal_dir, filename)

        if os.path.getsize(fight_path) != os.path.getsize(normal_path):
            different.append(filename)
        elif sha256(fight_path) == sha256(normal_path):
            identical.append(filename)
        else:
            different.append(filename)

    print("Exact duplicate files:", len(identical))
    print("Different video files:", len(different))

    print("\nFirst 15 exact duplicates:")
    for name in identical[:15]:
        print(name)

    print("\nFirst 15 different files:")
    for name in different[:15]:
        print(name)

print("\nDuplicate audit completed.")
