import os


with open("README.md", "r") as f:
    print("# --- path: README.md:")
    print(f.read())

example_dir = "example"
if os.path.isdir(example_dir):
    for root, dirs, files in os.walk(example_dir):
        if '.dcache' in root:
            continue
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, ".")
            print(f"\n# --- path: {rel_path}:")
            with open(fpath, "r") as f:
                print(f.read())

