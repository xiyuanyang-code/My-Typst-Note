import os


def remove_tex_artifacts(root_dir):
    """Remove all .pdf and .synctex.gz files, excluding target directory"""
    removed_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the target directory
        if "target" in dirpath.split(os.sep):
            continue

        for filename in filenames:
            if filename.endswith(".pdf") or filename.endswith(".synctex.gz"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

    return removed_files


if __name__ == "__main__":
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Scanning directory: {root_directory}")
    removed = remove_tex_artifacts(root_directory)
    print(f"\nRemoved {len(removed)} files.")
