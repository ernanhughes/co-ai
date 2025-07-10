import os

def annotate_python_files(root_dir="stephanie"):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, root_dir)
                full_comment = f"# {os.path.join(root_dir, rel_path)}\n"

                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Check if already annotated
                if lines and lines[0].strip() == full_comment.strip():
                    continue

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(full_comment)
                    f.writelines(lines)

if __name__ == "__main__":
    annotate_python_files("stephanie")
