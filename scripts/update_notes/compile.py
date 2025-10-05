import os
import subprocess
from tqdm import tqdm

TARGET_FILE_PATH = os.path.abspath("./LectureNote")


def get_all_notes():
    final_results = []
    for root, dirs, files in os.walk(TARGET_FILE_PATH, topdown=True):
        for file_name in files:
            if file_name.lower().endswith(".typ"):
                full_path = os.path.join(root, file_name)
                final_results.append(full_path)
    return final_results


def generate_output_path(input_path):
    output_dir = "./result"
    file_name = os.path.basename(input_path)
    base_name, extension = os.path.splitext(file_name)
    base_name = f"{base_name}.pdf"
    return os.path.join(output_dir, base_name)


def compile(input_files):
    length = len(input_files)
    for input_file in tqdm(input_files, total=length):
        output_path = generate_output_path(input_file)
        root_dir = "/home/xiyuanyang/Note/TypstNote/"
        command_list = [
            "make",
            f"TYPST_ROOT={root_dir}",
            f"SOURCE_FILE={input_file}",
            f"OUTPUT_FILE={output_path}",
        ]
        subprocess.run(
            command_list, capture_output=True, text=True, check=True, shell=False
        )


if __name__ == "__main__":
    compile(get_all_notes())
