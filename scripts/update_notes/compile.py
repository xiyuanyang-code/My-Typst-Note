import os
import subprocess
from tqdm import tqdm

TARGET_FILE_PATH = os.path.abspath("./LectureNote")
OUTPUT_DIR = "./result"  # Define the output directory once


def get_all_notes():
    """
    Recursively finds all Typst source files (.typ) in the TARGET_FILE_PATH.
    """
    final_results = []
    for root, dirs, files in os.walk(TARGET_FILE_PATH, topdown=True):
        for file_name in files:
            if file_name.lower().endswith(".typ"):
                full_path = os.path.join(root, file_name)
                final_results.append(full_path)
    return final_results


def generate_output_path(input_path):
    """
    Generates the expected PDF output path from a Typst source file path.
    """
    # Use the defined OUTPUT_DIR
    file_name = os.path.basename(input_path)
    base_name, _ = os.path.splitext(file_name)
    base_name = f"{base_name}.pdf"
    return os.path.join(OUTPUT_DIR, base_name)


def generate_source_path(output_path):
    """
    Generates the expected Typst source path from a PDF output file path.
    This assumes a flat structure for source files relative to TARGET_FILE_PATH
    if the original Typst file was found in the top level. Since get_all_notes
    scans recursively, we need to search the entire source tree for a match.
    """
    pdf_name = os.path.basename(output_path)
    typ_name = os.path.splitext(pdf_name)[0] + ".typ"

    # Search for the source file recursively
    for root, _, files in os.walk(TARGET_FILE_PATH):
        if typ_name in files:
            return os.path.join(root, typ_name)

    # Return None if the file isn't found
    return None


def cleanup_orphaned_pdfs():
    """
    Scans the output directory and deletes any PDF files for which
    the corresponding Typst source file no longer exists.
    """
    print(f"Scanning for orphaned PDFs in '{OUTPUT_DIR}'...")

    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory '{OUTPUT_DIR}' does not exist. Skipping cleanup.")
        return

    # List all files in the result directory
    output_files = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    orphaned_count = 0

    for output_pdf_path in tqdm(output_files, desc="Checking PDFs"):
        # We need to check if the corresponding .typ file exists anywhere
        # under TARGET_FILE_PATH.
        source_typ_path = generate_source_path(output_pdf_path)

        # Check if the generated source path is None, meaning the file wasn't found
        if source_typ_path is None:
            # The source file is missing, so this PDF is orphaned.
            print(f"-> Deleting orphaned PDF: {output_pdf_path}")
            try:
                os.remove(output_pdf_path)
                orphaned_count += 1
            except OSError as e:
                print(f"Error deleting file {output_pdf_path}: {e}")

    print(f"Cleanup complete. Deleted {orphaned_count} orphaned PDF files.")


def compile(input_files):
    """
    Compiles the list of Typst input files into PDFs using the 'make' command.
    """
    # Ensure the output directory exists before compiling
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    length = len(input_files)

    for input_file in tqdm(input_files, total=length, desc="Compiling"):
        output_path = generate_output_path(input_file)
        root_dir = "/Users/xiyuanyang/Desktop/Dev/Note/TypstNote"
        command_list = [
            "make",
            f"TYPST_ROOT={root_dir}",
            f"SOURCE_FILE={input_file}",
            f"OUTPUT_FILE={output_path}",
        ]

        try:
            subprocess.run(
                command_list, capture_output=True, text=True, check=True, shell=False
            )
        except subprocess.CalledProcessError as e:
            print(f"\nError compiling {input_file}:")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            # Continue to the next file even if one fails
            continue


if __name__ == "__main__":
    cleanup_orphaned_pdfs()
    compile(input_files=get_all_notes())
