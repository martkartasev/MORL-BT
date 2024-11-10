import os
import shutil
import sys

def create_win_paths_dir(original_dir):
    # Create the "winPaths" directory
    # win_paths_dir = os.path.join(original_dir, "winPaths")
    win_paths_dir = f"{original_dir}_winPaths"
    os.makedirs(win_paths_dir, exist_ok=True)
    return win_paths_dir

def sanitize_path_for_windows(path):
    # Replace ":" with "-" to make it Windows-compatible
    return path.replace(":", "-")

def copy_files_to_win_paths(original_dir, win_paths_dir):
    # Walk through all files and subdirectories in the original directory
    for root, dirs, files in os.walk(original_dir):
        for file in files:
            print(root, file)
            # Construct the full path to the file
            full_file_path = os.path.join(root, file)
            # Generate the destination path inside winPaths
            relative_path = os.path.relpath(full_file_path, original_dir)
            sanitized_relative_path = sanitize_path_for_windows(relative_path)
            dest_file_path = os.path.join(win_paths_dir, sanitized_relative_path)

            # Ensure the target subdirectories exist
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            # Copy the file to the destination
            shutil.copy2(full_file_path, dest_file_path)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <absolute_path_to_dir>")
        sys.exit(1)

    original_dir = sys.argv[1]
    if not os.path.isabs(original_dir):
        print("Please provide an absolute path.")
        sys.exit(1)

    win_paths_dir = create_win_paths_dir(original_dir)
    copy_files_to_win_paths(original_dir, win_paths_dir)
    print(f"All files copied to {win_paths_dir} with Windows-compatible paths.")

if __name__ == "__main__":
    main()
