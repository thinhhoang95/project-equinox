import shutil
import os

plots_dir = os.path.join("ronbun_experiments", "runs_LGAV_LFPG", "plots")
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir, ignore_errors=True)
    print(f"Deleted folder: {plots_dir}")
else:
    print(f"Folder does not exist: {plots_dir}")

runs_dir = os.path.join("ronbun_experiments", "runs_LGAV_LFPG")
if os.path.exists(runs_dir):
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file.endswith("LINK_LIKELIHOOD_WIND.pt"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")
else:
    print(f"Folder does not exist: {runs_dir}")
