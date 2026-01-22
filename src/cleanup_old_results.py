
import os
import shutil
import time

results_dir = "/root/moe-icl/results"
current_time = time.time()
one_hour_ago = current_time - 3600

# 1. Delete old log files
old_logs = ["log_dense.txt", "log_moe4.txt", "log_moe8.txt"]
for log in old_logs:
    path = os.path.join(results_dir, log)
    if os.path.exists(path):
        print(f"Deleting old log file: {path}")
        os.remove(path)

# 2. Delete old experiment directories (UUIDs)
# We look into subdirectories like 'moe8_8noise', 'dense_4noise' etc.
for root, dirs, files in os.walk(results_dir):
    for d in dirs:
        dir_path = os.path.join(root, d)
        # Check if it's a UUID-like directory (simple check: length > 30)
        # Or just check modification time for safety
        if len(d) > 30 or d.startswith("run-"): 
            mtime = os.path.getmtime(dir_path)
            if mtime < one_hour_ago:
                print(f"Deleting old directory: {dir_path}")
                shutil.rmtree(dir_path)

print("Cleanup complete.")
