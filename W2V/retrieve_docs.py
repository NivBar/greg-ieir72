import subprocess
import os
import concurrent.futures
import threading

# Paths and directories
index_dir = "/lv_local/home/niv.b/cluewebindex"
output_dir = "/lv_local/home/niv.b/content_modification_code-master/W2V/docFiles"
results_file = "/lv_local/home/niv.b/content_modification_code-master/W2V/results.txt"
dumpindex_path = "/lv_local/home/niv.b/indri/bin/dumpindex"

# Load document IDs from results.txt
with open(results_file, 'r') as file:
    doc_ids = [line.split()[2] for line in file.readlines()]

# Counter for progress tracking
counter_lock = threading.Lock()
created_files_count = 0

# Function to retrieve a single document's content using dumpindex and save to specified directory
def retrieve_and_save_doc(doc_id):
    global created_files_count

    output_filepath = os.path.join(output_dir, f"{doc_id}.txt")
    # Check if file already exists
    if not os.path.exists(output_filepath):
        # Command to retrieve document content using dumpindex
        command = [dumpindex_path, index_dir, "dt", f"docno={doc_id}"]
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Save the retrieved content to a file
        with open(output_filepath, 'w') as file:
            file.write(result.stdout)

        # Update and print progress
        with counter_lock:
            created_files_count += 1
            if created_files_count % 100 == 0:  # Print every 100 files for brevity
                print(f"{created_files_count} files created so far...")

# Using a ThreadPoolExecutor to retrieve documents in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
    list(executor.map(retrieve_and_save_doc, doc_ids))

print(f"Document retrieval and saving completed. Total {created_files_count} files were created.")
