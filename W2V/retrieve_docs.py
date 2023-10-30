import subprocess
import os
import concurrent.futures
import threading

# Paths and directories
index_dir = "/lv_local/home/niv.b/INDEX"
output_dir = "/lv_local/home/niv.b/content_modification_code-master/W2V/docFiles"
results_file = "/lv_local/home/niv.b/content_modification_code-master/W2V/results_new.txt"
dumpindex_path = "/lv_local/home/niv.b/indri/bin/dumpindex"

with open(results_file, 'r') as file:
    doc_ids = [line.split()[2] for line in file.readlines()]

counter_lock = threading.Lock()
created_files_count = 0
problematic_docs = 0

def retrieve_and_save_doc(doc_id):
    global created_files_count
    global problematic_docs

    output_filepath = os.path.join(output_dir, f"{doc_id}.txt")
    if not os.path.exists(output_filepath):
        command = f"{dumpindex_path} {index_dir} di docno {doc_id}"
        di_process = subprocess.run(command, capture_output=True, shell=True)

        # Extract the internal ID from the result, while gracefully handling any decoding errors
        di = di_process.stdout.decode('utf-8', errors='ignore').strip()

        # Now use this internal ID (di) to get the document text/content
        command2 = f"{dumpindex_path} {index_dir} dt {di}"
        result_process = subprocess.run(command2, capture_output=True, shell=True)

        # Decode the result output, while gracefully handling any decoding errors
        result_content = result_process.stdout.decode('utf-8', errors='ignore')

        if not result_content.strip():
            problematic_file_path = "/lv_local/home/niv.b/content_modification_code-master/W2V/problematic.txt"
            with open(problematic_file_path, 'a') as problematic_file:
                problematic_file.write(f"{doc_id}\n")
            with counter_lock:
                problematic_docs += 1

            x = 1
        else:
            with open(output_filepath, 'w') as file:
                file.write(result_content)
            x = 1

    with counter_lock:
        created_files_count += 1
        if created_files_count % 100 == 0:  # Print every 100 files for brevity
            print(f"files total : {created_files_count} , problematic docs : {problematic_docs}")

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:  # Adjust max_workers as needed
    list(executor.map(retrieve_and_save_doc, doc_ids))

print(f"END\n\nfiles total : {created_files_count} , problematic docs : {problematic_docs})")
