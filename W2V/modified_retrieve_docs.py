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


# Function to get internal ID from docno
def get_internal_id(docno):
    command = [dumpindex_path, index_dir, "di", "docno", docno]
    result = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return int(result.strip())


# Function to retrieve a single document's vector and save to specified directory
def retrieve_and_save_doc(docno):
    global created_files_count

    output_filepath = os.path.join(output_dir, f"{docno}.txt")

    # Skip files that already exist
    if os.path.exists(output_filepath):
        with counter_lock:
            created_files_count += 1
            if created_files_count % 100 == 0:
                print(f"Processed {created_files_count}/{len(doc_ids)} documents")
        return

    internal_id = get_internal_id(docno)
    command = [dumpindex_path, index_dir, "dv", str(internal_id)]
    output = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')

    # Extract terms from the output
    lines = output.split("\n")
    terms_start = lines.index("--- Terms ---") + 1
    terms = [(int(line.split()[0]), int(line.split()[1]), line.split()[2]) for line in lines[terms_start:] if line]

    # Generate content for the file
    content = " ".join(term for _, freq, term in terms for _ in range(freq))

    # Save content to file
    with open(output_filepath, 'w') as file:
        file.write(content)

    with counter_lock:
        created_files_count += 1
        if created_files_count % 100 == 0:
            print(f"Processed {created_files_count}/{len(doc_ids)} documents")


# Multithreaded processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(retrieve_and_save_doc, doc_ids)
