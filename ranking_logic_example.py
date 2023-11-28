import os
import subprocess
import time
import logging.handlers

# import multiprocessing


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Function '{func.__name__}' took:")
        print(f"Seconds: {elapsed_time:.2f}")
        print(f"Minutes: {elapsed_time / 60:.2f}")
        print(f"Hours: {elapsed_time / 3600:.2f}")

        return result

    return wrapper


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')


def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)

    out, err = p.communicate()
    return out


@measure_time
def build_index(filename, currentTime, baseDir):
    """
    Parse the trectext file given, and create an index.
    """
    pathToFolder = baseDir + 'Collections/IndriIndices/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)

    INDRI_BUILD_INDEX = '/lv_local/home/niv.b/indri/bin/IndriBuildIndex'
    CORPUS_PATH = filename
    CORPUS_CLASS = 'trectext'
    MEMORY = '1G'
    INDEX = pathToFolder + currentTime
    STEMMER = 'krovetz'
    if os.path.exists(INDEX):
        run_bash_command('rm -r ' + INDEX)
    command = INDRI_BUILD_INDEX + ' -corpus.path=' + CORPUS_PATH + ' -corpus.class=' + CORPUS_CLASS + ' -index=' + INDEX + ' -memory=' + MEMORY + ' -stemmer.name=' + STEMMER
    out = run_bash_command(command)
    return INDEX


@measure_time
def merge_indices(asrIndex, baseDir, currentTime):
    """
    Merge indices of ASR and ClueWeb09. If MergedIndx exists, it will be deleted.
    """

    INDRI_DUMP_INDEX = '/lv_local/home/niv.b/indri/bin/dumpindex'
    CLUEWEB = f'/lv_local/home/niv.b/cluewebindex'
    pathToFolder = baseDir + 'Collections/'
    MERGED_INDEX = pathToFolder + f'/mergedindex_{currentTime}'
    run_bash_command('rm -r ' + MERGED_INDEX)
    command = INDRI_DUMP_INDEX + ' ' + MERGED_INDEX + ' merge ' + CLUEWEB + ' ' + asrIndex
    out = run_bash_command(command)
    return MERGED_INDEX


@measure_time
def run_ranking_model(mergedIndex, workingSet, currentTime, baseDir):
    """
    workingSet - a file in trec format that dictates which population to work on
    format is: <qid> Q0 <docid> <rank> <score> <experiment name>\n - rank and score arguments can be filled
    arbitrarily they are simply for the desired format

    """
    pathToFolder = baseDir + 'Results/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
    INDEX = mergedIndex
    WORKING_SET_FILE = workingSet
    # MODEL_DIR = baseDir+"Code/Models/"
    # MODEL_FILE = MODEL_DIR+"model"
    MODEL_FILE = '/lv_local/home/niv.b/content_modification_code-master/rank_models/model_lambdatamart'
    # QUERIES_FILE = '/lv_local/home/niv.b/content_modification_code-master/data/queries_bot_modified_sorted_1.xml'
    QUERIES_FILE = f'/lv_local/home/niv.b/content_modification_code-master/data/query_files/queries_{current_time}.xml'

    FEATURES_DIR = pathToFolder + '/Features/' + currentTime
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    FEATURES_FILE = 'features'
    command = baseDir + 'scripts/LTRFeatures ' + QUERIES_FILE + ' -stream=doc -index=' + INDEX + ' -repository=' + INDEX + ' -useWorkingSet=true -workingSetFile=' + WORKING_SET_FILE + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    out = run_command('mv doc*_* ' + FEATURES_DIR)
    command = 'perl ' + baseDir + 'scripts/generate.pl ' + FEATURES_DIR + ' ' + WORKING_SET_FILE
    print(command)
    out = run_bash_command(command)
    print(out)
    command = '/lv_local/home/niv.b/opt/java/jdk1.8.0/bin/java -jar ' + baseDir + 'scripts/RankLib.jar -load ' + MODEL_FILE + ' -rank ' + FEATURES_FILE + ' -score predictions.tmp'
    print(command)
    out = run_bash_command(command)
    print(out)
    command = 'cut -f3 predictions.tmp > predictions'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command('rm predictions.tmp')
    RANKED_LIST_DIR = pathToFolder + 'RankedLists/'
    if not os.path.exists(RANKED_LIST_DIR):
        os.makedirs(RANKED_LIST_DIR)
    PREDICTIONS_FILE = 'predictions'
    command = 'perl ' + baseDir + '/scripts/order.pl ' + RANKED_LIST_DIR + '/LambdaMART' + currentTime + ' ' + FEATURES_FILE + ' ' + PREDICTIONS_FILE
    print(command)
    out = run_bash_command(command)
    print(out)
    return RANKED_LIST_DIR + '/LambdaMART' + currentTime


# if __name__ == '__main__':
#     workingSet = '/lv_local/home/niv.b/content_modification_code-master/trecs/working_set.trectext'
#     baseDir = '/lv_local/home/niv.b/content_modification_code-master/'
#     currentTime = "1"
#     documents = '/lv_local/home/niv.b/content_modification_code-master/trecs/bot_followup.trectext'
#
#     asrIndex = build_index(documents, currentTime, baseDir)
#     print("build_index done...")
#     mergedIndex = merge_indices(asrIndex, baseDir)
#     # mergedIndex = '/lv_local/home/niv.b/content_modification_code-master/Collections/mergedindex'
#     print("merge_indices done...")
#     res = run_ranking_model(mergedIndex, workingSet, currentTime, baseDir)
#     print("run_ranking_model done...")
#     print(res)
#
#     # res = run_ranking_model('/lv_local/home/niv.b/cluewebindex', workingSet, currentTime, baseDir)
#     x = 1

def main_task(currentTime):
    logger.info("Starting...")
    baseDir = '/lv_local/home/niv.b/content_modification_code-master/'
    documents = f'/lv_local/home/niv.b/content_modification_code-master/trecs/bot_followup_{currentTime}.trectext'
    workingSet = f'/lv_local/home/niv.b/content_modification_code-master/trecs/working_set_{currentTime}.trectext'

    # asrIndex = build_index(documents, currentTime, baseDir)
    # print("build_index done...")
    # logger.info("build_index done...")
    #
    # mergedIndex = merge_indices(asrIndex, baseDir, currentTime)
    # print("merge_indices done...")
    # logger.info("merge_indices done...")

    mergedIndex = baseDir + 'Collections/' + f'/mergedindex_{currentTime}' # in case of running only the ranking model
    res = run_ranking_model(mergedIndex, workingSet, currentTime, baseDir)
    print("run_ranking_model done...")
    logger.info("run_ranking_model done...")

    print(res)
    logger.info(f'{res}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    logger = logging.getLogger('ranking_logger')
    logger.setLevel(logging.DEBUG)
    # num_cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=num_cores)
    # current_times = ["g1"]
    # print(f"running {current_times[0]} model...")
    # pool.map(main_task, current_times)
    # pool.close()
    # pool.join()
    current_time = "asrcfull2"
    print(f'Starting version {current_time}...')
    main_task(current_time)
