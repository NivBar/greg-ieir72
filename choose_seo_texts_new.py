import glob
from collections import defaultdict
import pandas as pd
from utils import run_bash_command
from tqdm import tqdm
import os


def process_files(source_dir, target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over files in the source directory
    for filename in os.listdir(source_dir):
        if filename.startswith("features"):
            # Construct full file paths
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename + ".dat")

            # Read the source file and replace underscores
            lines = []
            with open(source_file, 'r') as file:
                content = file.readlines()
                for line in content:
                    line = line.replace("_", "", 1)
                    lines.append(line)

            # Write the modified content to the target file
            with open(target_file, 'w') as file:
                file.writelines(lines)


# Function to read the working set file
def read_working_set_file(working_set_file_path):
    init_df = pd.read_csv(working_set_file_path, delimiter=' ', header=None)
    init_df = init_df.rename({0: "qID", 2: "docno"}, axis=1)[["qID", "docno"]]
    init_docs = init_df.groupby("qID").docno.apply(list).to_dict()
    return init_docs


# Function to read feature files
def read_feature_files(features_dir, init_docs, feature_list, stream_list):
    features = defaultdict(lambda: defaultdict(dict))
    feature_id = {}
    no_stream = defaultdict(dict)
    f_id = 0
    feature_id_path = os.path.join(features_dir, 'featureID')
    lines = []

    for feature in tqdm(feature_list):
        for stream in stream_list:
            feature_name = f"{stream}{feature}"
            f_id += 1
            feature_id[feature_name] = f_id
            lines.append(f"{feature_name} {f_id}\n")

            for qID in init_docs:
                feature_file = os.path.join(features_dir, f"doc_{feature}_{qID}")
                if os.path.exists(feature_file):
                    with open(feature_file, 'r') as fi:
                        for line in fi:
                            dID, score = line.strip().split()
                            dID = dID.split("$")[-1]
                            if dID in init_docs[qID]:
                                features[feature_name][qID][dID] = float(score)
                else:
                    no_stream[feature_name][qID] = 1

    with open(feature_id_path, 'w') as fo:
        fo.writelines(lines)

    return features, feature_id, no_stream


# Function to normalize feature values
def normalize_features(features):
    min_max_values = {}
    for feature_name, queries in features.items():
        all_scores = [score for query in queries.values() for score in query.values()]
        min_max_values[feature_name] = (min(all_scores), max(all_scores))
    normalized_features = defaultdict(lambda: defaultdict(dict))
    for feature_name, queries in features.items():
        min_value, max_value = min_max_values[feature_name]
        range_value = max_value - min_value
        for qID, docs in queries.items():
            for dID, score in docs.items():
                if range_value == 0:
                    normalized_score = 0
                else:
                    normalized_score = (score - min_value) / range_value
                normalized_features[feature_name][qID][dID] = normalized_score
    return normalized_features


# Function to generate the output file
def generate_output_file_matching_perl(output_file_path, normalized_features, feature_id, init_docs):
    lines = []
    for qID in sorted(init_docs.keys(), key=int):
        for dID in init_docs[qID]:
            row = []
            qID_ = qID.replace("_", "")
            row.append(f"0 qid:{qID_}")
            for featureName in sorted(feature_id, key=lambda x: feature_id[x]):
                res = 0.0
                if (featureName in normalized_features and qID in normalized_features[featureName] and dID in
                        normalized_features[featureName][qID]):
                    res = normalized_features[featureName][qID][dID]
                res = f"{res:.8f}"
                row.append(f" {feature_id[featureName]}:{res}")
            row.append(f" # {dID}\n")

            line = ''.join(row)
            lines.append(''.join(row))

    with open(output_file_path, 'w') as file:
        file.writelines(lines)


def create_train_data():
    df_wl = pd.read_csv("/lv_local/home/niv.b/train_RankSVM/waterloo_scores_file.txt", delimiter='\t', header=None)
    rel_docs = df_wl[((df_wl[0].str.contains("ROUND-06")) |(df_wl[0].str.contains("ROUND-04"))) & (df_wl[1] >= 60)][0].tolist()
    df_qrels = pd.read_csv("/lv_local/home/niv.b/train_RankSVM/qrels_seo_bot.txt", delimiter=' ', header=None)
    df_qrels["ref"] = df_qrels[2].str.split("_", expand=True)[0]
    refs = df_qrels["ref"].unique().tolist()
    ref_docs = set(rel_docs).intersection(set(refs))

    rows = []
    counters = {2:0,3:0,4:0,5:0}
    for i in [2,3,4,5]:
        with open(f"/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/features_{i}.dat", 'r') as f:
            lines = f.readlines()
            for line in lines:
                ref_doc = line.split("#")[-1].split("_")[0].strip()
                if ref_doc in ref_docs:
                    feats,doc_ = line.split("#")
                    ref_,in_,out_ = doc_.split("_")
                    new_doc_ = "_".join([ref_,str(int(out_) + 1),str(int(in_) + 1)])
                    line = line.replace(doc_.strip(), new_doc_.strip())
                    rows.append(line)
                    counters[i] += 1
    with open(f"/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/features_train.dat", 'w') as f:
        f.writelines(rows)
    print("counters:", counters)

if __name__ == '__main__':

    features_dir = '/lv_local/home/niv.b/content_modification_code-master/greg_output/output_feature_files_dir'
    feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                    "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                    "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                    "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef",
                    "SimilarityToPredRef"]
    stream_list = ["test_"]
    # process_files('/lv_local/home/niv.b/content_modification_code-master/greg_output/output_final_feature_file_dir',
    #               '/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files')
    # create_train_data()

    file_to_nick = {"demotion_model": "D", "harmonic_model_2": "H2", "harmonic_model_1000": "H1E3",
                    "harmonic_model_0": "H0", "harmonic_model_100000": "H1E5"
        , "harmonic_model_1000000000": "H1E9", "harmonic_model_1": "H1", "harmonic_model_0.5": "H05",
                    "weighted_model_0.5": "W05", "weighted_model_0.7": "W07",
                    "weighted_model_0": "W0", "weighted_model_0.9": "W09", "weighted_model_0.6": "W06",
                    "weighted_model_1": "W1", "weighted_model_0.3": "W03",
                    "weighted_model_0.1": "W01", "weighted_model_0.4": "W04", "weighted_model_0.8": "W08",
                    "weighted_model_0.2": "W02"}

    nick_to_file = {v: k for k, v in file_to_nick.items()}

    

    for model in file_to_nick.keys():
        nick = file_to_nick[model]
        for pos in ["2", "3", "4", "5"]:
            print(f"\n\n########## model: {model}, pos: {pos} ##########\n\n")

            working_set_file_path = f'/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/ws_output_{pos}.txt'

            # output_file_path = f'/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/test_{pos}_new.dat'
            # init_docs = read_working_set_file(working_set_file_path)
            # features, feature_id, no_stream = read_feature_files(features_dir, init_docs, feature_list, stream_list)
            # normalized_features = normalize_features(features)
            # generate_output_file_matching_perl(output_file_path, normalized_features, feature_id, init_docs
            #
            # out = run_bash_command(f"/lv_local/home/niv.b/svm_rank/svm_rank_classify "
            #                        f"/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/test_{pos}_new.dat "
            #                        f"/lv_local/home/niv.b/content_modification_code-master/rank_models/harmonic_competition_model "
            #                        f"/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/predictions_{pos}.txt")

            command = f"/lv_local/home/niv.b/svm_rank/svm_rank_classify /lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/archive_test_google/features_{pos}.dat /lv_local/home/niv.b/content_modification_code-master/rank_models/{model} /lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/predictions_{nick}_{pos}.txt"
            out = run_bash_command(command)

            print(out)
            text_df = pd.DataFrame([line.strip().split(None, 2) for line in
                                    open(f'./greg_output/saved_result_files/raw_ds_out_{pos}_texts.txt')],
                                   columns=['index_', 'ID', 'text'])
            text_df[['ref', 'docid']] = text_df['ID'].str.split('$', n=1, expand=True)
            text_df["creator"] = text_df["ref"].str.split("-", expand=True)[3].astype(int)
            text_df["query_id"] = text_df["ref"].str.split("-", expand=True)[2].astype(int)

            df = pd.read_csv(working_set_file_path, delimiter=' ', header=None).sort_values([0,2])
            df["score"] = pd.read_csv(f"/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/predictions_{nick}_{pos}.txt", header=None)
            df['rank'] = df.groupby(0)['score'].rank(method='first', ascending=False).astype(int)
            df = df.rename(columns={0: 'qid', 2: 'docid'})[["qid", "docid", "score", "rank"]].sort_values(['qid','docid'])
            df_rank1 = df.query('rank == 1')
            df_rank1 = df_rank1[df_rank1.docid.str.contains("ROUND-07")]  # test data according to the article
            df_rank1["round_no"] = "07"
            final_df = pd.merge(df_rank1, text_df, on='docid', how='left')
            final_df["username"] = "BOT_" + nick
            final_df = final_df[["round_no", "query_id", "creator", "username", "text"]]
            final_df["round_no"] = final_df["round_no"].str.replace("0", "")
            final_df.to_csv(f"./greg_output/saved_result_files/bot_followup_asrc_{nick}_{pos}.csv", index=False)

            print("created bot followup file for position: ", pos, "\n")

        # # concat rounds 2-5
        df = pd.concat([pd.read_csv(file) for file in glob.glob("./greg_output/saved_result_files/bot_followup_asrc_*.csv")], ignore_index=True).sort_values(["round_no", "query_id", "creator"])


        df.to_csv("./greg_output/saved_result_files/bot_followup_asrc.csv", index=False)