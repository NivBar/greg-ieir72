import os
import pandas as pd
from tqdm import tqdm
from utils import run_bash_command

pos = 5

data_dict = {}
folder_path = '/lv_local/home/niv.b/content_modification_code-master/greg_output/output_feature_files_dir'
feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef",
                "SimilarityToPredRef"]
feature_to_ind_dict = {k: v for v, k in enumerate(feature_list, 1)}
ind_to_feature_dict = {v: k for k, v in feature_to_ind_dict.items()}

empty_files = []
for filename in tqdm(os.listdir(folder_path)):
    feature_name = filename.split('doc_')[-1].split('_')[0]
    if feature_name not in feature_list:
        raise Exception(f"feature name {feature_name} not in feature list")

    with open(os.path.join(folder_path, filename), 'r') as file:
        filepath = os.path.join(folder_path, filename)
        # Check if the file is empty
        if os.path.getsize(filepath) == 0:
            empty_files.append(filename)
            continue

        for line in file:
            key, value = line.strip().split(' ')
            if key not in data_dict:
                data_dict[key] = {}
            data_dict[key][feature_name] = float(value)

for file in empty_files:
    feature_name = file.split('doc_')[-1].split('_')[0]
    qid, round_no = file.split("_")[-2:]
    rel_keys = [key for key in data_dict.keys() if f"ROUND-0{round_no[0]}-{qid}" in key]
    for key in rel_keys:
        data_dict[key][feature_name] = 0

df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.reset_index()
df['qid'] = df['index'].str.split('-').str.get(2)
df.sort_values(by="qid", ascending=True, inplace=True)

df.to_csv(f"./greg_output/feature_summary_{pos}.csv", index=False)
with open(f"./greg_output/test_{pos}.dat", 'w') as file:
    for idx, row in tqdm(df.iterrows()):
        str_ = f"0 qid:{row.qid} "
        for feature in feature_list:
            str_ += f"{feature_to_ind_dict[feature]}:{row[feature]} "
        file.writelines(str_ + '\n')

print("created test.dat file")

run_bash_command(f"/lv_local/home/niv.b/svm_rank/svm_rank_classify /lv_local/home/niv.b/content_modification_code-master/greg_output/test_{pos}.dat /lv_local/home/niv.b/content_modification_code-master/rank_models/harmonic_competition_model /lv_local/home/niv.b/content_modification_code-master/greg_output/predictions_{pos}.txt")

print("created predictions file")

df = pd.read_csv(f"./greg_output/feature_summary_{pos}.csv")[["index"]].rename(columns={"index":"ID"})
text_df = pd.DataFrame([line.strip().split(None, 2) for line in open('./greg_output/raw_ds_out.txt')], columns=['index_', 'ID', 'text'])


with open(f"./greg_output/predictions_{pos}.txt", 'r') as file:
    vals = [float(val.replace("\n","")) for val in file.readlines()]

df["score"] = vals
df["docno"] = df["ID"].str.split("_").str.get(0).str.split("$").str.get(0)
df['rank'] = df.groupby('docno')['score'].rank(method='first', ascending=False).astype(int)
df.sort_values(["docno", "rank"], inplace=True)
df[["_", "round_no", "query_id", "creator"]] = df["docno"].str.split("-", expand=True)
df_rank1 = df.query('rank == 1')
df_rank1 = df_rank1[df_rank1.round_no == '06'] #train data according to the articlechoos
final_df = df_rank1.merge(text_df, on="ID", how="left")
final_df["username"] = "BOT"
final_df = final_df[["round_no", "query_id", "creator", "username", "text"]]
final_df["round_no"] = final_df["round_no"].str.replace("0", "")
final_df.to_csv("./greg_output/bot_followup_asrc.csv", index = False)

print("created bot followup file")