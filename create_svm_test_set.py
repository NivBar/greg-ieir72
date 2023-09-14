import os
import pandas as pd
from tqdm import tqdm

data_dict = {}
folder_path = '/lv_local/home/niv.b/content_modification_code-master/greg_output/calculated_features'
feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                    "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                    "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                    "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef",
                    "SimilarityToPredRef"]
feature_to_ind_dict = {k:v for v,k in enumerate(feature_list,1)}
ind_to_feature_dict = {v:k for k,v in feature_to_ind_dict.items()}

empty_files = []
for filename in tqdm(os.listdir(folder_path)):
    feature_name = filename.split('doc')[-1].split('_')[0]
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
    feature_name = file.split('doc')[-1].split('_')[0]
    qid, round_no = file.split("_")[-2:]
    rel_keys = [key for key in data_dict.keys() if f"ROUND-0{round_no[0]}-{qid}" in key]
    for key in rel_keys:
        data_dict[key][feature_name] = 0

df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.reset_index()
df['qid'] = df['index'].str.split('-').str.get(2)
df.sort_values(by="qid", ascending=True, inplace=True)

df.to_csv("./greg_output/feature_summary.csv", index=False)
with open("./greg_output/test.dat", 'w') as file:
    for idx, row in tqdm(df.iterrows()):
        str_ = f"0 qid:{row.qid} "
        for feature in feature_list:
            str_ += f"{feature_to_ind_dict[feature]}:{row[feature]} "
        file.writelines(str_ + '\n')