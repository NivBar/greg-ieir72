import glob
from collections import defaultdict
import pandas as pd
from utils import run_bash_command
from tqdm import tqdm
import os
import utils
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    greg_df = pd.read_csv("/lv_local/home/niv.b/CharPDM/greg_data.csv")
    greg_df = greg_df[greg_df.round_no == 7]

    archive_dir = "archive_test_w2v"
    model_dir = "/lv_local/home/niv.b/train_RankSVM/feature_sets"

    features_dir = '/lv_local/home/niv.b/content_modification_code-master/greg_output/output_feature_files_dir'
    feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                    "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                    "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                    "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef",
                    "SimilarityToPredRef"]
    stream_list = ["test_"]

    file_to_nick = {k: k.split("_")[-1] for k in os.listdir(model_dir) if "model" in k}

    for model in file_to_nick.keys():
        print(f"\n\n########## model: {model} ##########\n\n")

        nick = file_to_nick[model]
        if "$" in model:
            print("ERROR! bot name contains $")

        for pos in tqdm(["2", "3", "4", "5"]):
            if os.path.exists(f"./greg_output/saved_result_files/bot_followup_asrc_{nick}_{pos}.csv"):
                continue


            working_set_file_path = f'/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/{archive_dir}/ws_output_{pos}.txt'

            features_file_path = f'/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/{archive_dir}/features_{pos}.dat'
            predictions_file_path = f'/lv_local/home/niv.b/content_modification_code-master/greg_output/saved_result_files/predictions_{nick}_{pos}.txt'

            command = f"/lv_local/home/niv.b/opt/java/jdk1.8.0/bin/java -jar /lv_local/home/niv.b/content_modification_code-master/scripts/RankLib.jar -load {model_dir}/{model} -rank {features_file_path} -score {predictions_file_path}"
            # command = f"/lv_local/home/niv.b/opt/java/jdk-21.0.1/bin/java -jar /lv_local/home/niv.b/content_modification_code-master/scripts/RankLib-2.18.jar -load /lv_local/home/niv.b/train_RankSVM/feature_sets/{model} -rank {features_file_path} -score {predictions_file_path}"

            out = run_bash_command(command)

            if "Exception" in str(out):
                utils.check_and_update_permissions("/lv_local/home/niv.b/opt/java/jdk1.8.0/bin/java")
                utils.check_and_update_permissions("/lv_local/home/niv.b/content_modification_code-master/scripts/RankLib.jar")
                utils.check_and_update_permissions(f"{model_dir}/{model}")
                utils.check_and_update_permissions(features_file_path)
                out = run_bash_command(command)
                if "Exception" in str(out):
                    raise Exception(str(out))
                x = 1

            print(out)
            text_df = pd.DataFrame([line.strip().split(None, 2) for line in
                                    open(f'./greg_output/saved_result_files/{archive_dir}/raw_ds_out_{pos}_texts.txt')],
                                   columns=['index_', 'ID', 'text'])
            text_df[['ref', 'docid']] = text_df['ID'].str.split('$', n=1, expand=True)
            text_df["creator"] = text_df["ref"].str.split("-", expand=True)[3].astype(int)
            text_df["query_id"] = text_df["ref"].str.split("-", expand=True)[2].astype(int)

            df = pd.read_csv(working_set_file_path, delimiter=' ', header=None).sort_values([0, 2])
            score_column = pd.read_csv(predictions_file_path, header=None, delimiter='\t', usecols=[2])

            # text_df['text_len'] = text_df['text'].str.split().str.len()

            if int(score_column.isna().sum()) > 0:
                print(f"ERROR in {model}_{pos}: score column contains NaN values")
                continue

            df["score"] = score_column
            df['rank'] = df.groupby(0)['score'].rank(method='first', ascending=False).astype(int)
            df = df.rename(columns={0: 'qid', 2: 'docid'})[["qid", "docid", "score", "rank"]].sort_values(
                ['qid', 'docid'])
            df_rank1 = df.query('rank == 1')
            df_rank1 = df_rank1[df_rank1.docid.str.contains("ROUND-07")]  # test data according to the article
            df_rank1["round_no"] = "07"

            final_df = pd.merge(df_rank1, text_df, on='docid', how='left')
            final_df["username"] = "BOT_" + nick
            final_df = final_df[["round_no", "query_id", "creator", "username", "text"]]
            final_df["round_no"] = final_df["round_no"].str.replace("0", "")

            df_wl = pd.read_csv("/lv_local/home/niv.b/train_RankSVM/waterloo_scores_file.txt", delimiter='\t',
                                header=None).rename({0: "docno", 1: "wl_score"}, axis=1)
            df_wl = df_wl[df_wl.wl_score >= 60]
            df_wl[['round_no', 'query_id', 'creator']] = df_wl['docno'].str.split('-', expand=True)[[1, 2, 3]].astype(
                int)
            df_wl['query_id'] = df_wl['query_id'].astype(final_df.query_id.dtype)
            df_wl['creator'] = df_wl['creator'].astype(final_df.creator.dtype)
            df_wl = df_wl[df_wl.round_no == 7]

            filtered_df = final_df.merge(df_wl, on=["query_id", "creator"], how="inner", suffixes=(None, '_y'))
            filtered_df = filtered_df[["round_no", "query_id", "creator", "username", "text"]]
            filtered_df['text_len'] = filtered_df['text'].str.split().str.len()

            print(f"csv file for MODEL: {nick}, POS: {pos} was created successfully. length: {len(filtered_df)}/31")
            filtered_df.to_csv(f"./greg_output/saved_result_files/bot_followup_asrc_{nick}_{pos}.csv", index=False)
            # print("created bot followup file for position: ", pos, "\n")

    df = pd.concat(
        [pd.read_csv(file) for file in glob.glob("./greg_output/saved_result_files/bot_followup_asrc_*.csv")],
        ignore_index=True).sort_values(["round_no", "query_id", "creator"])

    df.to_csv("./greg_output/saved_result_files/bot_followup_asrc.csv", index=False)
    print("\n\n########## created bot followup file for all models and positions ##########\n\n")
