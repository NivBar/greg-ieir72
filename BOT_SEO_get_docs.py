import pandas as pd

from utils import read_trec_file, parse_with_regex


def trec_to_dataframe(ranking_fil, trectext_file):
    df = pd.read_csv(ranking_fil, sep=' ', names=['QueryId', 'Q0', 'docno', 'Rank', 'Score', 'RunName'])
    df[['round_no', 'query_id', 'username_full', 'creator']] = df['docno'].str.split('-', n=3, expand=True)
    df[['username', 'out_', 'in_']] = df['username_full'].str.rsplit('_', n=2, expand=True)
    docs_df = pd.DataFrame(list(parse_with_regex(open(trectext_file, 'r').read()).items()), columns=['docno', 'text'])
    df = df.merge(docs_df, how='left', on='docno')
    return df


def get_tops_df():
    trec_file = "./data/original_data_files/documents.positions"  # ranked document list
    trectext_file = "./data/original_data_files/documents.trectext"  # documents trectext file
    ranked_lists = read_trec_file(trec_file)
    doc_texts = parse_with_regex(open(trectext_file, 'r').read())
    tops = []
    for round_no in ranked_lists:
        for query_id in ranked_lists[round_no]:
            tops.append({
                "round_no": round_no,
                "query_id": query_id,
                "docno": ranked_lists[round_no][query_id][0],
                "text": doc_texts[ranked_lists[round_no][query_id][0]]
            })
    tops_df = pd.DataFrame(tops)
    tops_df['creator'] = tops_df['docno'].str.split('-', n=3, expand=True).iloc[:, -1]
    return tops_df.drop("docno", axis=1)


if __name__ == '__main__':
    ranking_file = "/lv_local/home/niv.b/content_modification_code-master/Results/RankedLists/LambdaMARTasrc"
    trectext_file = "./trecs/bot_followup_asrc.trectext"
    df = trec_to_dataframe(ranking_file, trectext_file)
    df = df[['Rank', 'round_no', 'query_id', 'creator', 'username', 'text']]
    min_rank_rows = df.loc[df.groupby(['round_no', 'query_id', 'creator'])['Rank'].idxmin()][
        ['round_no', 'query_id', 'creator', 'username', 'text']]
    tops_df = get_tops_df()
    final_df = pd.concat([min_rank_rows, tops_df])
    final_df.query_id = final_df.query_id.astype(int)
    final_df.round_no = final_df.round_no.astype(int) + 1
    final_df = final_df[final_df.round_no != max(final_df.round_no)]
    final_df = final_df.sort_values(['round_no', 'query_id', 'creator']).reset_index(drop=True)
    final_df.username = 'BOT'
    final_df.to_csv("../CharPDM/bot_followup_asrc.csv", index=False)
