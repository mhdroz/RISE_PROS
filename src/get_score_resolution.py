import pandas as pd
import numpy as np
import glob
import HP
from multiprocessing import Pool


def merge_assessment_score(df):
    new_df = pd.DataFrame()
    for note in note_list:
        tmp = df.loc[df['note_id'] == note]
        score_list = tmp.score.unique()
        if 'No score' in score_list:
            if tmp.shape[0] < 2:
                print("no assessment extracted")
            else:
                idx_list = tmp.loc[tmp['tag'] == 'INTERPRETATION'].index.values
                all_idx = tmp.index.values

                for idx in idx_list:
                    if idx == all_idx[0]:
                        print('INTERPRETATION is before assessment')
                    if idx == all_idx[-1]:
                        print('INTERPRETATION is the last extraction for this note')
                    else:    
                        if tmp['score'][idx+1] == 'No score':
                            print('No score at idx: ', idx+1)
                            assess = tmp['assessment'][idx+1]
                            score = tmp['score'][idx]
                            tmp['score'][idx+1] = score                       
        else:
            print('Only INTERPRETATION tags. No empty assessment found')
        new_df = new_df.append(tmp)
    return new_df

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


if __name__ == '__main__': 

    print("Loading the data...")

  #  fnames = glob.glob(HP.output_path + 'set_proto_batch000_extracted.parquet') #TODO make name more general for github
  #  fnames.sort()

  #  full_df = pd.DataFrame()
  #  for f in fnames:
  #      df = pd.read_parquet(f)
  #      full_df = full_df.append(df)

    full_df = pd.read_parquet(HP.extracted_pros)
    print(full_df.shape)

    full_df = full_df.rename(columns={'other_id':'practice_id'})
    full_df = full_df.reset_index(drop=True)

    note_list = full_df.loc[full_df['tag'] == 'INTERPRETATION'].note_id.unique()

    selected = full_df.loc[full_df['note_id'].isin(note_list)]
    selected = selected.reset_index(drop=True)

    print("Starting multithread score resolution:")
    selected_new = parallelize_dataframe(selected, merge_assessment_score, n_cores=HP.threads)
    
    print("Done! Saving...")
    selected_new.to_parquet(HP.extracted_clean) #TODO put name in HP 
