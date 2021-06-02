import pandas as pd
import numpy as np
import glob
import HP
from multiprocessing import Pool


def merge_assessment_score(df):
    new_df = pd.DataFrame()
    for note in note_list:
        #print(note)
        tmp = df.loc[df['note_id'] == note]
        #print(tmp.shape)
        score_list = tmp.score.unique()
        if 'No score' in score_list:
            if tmp.shape[0] < 2: 
                print("no assessment extracted")
            else:
            
                idx_list = tmp.loc[tmp['tag'] == 'INTERPRETATION'].index.values
        #print(idx_list)
                all_idx = tmp.index.values
                #print(all_idx)
                for idx in idx_list:
                    #print(idx)
                    if idx == all_idx[0]:
                        print('INTERPRETATION is before assessment')
                        for idx2 in all_idx[1:]:
                    #print(idx2)
                            if tmp['score'][idx2] == 'No score':
                                assess = tmp['assessment'][idx2]
                                score = tmp['score'][idx]
                                tmp.loc[tmp['assessment'] == assess, 'score'] = score
                    else:
                        if tmp['score'][idx-1] == 'No score':
                            assess = tmp['assessment'][idx-1]
                            score = tmp['score'][idx]
                            tmp.loc[tmp['assessment'] == assess, 'score'] = score
                        else:
                            print("no assessment extracted")
            
        else:
            print('Only INTERPRETATION tags. No empty assessment found')
        new_df = new_df.append(tmp)
    return new_df

def merge_assessment_score_v2(df):
    new_df = pd.DataFrame()
    for note in note_list:
        #print(note)
        tmp = df.loc[df['note_id'] == note]
        #print(tmp.shape)
        score_list = tmp.score.unique()
        if 'No score' in score_list:
            if tmp.shape[0] < 2:
                print("no assessment extracted")
            else:

                idx_list = tmp.loc[tmp['tag'] == 'INTERPRETATION'].index.values
        #print(idx_list)
                all_idx = tmp.index.values
                #print(all_idx)
             #   for idx in idx_list:
                    
                for idx in idx_list:
                    print('idx: ', idx)
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

    #path_data = '/share/pi/stamang/rise_pro/output/extracted_v4/'

    fnames = glob.glob(HP.output_path + '*.parquet')
    fnames.sort()

    full_df = pd.DataFrame()
    for f in fnames:
        df = pd.read_parquet(f)
        full_df = full_df.append(df)
        print(full_df.shape)

    full_df = full_df.rename(columns={'other_id':'practice_id'})
    full_df = full_df.reset_index(drop=True)

    note_list = full_df.loc[full_df['tag'] == 'INTERPRETATION'].note_id.unique()
    print(len(note_list))

    selected = full_df.loc[full_df['note_id'].isin(note_list)]
    selected = selected.reset_index(drop=True)
    print(selected.shape)

    print("Starting multithread score resolution:")
    selected_new = parallelize_dataframe(selected, merge_assessment_score_v2, n_cores=HP.threads)
    
    print("Done! Saving...")
    selected_new.to_parquet(HP.output_path + 'merged_scores.parquet')
