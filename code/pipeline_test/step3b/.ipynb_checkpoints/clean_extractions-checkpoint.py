import pandas as pd
import numpy as np
import HP
from multiprocessing import Pool

def clean_interpretation(df):

    idx_list = list(df.loc[df['tag'] == 'INTERPRETATION'].index)
    all_idx = list(df.index)
    print(idx_list)
    print(all_idx)
    tmp = df.copy()
    tmp = tmp.assign(interpretation=np.nan)
    for idx in idx_list:
        print(idx)
        if idx-1 not in all_idx:
            print('interpretation only')
            tmp['interpretation'][idx] = tmp['score'][idx]
            tmp['assessment'][idx] = 'No assessment'
        else:
            #print('idx: ', idx, 'idx-1: ', idx-1)
            #print(tmp['pos'][idx], tmp['pos'][idx-1])
            if tmp['pos'][idx] - tmp['pos'][idx-1] < 10:
                tmp['interpretation'][idx-1] = tmp['score'][idx]
                print('Dropping row ', idx)
                tmp = tmp.drop(idx)
                all_idx.remove(idx)

            else:
                tmp['interpretation'][idx] = tmp['score'][idx]
                tmp['assessment'][idx] = 'No assessment'

    return tmp

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df






if __name__ == '__main__':
    print('Loading the data...')

    df = pd.read_parquet(HP.extracted_clean)
    print(df.shape)

    print("Processing interpretation column...")
    cleaned_df = parallelize_dataframe(df, clean_interpretation, n_cores=HP.threads)
    print(cleaned_df.shape)

    print('Done! saving...')
    cleaned_df.to_parquet(HP.extracted_clean_formatted)
