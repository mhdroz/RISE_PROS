import pandas as pd
import numpy as np
import pickle
import HP
import time
import glob
import spacy
from spacy.matcher import PhraseMatcher
import sys, time, re
from multiprocessing import Pool
#from bs4 import BeautifulSoup
from utils import *


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def extract_multithread(df):
    extracted_df = PRO_scores_extraction_v4(df, nlp, matcher, window=10)
    return extracted_df










if __name__ == '__main__':
    print("Generating the matcher:")
    
    nlp = spacy.load('en_core_sci_md')

    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

    terms_da = pd.read_csv(HP.terms_da, sep='|', header=None)
    terms_da.columns = ['idx', 'term', 'tag']
    terms_da = terms_da.loc[terms_da['tag'] == 'DA']
    da_list = terms_da.term.to_list()

    for term in da_list:
        matcher.add('DA', None, nlp(str(term)))

    terms_fs = pd.read_csv(HP.terms_fs, sep='|', header=None)
    terms_fs.columns = ['idx', 'term', 'tag']
    terms_fs = terms_fs.loc[terms_fs['tag'] == 'FS']
    fs_list = terms_fs.term.to_list()

    for term in fs_list:
        matcher.add('FS', None, nlp(str(term)))

    interpretation = ['remission', 'low activity', 'moderate activity', 'high activity',
            'high severity', 'low severity', 'moderate severity', 'low disease activity', 
            'moderate disease activity', 'high disease activity']

    for term in interpretation:
        matcher.add('INTERPRETATION', None, nlp(term)) 

    print("Loading raw notes file:")
    df = pd.read_parquet(HP.input_notes)
    print(df.shape)

#print(df.shape)
    df = df.loc[(df['date'] > HP.start_date) & (df['date'] < HP.end_date)]
    print(df.shape)
#df.head()

    #df['clean'] = df['note'].apply(lambda x: clean_html(x))
    #df['clean'] = df['clean'].apply(lambda x: clean_text(x))
    df['clean'] = df['note'].apply(lambda x: clean_text(x))
    df['length_clean'] = df['clean'].apply(lambda x: len(x))
#df.head()
    #Split the df to identify where is the bug! (set2-batch002)
    #interval = int(df.shape[0]/10)
    #for i in range(0,11):
    #    print('Processing subset ', i)
    #    df_sub = df.iloc[i*interval:(i+1)*interval].reset_index()
    #    extracted_df = parallelize_dataframe(df_sub, extract_multithread, n_cores=HP.threads)
    #    extracted_df.to_parquet('/share/pi/stamang/rise_pro/output/extracted/set2_batch002_sub%d_extracted.parquet' % i)

    #For when things run smoothly.... 
    start = time.time()
    extracted_df = parallelize_dataframe(df, extract_multithread, n_cores=HP.threads)
    print("Done extrating PROs in time {}s".format(time.time() - start))
    extracted_df.to_parquet(HP.extracted_pros)
