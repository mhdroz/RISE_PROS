import pandas as pd
import numpy as np
import pickle
import re,sys
import HP
from utils import *
import spacy
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns

def get_shape(text, nlp):
    doc = nlp(text)
    for tok in doc:
        shape = tok.shape_
    return shape

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def get_shape_df(df):
    df['_shape'] = df['score'].apply(lambda x: get_shape(x, nlp))
    return df


def get_subset_statistics(df, assessment):
    print("Statistics for instrument: ", assessment)
    print("---------------------------------------------")
    print("Number of unique patients: ", df.patient_id.nunique())  #TODO: make it generic or require this column name
    print("  ")
    print("Number of unique practice: ", df.practice_id.nunique())
    print("---------------------------------------------")
    dates = df.date.unique()
    print("Date range:")
    print(min(dates))
    print(max(dates))
    print("---------------------------------------------")
    df = df.assign(score_num=np.nan)
    df = df.fillna(0)
    scores = df.score_updated_num.unique()
    print("Score range:")
    print(min(scores))
    print(max(scores))
    print("---------------------------------------------")
    print("Number of assessments: ", df.shape[0])


if __name__ == '__main__':
    print("Loading data for cleanup of RAPID3:")

    nlp = spacy.load(HP.spacy_model)

    df_clean = pd.read_parquet(HP.extracted_clean_formatted)

    df_clean['assessment_norm'] = df_clean['assessment'].apply(lambda x: x.lower())

    print("resolving scores for %s:" % HP.assessment)
    if HP.assessment == 'rapid3':
        pro_df = df_clean.loc[df_clean['assessment_norm'].isin(HP.rapid3_list)]
    elif HP.assessment == 'sdai':
        pro_df = df_clean.loc[df_clean['assessment_norm'] == 'sdai']
    elif HP.assessment == 'cdai':
        pro_df = df_clean.loc[df_clean['assessment_norm'] == 'cdai']
    elif HP.assessment == 'das28':
        pro_df = df_clean.loc[df_clean['assessment_norm'].isin(HP.das28_list)]
    elif HP.assessment == 'haq':
        pro_df = df_clean.loc[df_clean['assessment_norm'] == 'haq']
    elif HP.assessment == 'haqii':
        pro_df = df_clean.loc[df_clean['assessment_norm'] == 'haqii']
    elif HP.assessment == 'mdhaq':
        pro_df = df_clean.loc[df_clean['assessment_norm'] == 'mdhaq']
    elif HP.assessment == 'mhaq':
        pro_df = df_clean.loc[df_clean['assessment_norm'] == 'mhaq']
    else:
        print("%s score resolution not implemented yet" % HP.assessment)
        sys.exit()
    print(pro_df.shape)

    pro_scores = pro_df.loc[pro_df['score'] != 'No score']
    print(pro_scores.shape)

    pro_scores = parallelize_dataframe(pro_scores, get_shape_df, n_cores=HP.threads)

    pro_scores = pro_scores.assign(pro_range_nlp=np.nan)
    pro_scores = pro_scores.assign(score_updated=np.nan)
    if HP.assessment == 'rapid3':
        pro_scores = resolve_rapid3_scores(pro_scores)
    elif HP.assessment == 'sdai':
        pro_scores = resolve_sdai_scores(pro_scores)
    elif HP.assessment == 'cdai':
        pro_scores = resolve_cdai_scores(pro_scores)
    elif HP.assessment == 'das28':
        pro_scores = pro_scores
    elif HP.assessment == 'haq':
        pro_scores = resolve_haq_scores(pro_scores)
    elif HP.assessment == 'haqii':
        pro_scores = pro_scores
    elif HP.assessment == 'mdhaq':
        pro_scores = pro_scores
    elif HP.assessment == 'mhaq':
        pro_scores = pro_scores
    else:
        print("cleanup of %s score not supported")
        sys.exit()
    pro_scores.loc[pro_scores['score_updated'].isnull(), 'score_updated'] = pro_scores['score']
    pro_scores['score_updated_num'] = pd.to_numeric(pro_scores['score_updated'], errors='coerce')

    for row in pro_scores.iterrows():
        if HP.assessment == 'rapid3':
            if row[1]['score_updated_num'] > 30:
                pro_scores['score_updated_num'][row[0]] = np.nan
        if HP.assessment == 'sdai':
            if row[1]['score_updated_num'] > 86:
                pro_scores['score_updated_num'][row[0]] = np.nan
        if HP.assessment == 'cdai':
            if row[1]['score_updated_num'] > 76:
                pro_scores['score_updated_num'][row[0]] = np.nan
        if HP.assessment == 'das28':
            if row[1]['score_updated_num'] > 10:
                pro_scores['score_updated_num'][row[0]] = np.nan
        if HP.assessment == 'haq':
            if row[1]['score_updated_num'] > 3:
                pro_scores['score_updated_num'][row[0]] = np.nan
        if HP.assessment == 'haqii':
            if row[1]['score_updated_num'] > 3:
                pro_scores['score_updated_num'][row[0]] = np.nan
        if HP.assessment == 'mdhaq':
            if row[1]['score_updated_num'] > 3:
                pro_scores['score_updated_num'][row[0]] = np.nan
        if HP.assessment == 'mhaq':
            if row[1]['score_updated_num'] > 3:
                pro_scores['score_updated_num'][row[0]] = np.nan
        if row[1]['score_updated_num'] < 0:
            pro_scores['score_updated_num'][row[0]] = np.nan

    get_subset_statistics(pro_scores, HP.assessment)

    sns.set(rc={'figure.figsize':(16,10)})
    score_distribution_plot = sns.distplot(pro_scores['score_updated_num'], hist=False).set_title('%s NLP result values' % HP.assessment)
    fig = score_distribution_plot.get_figure()
    fig.savefig('%s_distribution_v3.pdf' % HP.assessment)

    pro_scores['score_updated'] = pro_scores['score_updated'].astype(str)
    pro_scores.to_parquet(HP.output_path + '%s_scores_processed.parquet' % HP.assessment)

