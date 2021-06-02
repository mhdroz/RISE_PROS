import pandas as pd
import numpy as np
import re




def resolve_rapid3_scores(df):
    #for row in df.loc[df['shape'] == shape].iterrows():
    for row in df.iterrows():
        proto = row[1]['score']
        if row[1]['_shape'] == 'd/dd':
            if '/10' in proto:
            #print('scale 10')
            #print(len(proto))
            #print(proto[0])
            #print(row[0])
                df['pro_range_nlp'][row[0]] = '0-10'
                df['score_updated'][row[0]] = proto[0]
            elif '/30' in proto:
            #print('scale 30')
            #print(len(proto))
            #print(proto[0])
            #print(row[0])
        #print(row[1]['score'])
                df['pro_range_nlp'][row[0]] = '0-30'
                df['score_updated'][row[0]] = proto[0]
        #print('-------------------')
        elif row[1]['_shape'] == 'd.d/dd':
            if '/10' in proto:
                df['pro_range_nlp'][row[0]] = '0-10'
                df['score_updated'][row[0]] = proto[0:3]
            elif '/30' in proto:
                df['pro_range_nlp'][row[0]] = '0-30'
                df['score_updated'][row[0]] = proto[0:3]

        elif row[1]['_shape'] == 'dd/dd':
            if '/10' in proto:
                df['pro_range_nlp'][row[0]] = '0-10'
                df['score_updated'][row[0]] = proto[0:2]
            elif '/30' in proto:
                df['pro_range_nlp'][row[0]] = '0-30'
                df['score_updated'][row[0]] = proto[0:2]
        elif ',' in row[1]['shape']:
            proto2 = re.sub(',', '.', proto)
            df['score_updated'][row[0]] = proto2
    return df


def resolve_sdai_scores(df):
    #for row in df.loc[df['shape'] == shape].iterrows():
    for row in df.iterrows():
        proto = row[1]['score']
        if '/86' in row[1]['_shape']:
            df['range'][row[0]] = proto[-2:]
            df['score_updated'][row[0]] = proto[0:-3]
    return df




def resolve_cdai_scores(df):
    #for row in df.loc[df['shape'] == shape].iterrows():
    for row in df.iterrows():
        proto = row[1]['score']
        if '/76' in row[1]['_shape']:
            #df['range'][row[0]] = proto[-2:]
            if proto[-2:] == '76':
                df['score_updated'][row[0]] = proto[0:-3]
        elif ',' in row[1]['_shape']:
            proto2 = re.sub(',', '.', proto)
            if len(proto2) < 6:
                df['score_updated'][row[0]] = proto2
    return df

def resolve_haq_scores(df):
    #for row in df.loc[df['shape'] == shape].iterrows():
    for row in df.iterrows():
        proto = row[1]['score']
        if ',' in row[1]['_shape']:
            proto2 = re.sub(',', '.', proto)
            if len(proto2) < 6:
                df['score_updated'][row[0]] = proto2
    return df


