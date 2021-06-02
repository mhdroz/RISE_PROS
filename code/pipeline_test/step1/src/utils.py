import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import sys, os, re
import spacy
import pickle


def clean_html(raw_text):

    text = re.sub('<', ' <', raw_text)
    text = re.sub('>', '> ', text)
    soup = BeautifulSoup(text)
    text = soup.get_text()
    text = re.sub(' +', ' ', text)

    return text

def clean_text(text):
    text = re.sub("\\\\tab",' ',text)
    text = re.sub('\\\\par', ' ', text)
    text = re.sub('\\\\fs20', ' ', text)
    text = re.sub('\\\\b0', ' ', text)
    text = re.sub('\\\\fs22', ' ', text)
    text = re.sub('\\\\ul0', ' ', text)
    text = re.sub('\\\\b', ' ', text)
    text = re.sub('\\\\ul', ' ', text)
    text = re.sub('\\\\fs24', ' ', text)
    text = re.sub('\\\\plain', ' ', text)
    text = re.sub('\\\\f0', ' ', text)
    text = re.sub('\\\\cf0', ' ', text)
    text = re.sub('\\xa0', ' ', text)
    text = re.sub('\\t', ' ', text)
    text = re.sub(' +', ' ', text)
    return text


def PRO_scores_extraction(original_notes, nlp, matcher, window=10):
    """

    Args:
        original_notes: clinical notes to process in pandas dataframe format
        nlp: SpaCy language model
        matcher: SpaCy PhraseMatcher built with nlp model
        window: window of tokens for left and right contexts. Default: 10

    Returns:
        concepts_df: Pandas dataframe with extracted clinical term, their position in the text, as well as left- and right-context
    """


    concepts_df = pd.DataFrame(columns=['patid', 'note_id', 'practice_id','date', 'tag', 'assessment',
                               'score', 'pos', 'context_left', 'context_right'])

    for index in original_notes.index:
        new_df = pd.DataFrame(columns=['patid', 'note_id', 'practice_id','date', 'tag','assessment',
                               'score', 'pos', 'context_left', 'context_right'])

        if original_notes['length_clean'][index] > 1000000:
            print("Doc length exceeds the limit. Skipping...")
            doc = nlp('Doc too large')
        else:
            doc = nlp(original_notes['clean'][index])
        #access_number = original_notes['accessionNumber'][index]
        patid = original_notes['patient_id'][index]
        noteid = original_notes['note_id'][index]
        otherid = original_notes['other_id'][index]
        date = original_notes['date'][index]
        #note_title = original_notes['note_title'][index]

        matches = matcher(doc)
        tag = []
        start_idx = []
        end_idx = []
        term = []
        loc_concept = []
        pos = []
        score = []

        #tokens = [token.text for token in doc]
        tokens = [token.text.lower() for token in doc]
        left_tokens = []
        right_tokens = []
        #window = 10



        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            #print(match_id, string_id, start, end, span.text)
            if end == len(doc):    #Check if tagged entity is at the end of the doc. For when score interpretation is tagged instead of assessment
                #print("End doc")
                next_tok = doc[end-1:]
            else:
                next_tok = doc[end:end+1]  #Extract score related to assessment tagged
            #print(next_tok)
            for tok in next_tok:
                if tok.like_num == True:    #If the score is the assessment's next token
                    score.append(next_tok.text)
                elif tok.text == 'Interpretation':  #Detect if the interpretation range is given
                    if (len(doc) - end) < (end+4):
                        score.append('No score')
                        span = doc[start:end+1]
                    else:
                    #print(span, tok, doc[end+1:end+4])
                        score_int = doc[end+1:end+4]
                        score.append(score_int.text)
                        span = doc[start:end+1]       #Update the extracted entity to avoid confusion
                elif tok.text == 'Function':    #Detect occurences of MDHAQ Function Index
                    if (len(doc) - end) < (end+2):
                        score.append('No score')
                        span = doc[start:end+1]
                    else:
                    #print("New")
                        span = doc[start:end+2]     #Update the extracted entity
                        score_func = doc[end+2:end+3]
                        score.append(score_func.text) #Get the score

                else:
                    #print("searching further:")
                    if end == len(doc):       #Another check for the end of the document
                  #  if end+1 <= len(doc):
                        #next_span = doc[end-1:]
                        score.append('No score')
                    else:
                        next_span = doc[end+1:end+2] #Look for one token further. For the cases when Score is between the assessment and the score
                        if next_span:
                            for tok in next_span:
                            #print(tok)
                                if tok.like_num == True:
                            #print(tok)
                                    score.append(tok.text)
                            #print(span, 'score:', tok)
                                else:
                                    #print("searching even further:")
                                    next_span2 = doc[end+2:end+3]
                                    if next_span2:
                                        for tok2 in next_span2:
                                            if tok2.like_num == True:
                                                score.append(tok2.text)
                                            else:
                                                score.append('No score')  #If no score is detected
                                    else:
                                        score.append('No score')
                        else:
                            score.append('No score')
            #print(score)



            tag.append(string_id)
            start_idx.append(start)
            end_idx.append(end)
            term.append(span.text)
            loc = (start, end)
            loc_concept.append(loc)
    #print(loc_concept)

        for idx in loc_concept:
            start = idx[0]+1
            end = idx[1]
            left_tokens.append(tokens[0:start][-1 -window : -1])
            right_tokens.append(tokens[end:-1][0 : 1+window])
            pos.append(start)

        #if term:
        #    print(index)
        #    print(score)
        #    print(tag)
        #    print(term)

        new_df['tag'] = tag
        new_df['assessment'] = term
        new_df['patid'] = patid
        new_df['note_id'] = noteid
        new_df['practice_id'] = otherid
        #new_df['note_title'] = note_title
        #new_df['access_number'] = access_number
        new_df['date'] = date
        new_df['pos'] = pos
        new_df['score'] = score
        new_df['context_right'] = right_tokens
        new_df['context_left'] = left_tokens

        concepts_df = concepts_df.append(new_df)
        concepts_df = concepts_df.reset_index(drop=True)

    return concepts_df

def PRO_scores_extraction_v2(original_notes, nlp, matcher, window=10):
    """

    Args:
        original_notes: clinical notes to process in pandas dataframe format
        nlp: SpaCy language model
        matcher: SpaCy PhraseMatcher built with nlp model
        window: window of tokens for left and right contexts. Default: 10

    Returns:
        concepts_df: Pandas dataframe with extracted clinical term, their position in the text, as well as left- and right-context
    """


    concepts_df = pd.DataFrame(columns=['patid', 'note_id', 'other_id','date', 'tag', 'assessment',
                               'score', 'pos', 'context_left', 'context_right'])

    for index in original_notes.index:
        new_df = pd.DataFrame(columns=['patid', 'note_id', 'other_id','date', 'tag','assessment',
                               'score', 'pos', 'context_left', 'context_right'])


        if original_notes['length_clean'][index] > 1000000:
            print("Doc length exceeds the limit. Skipping...")
            doc = nlp('Doc too lagre')
        else:
            doc = nlp(original_notes['clean'][index])
        #access_number = original_notes['accessionNumber'][index]
        patid = original_notes['patient_id'][index]
        noteid = original_notes['note_id'][index]
        otherid = original_notes['other_id'][index]
        date = original_notes['date'][index]
        #note_title = original_notes['note_title'][index]

        matches = matcher(doc)
        tag = []
        start_idx = []
        end_idx = []
        term = []
        loc_concept = []
        pos = []
        score = []

        #tokens = [token.text for token in doc]
        tokens = [token.text.lower() for token in doc]
        left_tokens = []
        right_tokens = []

        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]

            if string_id == 'INTERPRETATION':
                score.append(span.text)

            else:
            
                for tok in doc[end:span.sent.end]:
                    if tok.like_num == True:
                    #print(span, tok)
                        score.append(tok.text)
                        if score: break
            #if string_id == 'INTERPRETATION':
            #    score.append(span.text)
                if not score:
                    score.append('No score')
                
            tag.append(string_id)
            start_idx.append(start)
            end_idx.append(end)
            term.append(span.text)
            loc = (start, end)
            loc_concept.append(loc)

            if len(score) < len(term):
                score.append('No score')
    #print(loc_concept)

        if 'DAS 28' in term:
            if term.index('DAS 28') - 1 == term.index('DAS'):
                print("Double tag!")
                idx = term.index('DAS')
                term.pop(idx)
                tag.pop(idx)
                score.pop(idx)
                loc_concept.pop(idx)

        for idx in loc_concept:
            start = idx[0]+1
            end = idx[1]
            left_tokens.append(tokens[0:start][-1 -window : -1])
            right_tokens.append(tokens[end:-1][0 : 1+window])
            pos.append(start)

        #debug
        #print(index)
        #print(score)
        #print(tag)
        #print(term)
      #  if term:
      #      print(index)
      #      print(score)
      #      print(tag)
      #      print(term)

        new_df['tag'] = tag
        new_df['assessment'] = term
        new_df['patid'] = patid
        new_df['note_id'] = noteid
        new_df['other_id'] = otherid
        #new_df['note_title'] = note_title
        #new_df['access_number'] = access_number
        new_df['date'] = date
        new_df['pos'] = pos
        new_df['score'] = score
        new_df['context_right'] = right_tokens
        new_df['context_left'] = left_tokens

        concepts_df = concepts_df.append(new_df)
        concepts_df = concepts_df.reset_index(drop=True)

    return concepts_df

def PRO_scores_extraction_v4(original_notes, nlp, matcher, window=10):
    """

    Args:
        original_notes: clinical notes to process in pandas dataframe format
        nlp: SpaCy language model
        matcher: SpaCy PhraseMatcher built with nlp model
        window: window of tokens for left and right contexts. Default: 10

    Returns:
        concepts_df: Pandas dataframe with extracted clinical term, their position in the text, as well as left- and right-context
    """


    concepts_df = pd.DataFrame(columns=['patid', 'note_id', 'other_id','date', 'tag', 'assessment',
                               'score', 'pos', 'context_left', 'context_right'])

    for index in original_notes.index:
        new_df = pd.DataFrame(columns=['patid', 'note_id', 'other_id','date', 'tag','assessment',
                               'score', 'pos', 'context_left', 'context_right'])


        doc = nlp(original_notes['clean'][index])
        #access_number = original_notes['accessionNumber'][index]
        patid = original_notes['patient_id'][index]
        noteid = original_notes['note_id'][index]
        otherid = original_notes['other_id'][index]
        date = original_notes['date'][index]
        #note_title = original_notes['note_title'][index]

        matches = matcher(doc)
        tag = []
        start_idx = []
        end_idx = []
        term = []
        loc_concept = []
        pos = []
        score = []
        tmp = 0

        #tokens = [token.text for token in doc]
        tokens = [token.text.lower() for token in doc]
        left_tokens = []
        right_tokens = []
        #window = 10



        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            print(span)
            if string_id == 'INTERPRETATION':
                score.append(span.text)

            else:

                for tok in doc[end:span.sent.end]:
                    print(doc[end:span.sent.end])
                    #print(tok)
                    if tok.text == '-':
                        #print(tok.shape)
                        #print('span: ', span, 'start: ', start, 'end: ', end, 'score: ', tok, 'pos: ', tok.i)
                        #tmp.append(tok.i)
                        tmp = tok.i
                        #print("TMP: ", tmp)
                    #print(tmp)
                    if tok.like_num == True:
                        print("NUM: ", tok.text)
                        print("Delta '-' and num: ", tmp - tok.i)
                        print("Delta '-' and tag: ", tmp - end)
                        print("Delta num and tag: ", tok.i - end)
                        if (tmp - end) > 0 & (tmp - end) < (tok.i - end):
                        #    print("Num is not score!")
                            score.append('No score')
                        else:
                        #    print('span: ', span, 'start: ', start, 'end: ', end, 'score: ', tok, 'pos: ', tok.i)
                            score.append(tok.text)


                        #score.append(tok.text)
                        if score: break
                    print(score)
            #if string_id == 'INTERPRETATION':
            #    score.append(span.text)
                if not score:
                    score.append('No score')

            tag.append(string_id)
            start_idx.append(start)
            end_idx.append(end)
            term.append(span.text)
            loc = (start, end)
            loc_concept.append(loc)

            if len(score) < len(term):
                score.append('No score')
    #print(loc_concept)

        if 'DAS 28' in term:
            if term.index('DAS 28') - 1 == term.index('DAS'):
                print("Double tag!")
                idx = term.index('DAS')
                term.pop(idx)
                tag.pop(idx)
                score.pop(idx)
                loc_concept.pop(idx)
            print(loc_concept)
            print(term)
            print(tag)
            print(score)

        for idx in loc_concept:
            start = idx[0]+1
            end = idx[1]
            left_tokens.append(tokens[0:start][-1 -window : -1])
            right_tokens.append(tokens[end:-1][0 : 1+window])
            pos.append(start)




        #debug
        #print(index)
        #print(score)
        #print(tag)
        #print(term)
     #   if term:
     #       print(index)
     #       print(score)
     #       print(tag)
     #       print(term)

        new_df['tag'] = tag
        new_df['assessment'] = term
        new_df['patid'] = patid
        new_df['note_id'] = noteid
        new_df['other_id'] = otherid
        #new_df['note_title'] = note_title
        #new_df['access_number'] = access_number
        new_df['date'] = date
        new_df['pos'] = pos
        new_df['score'] = score
        new_df['context_right'] = right_tokens
        new_df['context_left'] = left_tokens

        concepts_df = concepts_df.append(new_df)
        concepts_df = concepts_df.reset_index(drop=True)

    return concepts_df
