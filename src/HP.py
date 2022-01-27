#####################################################
#           Parameters for the project              #
#####################################################

#Parameters
project = 'rise_pro'
start_date = '2014-12-31'
end_date = '2019-01-01'
notes_set = 'proto'
#BatchID = 'BATCHID'
threads = 4


#Paths
path_termino = '../res/'
path_data = '../data/'
output_path = '../output/' 

#Datasets
input_notes = path_data + 'notes_pipeline_test.parquet'
extracted_clean = output_path + 'scores_resolved.parquet'


#NLP-terminologies
spacy_model = 'en_core_sci_md'
terms_da = path_termino + 'da.txt'
terms_fs = path_termino + 'fs.txt'




#Outputs
extracted_pros = output_path + notes_set+'_extracted.parquet'
extracted_clean_formatted = output_path + 'extracted_clean_formatted.parquet'

#Instrument-specific parameters
assessment = 'cdai'
rapid3_list = ['rapid3', 'rapid 3', 'rapid-3', 'rapid3/mdhaq']
das28_list = ['das', 'das28', 'das 28']

