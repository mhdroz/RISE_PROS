#####################################################
#           Parameters for the project              #
#####################################################

#Parameters
project = 'rise_pro'
start_date = '2014-12-31'
end_date = '2019-01-01'
notes_set = 'NOTESET'
BatchID = 'BATCHID'
threads = 14


#Paths
path_termino = '/share/pi/stamang/rise_pro/res/'
path_data = '/share/pi/stamang/rise_pro/data/'
output_path = '/share/pi/stamang/rise_pro/output/pipeline_test/' 

#Datasets
input_notes = path_data + 'notes_pipeline_test.parquet'
extracted_clean = output_path + 'extracted_clean.parquet'



#NLP-terminologies
terms_da = path_termino + 'da.txt'
terms_fs = path_termino + 'fs.txt'




#Outputs
extracted_pros = output_path + 'set'+notes_set+'_batch'+BatchID+'_extracted.parquet'
extracted_clean_formatted = output_path + 'extracted_clean_assessment_column.parquet'
