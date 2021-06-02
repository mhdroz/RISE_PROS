import os, sys
import time
import glob
import re


#python setup.py job_start job_end note_set

job_start = int(sys.argv[1])
job_end = int(sys.argv[2])
note_set = int(sys.argv[3])
joblist = list(range(job_start, job_end, 1))

for jobid in joblist:
        
    os.system('mkdir set%01d_batch%03d' % (note_set, jobid))
    os.system('cp -r src set%01d_batch%03d' % (note_set, jobid))
    #os.system('cp *sh batch%02d' % jobid)
    os.chdir('set%01d_batch%03d/src' % (note_set, jobid))

    #slurm_file = 'run.sh'
    #slurm_gpu = 'run-gpu.sh'
    header_file = 'HP.py'
    #os.system('sed -i "s/BATCHID/%02d/g" %s' % (jobid, slurm_file))
    #os.system('sed -i "s/BATCHID/%02d/g" %s' % (jobid, slurm_gpu))
    #os.system('sed -i "s/BATCHID2/%02d/g" %s' % (jobid, input_file))
    os.system('sed -i "s/BATCHID/%03d/g" %s' % (jobid, header_file))
    os.system('sed -i "s/NOTESET/%01d/g" %s' % (note_set, header_file))
    print('starting the job:')
    os.system("nohup python extract_pros.py > extract_pros.out")
    time.sleep(0.5)

    os.chdir('../../')
