import os, sys
import time
import glob
import re


#python setup.py jobid note_set

#job_start = int(sys.argv[1])
#job_end = int(sys.argv[2])
jobid = int(sys.argv[1])
note_set = str(sys.argv[2])
#joblist = list(range(job_start, job_end, 1))

#for jobid in joblist:
        
os.system('mkdir proto')
os.system('cp -r src proto')
    #os.system('cp *sh batch%02d' % jobid)
os.chdir('proto/src')

    #slurm_file = 'run.sh'
    #slurm_gpu = 'run-gpu.sh'
header_file = 'HP.py'
    #os.system('sed -i "s/BATCHID/%02d/g" %s' % (jobid, slurm_file))
    #os.system('sed -i "s/BATCHID/%02d/g" %s' % (jobid, slurm_gpu))
    #os.system('sed -i "s/BATCHID2/%02d/g" %s' % (jobid, input_file))
os.system('sed -i "s/BATCHID/%03d/g" %s' % (jobid, header_file))
os.system('sed -i "s/NOTESET/%s/g" %s' % (note_set, header_file))
print('starting the job:')
os.system("nohup python extract_pros.py > extract_pros.out")
time.sleep(0.5)

os.chdir('../../')
