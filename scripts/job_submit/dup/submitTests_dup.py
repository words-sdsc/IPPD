#!/usr/bin/env python
'''
@author a3rao
submitTests.py for MTGA CD-HIT-DUP
'''

import os                       # used when creating directories
import subprocess               # for calling bash script and commands
from itertools import product   # for easy dynamic nested for-loops


## Define lists of values for each iterable parameter
module_list = ["dup"]

sample_list = ["LS002"]

mem_list = ["64G"]  # Fixed for now... taken care of in workflow by numactl --membind

seqlen_list = [
"5M", "10M", "15M", "20M", 
"25M", "30M", "35M", "40M", 
"60M", "80M", "100M", "120M", 
"140M", "160M"]

ppn_list = [2, 5, 8, 11]    # Note: numactl 3, 6, 9, 12 cores respectively

NumOfIterations = 1     # number of trials per configuration here
trial_list = list(range(1, NumOfIterations+1))


# Map the param name to the corresponding list
params = {
    'module' : module_list,
    'sample' : sample_list,
    'ppn' : ppn_list,
    'mem' : mem_list,
    'seqlen' : seqlen_list,
    'trial' : trial_list
}


# itertools.product will give us a list of all possible 
# combinations for each changing parameters 
keys = list(params.keys())                  # list of all dynamic param names
var_combs = list(product(*params.values())) # create all combinations
num_combs = len(var_combs)                  # count number of combinations

# Sequence of parameter names to be used to determine file naming 
paramKeys = ["module", "sample", "seqlen", "ppn", "mem", "trial"]


# For each trial of each configuration,
# - defines appropriate directory location for input/output files
# - submits a job to comet through specified Kepler workflow, passing params
for i in range (0, num_combs):

    print ("---------------")

    # Combination specific values
    module = str(var_combs[i][keys.index("module")])
    sample = str(var_combs[i][keys.index("sample")])
    ppn = str(var_combs[i][keys.index("ppn")])
    mem = str(var_combs[i][keys.index("mem")])
    seqlen = str(var_combs[i][keys.index("seqlen")])
    trial = str(var_combs[i][keys.index("trial")])

    # Generate identifier string based on combination 
    # Example: 'ppn_16_mem_64G_seqlen_64G_.....
    NameToAppend = ""
    for key in paramKeys:
        NameToAppend = NameToAppend + key + "_" + str(var_combs[i][keys.index(key)]) + "_"

    # SLURM Script Configs
    Jobname = module 
    JobErrorFile = "qsubError_" + Jobname
    JobOutputFile = "qsubOutput_" + Jobname
    Account = "ddp193"
    NumOfNodes = "1"
    Walltime = "24:00:00"
    QueueType = "compute"
    SchedulerType = "SLURM"
    Mail = "a3rao@ucsd.edu"
    TargetHost = "a3rao@comet.sdsc.edu"

    # Local File Locations - fill out appropriately
    IdentityFile = '/Users/arvind/.ssh/id_rsa'
    KeplerLocation = '/Users/arvind/Projects/kepler.modules/kepler.sh'
    WorkflowLocation = '/Users/arvind/Projects/ippd/workflows/comet_xml/comet_dup.xml'


    # 'real remote' - location of base project on comet, 'on login node'
    # this is used by when copying executable and data to compute node
    realRemoteDir = "/oasis/scratch/comet/a3rao/temp_project/NGS-ann-project"

    # 'remote project dir' - project base from the perspective of the trial, 'on compute node'
    RemoteProjectDir = "."
    RemoteJobDir = "."

    RemoteScriptDir = RemoteProjectDir + "/NGS-ann"
    RemoteToolDir = RemoteProjectDir + "/NGS-ann-program/BIN"
    RemoteRefDir = RemoteProjectDir + "/NGS-ann-ref"

    # Directory Setup
    JobFolderName = module
    SampleName = sample 

    ErrTopDir = RemoteProjectDir    + "/qsubErrOut"
    OutputDir = RemoteProjectDir    + "/outputDir"
    InputDir = RemoteProjectDir     + "/data"

    # Outputs will be copied back here
    realRemoteSampleDir = realRemoteDir + "/" + SampleName + "/dataFiles." + Jobname
    realRemoteErrorDir =  realRemoteSampleDir +"/qsubErrOut" 
    realRemoteOutputDir = realRemoteSampleDir + "/OutputDir/seqlen" + seqlen 

    # Embedding property details into the filename
    TrialErrorFile = NameToAppend + JobErrorFile
    TrialOutputFile = NameToAppend + JobOutputFile
    
    # Output file will be stored here.. 
    # This is to make sure that we don't overwrite into the same files. 
    TrialInputDir = InputDir
    TrialOutputDir = OutputDir
    TrialErrDir = ErrTopDir

    # Inputs files for DUP
    Input1 = TrialInputDir + '/no-human-1'
    Input2 = TrialInputDir + '/no-human-2' 
    TrialReadSplitDir = TrialOutputDir + "/read-split"
    NumOfSplit = "512"
    
    # Print statements to display submission progress
    vars_str = ", ".join(str(x) for x in var_combs[i])
    print ("Submitting Combination # %d out of %d: (%s)" % (i+1, num_combs, vars_str))

    # Submit task to comet via Kepler
    wf_module_name="NGS_preprocessing." + Jobname + ".CometExec."
    p = subprocess.check_call([
            KeplerLocation,'-runwf', '-nogui', '-nocache', 
            '-' + wf_module_name + 'Account', Account,
            '-' + wf_module_name + 'IdentityFile', IdentityFile,
            '-' + wf_module_name + 'Input1', Input1,
            '-' + wf_module_name + 'Input2', Input2,
            '-' + wf_module_name + 'JobErrorFile', TrialErrorFile,
            '-' + wf_module_name + 'JobOutputFile', TrialOutputFile,
            '-' + wf_module_name + 'Jobname', Jobname,
            '-' + wf_module_name + 'Mail', Mail,
            '-' + wf_module_name + 'NumOfNodes', NumOfNodes,
			'-' + wf_module_name + 'NumOfSplit', NumOfSplit,
            '-' + wf_module_name + 'QueueType', QueueType,
            '-' + wf_module_name + 'RemoteJobDir', RemoteJobDir,
            '-' + wf_module_name + 'RemoteProjectDir', RemoteProjectDir,
            '-' + wf_module_name + 'RemoteRefDir', RemoteRefDir,
            '-' + wf_module_name + 'RemoteScriptDir', RemoteScriptDir,
            '-' + wf_module_name + 'RemoteToolDir', RemoteToolDir,
            '-' + wf_module_name + 'SampleName', SampleName,
            '-' + wf_module_name + 'SchedulerType', SchedulerType,
            '-' + wf_module_name + 'TargetHost', TargetHost,
            '-' + wf_module_name + 'TrialInputDir', TrialInputDir,
            '-' + wf_module_name + 'TrialErrDir', TrialErrDir,
            '-' + wf_module_name + 'TrialOutputDir', TrialOutputDir,
			'-' + wf_module_name + 'TrialReadSplitDir', TrialReadSplitDir,
            '-' + wf_module_name + 'Walltime', Walltime,
            '-' + wf_module_name + 'mem', mem,
            '-' + wf_module_name + 'ppn', ppn,
            '-' + wf_module_name + 'seqlen', seqlen,
            '-' + wf_module_name + 'realRemoteErrorDir', realRemoteErrorDir,  
            '-' + wf_module_name + 'realRemoteOutputDir', realRemoteOutputDir,  
            '-' + wf_module_name + 'realRemoteDir', realRemoteDir,
            WorkflowLocation ])
    print ("Submitted Combination # %d out of %d: (%s)" % (i+1, num_combs, vars_str))
    subprocess.check_call(['sleep','15'])       # do we need to delay?...

print ("---------------")
print ("All jobs have been submitted to Comet.")
