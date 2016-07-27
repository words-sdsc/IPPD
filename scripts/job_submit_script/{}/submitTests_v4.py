#!/usr/bin/python


'''
for python 3, uncomment this and preceed it with a hash-bang 
/Library/Frameworks/Python.Framework/Versions/3.5/bin/python3.5
'''


'''
The subprocess calls used here include the timeout argument, which
only in Python 3.3+. Mac computers by default have a version of 
Python 2.7. THe #! statement above modifies which python to use.
'''

'''
submitTests.py version 4 
Arvind Rao (a3rao at ucsd dot edu)
2.17.2016

The purpose of this script is to automate the experiment process.
Users can list the variable parameters that need to be iterated over.
Still undergoing some changes.

I tried to design this in a way that user needs to add minimal modifications 
when adding additional variables to iterate over. It is basically doing what
the Kepler workflows do, but having it in a script makes it easier for 
addition of more parameters to iterate over. 

This script will invoke the Kepler workflow from the shell script, passing
the parameters needed. It will do so for each needed combination of parameters,
for the specified number of trials. 

'''


import os			# used when creating directories
import subprocess		# for calling bash script and commands
from itertools import product   # for easy dynamic nested for-loops


'''
Define number of trials per configuration here
'''
NumOfIterations = 1 

'''
Define a list of values for each iterable parameters here
'''
# list of all possible values for each changing param.
# Example: ppn_list = [2, 4, 8, 12, 16, 20, 24]
ppn_list = [24]
mem_list = ["110G"]
seqlen_list = ["5M"]


#seqlen_list = ["5M", "10M"]
#seqlen_list = ["5M"
#, "10M", "15M", "20M", "25M", "30M", "35M", "40M"]

# test for functonality
#ppn_list = [12]
#mem_list = ["60G", "90G"]

# LS002 limit check
#mem_list = ["110G"]

# map the param name to the corresponding list
params =    {

	'ppn' : ppn_list,
	'mem' : mem_list,
	'seqlen' : seqlen_list

	}


''' 
Define the remaining unchanging parameters here
For any unused/unneeded params, you can just leave them blank
'''

# SLURM Script Configs
#Jobname = "qa_filter"
#Jobname = "filter_human"
Jobname = "dup_seq"
JobErrorFile = "qsubError_" + Jobname
JobOutputFile = "qsubOutput_" + Jobname
Account = "ddp193"
NumOfNodes = "1"
Walltime = "24:00:00"
QueueType = "compute"
Mail = "a1singh@ucsd.edu"
SchedulerType = "SLURM"
TargetHost = "a3rao@comet.sdsc.edu"


# Program Specific / Directory Parameters
#JobFolderName = "qc"
#JobFolderName = "filter-human"
JobFolderName = "dup"
RemoteProjectDir = "/oasis/scratch/comet/a3rao/temp_project/NGS-ann-project"
RemoteScriptDir = RemoteProjectDir + "/NGS-ann"
RemoteToolDir = RemoteProjectDir + "/NGS-ann-program/bin"
RemoteRefDir = RemoteProjectDir + "/NGS-ann-ref"

#SampleName = "sHE-SRS012273"
SampleName = "LS002"
ErrTopDir = RemoteProjectDir + "/" + SampleName + "/qsubErrOut/" + JobFolderName
OutputDir = RemoteProjectDir + "/" + SampleName + "/" + JobFolderName
#InputDir = RemoteProjectDir + "/" + SampleName + "/qc"
InputDir = RemoteProjectDir + "/" + SampleName + "/filter-human"
ReadSplitDir = RemoteProjectDir + "/" + SampleName + "/read-split"
NumOfSplit = "512"

# names of input files, used for qc.
# Note that right now, it assumes that seqlen_list is defined
Input1_Prefix = "seq1_"
Input2_Prefix = "seq2_"



'''
itertools.product will give us a list of all possible 
combinations for each of the dynamic paramters
'''
keys = params.keys()				# list of all dynamic param names
var_combs = list(product(*params.values()))	# create all combinations
num_combs = len(var_combs)			# count number of combinations



'''
For each trial of each configuration,
1) creates script file with specified parameter values 
2) creates destination folders for each configuration/trial
3) submits a job to comet, to run the created script file
Note that it assumes that the results of filter-human step 
already exists in the project directory
'''
for i in range (0, num_combs):

	# file_specific variables
	ppn = str(var_combs[i][keys.index("ppn")])
	mem = str(var_combs[i][keys.index("mem")])
	seqlen = str(var_combs[i][keys.index("seqlen")])
	DirToAppend = '/ppn'+str(ppn)+'/mem'+str(mem)+'/seqlen'+str(seqlen)
	NameToAppend = 'ppn_'+str(ppn)+'_mem_'+str(mem)+'_seqlen_'+str(seqlen)
	

	# define strings to use for this config
	ConfigErrDir = ErrTopDir + DirToAppend
	ConfigOutputDir = OutputDir + DirToAppend
	ConfigInputDir = InputDir + DirToAppend
	ConfigReadSplitDir = ReadSplitDir + DirToAppend

	# define input name according to current seqlen setting

	# Used for QC
	#Input1 = Input1_Prefix + seqlen
	#Input2 = Input2_Prefix + seqlen



	print ("---------------")
	
	# run numOfIterations trials for each possible configuration
	for c in range (1, NumOfIterations+1):
		
		counter = str(c)	# string conversion
		
		# define strings to use for this trial 
		#TrialErrDir = ConfigErrDir + '/trial' + counter	## Old way... making directories

		
		# new way... embedding property details into the filename
		TrialErrorFile = NameToAppend + '_trial_' + counter + '_' + JobErrorFile
		TrialOutputFile = NameToAppend + '_trial_' + counter + '_' + JobOutputFile
		TrialErrDir = ErrTopDir

	
		# output file will be stored here.. This is to make sure that we don't overwrite into the same files. 
		TrialOutputDir = ConfigOutputDir + '/trial' + counter
		TrialInputDir = ConfigInputDir + '/trial' + counter
		TrialReadSplitDir = ConfigReadSplitDir + '/trial' + counter

		# Used for Filter Human... Todo: Need to figure out how to make it streamlined
		#Input1 = TrialInputDir + '/fltd-1'
		#Input2 = TrialInputDir + '/fltd-2' 

		# Used for Dup... Todo: Need to figure out how to make it streamlined
		Input1 = TrialInputDir + '/no-human-1'
		Input2 = TrialInputDir + '/no-human-2' 
	
		# print statements, so the user can see submission progress
		vars_str = ", ".join(str(x) for x in var_combs[i])
		print ("Submitting Combination # %d out of %d, Trial # %d out of %d: (%s)" % (i+1, num_combs, c, NumOfIterations, vars_str))


		# submit task to comet via Kepler
	
		#wf_module_name="NGS_preprocessing.NGS-qa-filter.CometExec."
		#wf_module_name="NGS_preprocessing.bowtie2.CometExec."
		wf_module_name="NGS_preprocessing.DUP-TEST.CometExec."
		#p = subprocess.Popen(	#### don't use Popen, this will create a separate process, not a call from same process.	

		p = subprocess.check_call(
								['./kepler.sh','-runwf', '-nogui', '-nocache', 
								'-' + wf_module_name + 'Account', Account,
								'-' + wf_module_name + 'Input1', Input1,
								'-' + wf_module_name + 'Input2', Input2,
								'-' + wf_module_name + 'JobErrorFile', TrialErrorFile,
								'-' + wf_module_name + 'JobOutputFile', TrialOutputFile,
								'-' + wf_module_name + 'Jobname', Jobname,
								'-' + wf_module_name + 'Mail', Mail,
								'-' + wf_module_name + 'NumOfNodes', NumOfNodes,
								'-' + wf_module_name + 'NumOfSplit', NumOfSplit,
								'-' + wf_module_name + 'QueueType', QueueType,
								'-' + wf_module_name + 'RemoteJobDir', TrialOutputDir,
								'-' + wf_module_name + 'RemoteProjectDir', RemoteProjectDir,
								'-' + wf_module_name + 'RemoteRefDir', RemoteRefDir,
								'-' + wf_module_name + 'RemoteScriptDir', RemoteScriptDir,
								'-' + wf_module_name + 'RemoteToolDir', RemoteToolDir,
								'-' + wf_module_name + 'SampleName', SampleName,
								'-' + wf_module_name + 'SchedulerType', SchedulerType,
								'-' + wf_module_name + 'TargetHost', TargetHost,
								'-' + wf_module_name + 'TrialErrDir', TrialErrDir,
								'-' + wf_module_name + 'TrialOutputDir', TrialOutputDir,
								'-' + wf_module_name + 'TrialReadSplitDir', TrialReadSplitDir,
								'-' + wf_module_name + 'Walltime', Walltime,
								'-' + wf_module_name + 'mem', mem,
								'-' + wf_module_name + 'ppn', ppn,
								'-' + wf_module_name + 'seqlen', seqlen,
								#'../Desktop/arvind_workflows/new_dup.xml'])		### call Kepler, specify workflow name here.
								#'../Desktop/arvind_workflows/comet_LS002_first1.xml'])		### call Kepler, specify workflow name here.
								#'../Desktop/arvind_workflows/comet_LS002_secondOnly.xml'])		### call Kepler, specify workflow name here.
								'/home/comet/workflows/IPPD/sampleWorkflow/comet_LS002_thirdOnly.xml'])		### call Kepler, specify workflow name here.
		print ("Submitted Combination # %d out of %d, Trial # %d out of %d: (%s)" % (i+1, num_combs, c, NumOfIterations, vars_str))
		#subprocess.check_call(['sleep','15'])		# do we need to delay?...


print ("---------------")
print ("All jobs have been submitted to Comet.")

