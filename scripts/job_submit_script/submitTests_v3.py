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
submitTests.py version 3
Arvind Rao (a3rao@ucsd.edu)
02.16.2016

The purpose of this script is to automate the experiment process.
Users can list the variable parameters that need to be iterated over.

I tried to design this in a way that user needs to add minimal modifications 
when adding additional variables to iterate over. It is basically doing what
the Kepler workflows do, but having it in a script makes it easier for 
addition of more parameters to iterate over. 

This script will invoke the Kepler workflow from the shell script, passing
the parameters needed. It will do so for each needed combination of parameters,
for the specified number of trials. 

The previous version of the script was intended to be run on Comet, and it would
create a file every time. This version will get rid of that problem.

'''


import os			# used when creating directories
import subprocess		# for calling bash script and commands
from itertools import product   # for easy dynamic nested for-loops


'''
Define number of trials per configuration here
'''
NumOfIterations = 3 

'''
Define a list of values for each iterable parameters here
'''
# list of all possible values for each changing param.
# Example: ppn_list = [2, 4, 8, 12, 16, 20, 24]
#ppn_list = [24]
#mem_list = ["60G", "90G"]


# test for functonality
#ppn_list = [12]
#mem_list = ["60G", "90G"]

# LS002 limit check
#mem_list = ["110G"]

#LS002 profiling check

ppn_list = [2, 4, 8, 12, 16, 20, 24]
mem_list = ["110G"]


# map the param name to the corresponding list
params =    {

	'ppn' : ppn_list,
	'mem' : mem_list

	}


''' 
Define the remaining unchanging parameters here
'''

# SLURM Script Configs
Jobname = "dup_seq"
JobErrorFile = "qsubError_" + Jobname
JobOutputFile = "qsubOutput_" + Jobname
Account = ""			## Specify Account
NumOfNodes = "1"
Walltime = "24:00:00"
QueueType = "compute"		
Mail = ""					## Specify EMAIL


# Program Specific / Directory Parameters
RemoteProjectDir = "" 		## Specify REMOTEPROJECTDIR here.
RemoteToolDir = RemoteProjectDir + "/NGS-ann-program/bin"
#SampleName = "sHE-SRS012273"
SampleName = "LS002"
ErrTopDir = RemoteProjectDir + "/" + SampleName + "/qsubErrOut/dup"
OutputDir = RemoteProjectDir + "/" + SampleName + "/dup"
ReadSplitDir = RemoteProjectDir + "/" + SampleName + "/read-split"
NumOfSplit = "512"


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
	DirToAppend = '/ppn'+str(ppn)+'/mem'+str(mem)
	NameToAppend = 'ppn_'+str(ppn)+'_mem_'+str(mem)
	
	# define strings to use for this config
	ConfigErrDir = ErrTopDir + DirToAppend
	ConfigOutputDir = OutputDir + DirToAppend
	ConfigReadSplitDir = ReadSplitDir + DirToAppend

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
		TrialReadSplitDir = ConfigReadSplitDir + '/trial' + counter
	
		# print statements, so the user can see submission progress
		vars_str = ", ".join(str(x) for x in var_combs[i])
		print ("Submitting Combination # %d out of %d, Trial # %d out of %d: (%s)" % (i+1, num_combs, c, NumOfIterations, vars_str))


		# submit task to comet via Kepler
	
		wf_module_name="NGS_preprocessing.DUP-TEST.CometExec."
		#p = subprocess.Popen(					## using this will create a separate process, not a call from same process.	
		p = subprocess.check_call(
								['./kepler.sh','-runwf', '-nogui', '-nocache', 
								'-' + wf_module_name + 'Account', Account,
								'-' + wf_module_name +'JobErrorFile', TrialErrorFile,
								'-' + wf_module_name +'JobOutputFile', TrialOutputFile,
								'-' + wf_module_name +'Jobname', Jobname,
								'-' + wf_module_name +'Mail', Mail,
								'-' + wf_module_name +'NumOfNodes', NumOfNodes,
								'-' + wf_module_name +'NumOfSplit', NumOfSplit,
								'-' + wf_module_name +'QueueType', QueueType,
								'-' + wf_module_name +'RemoteProjectDir', RemoteProjectDir,
								'-' + wf_module_name +'RemoteToolDir', RemoteToolDir,
								'-' + wf_module_name +'SampleName', SampleName,
								'-' + wf_module_name +'SchedulerType', 'SLURM',
								'-' + wf_module_name +'TargetHost', '',				## SPECIFY HOST HERE.
								'-' + wf_module_name +'TrialErrDir', TrialErrDir,
								'-' + wf_module_name +'TrialOutputDir', TrialOutputDir,
								'-' + wf_module_name +'TrialReadSplitDir', TrialReadSplitDir,
								'-' + wf_module_name +'Walltime', Walltime,
								'-' + wf_module_name +'mem', mem,
								'-' + wf_module_name +'ppn', ppn,
								'-' + wf_module_name +'remoteJobDir', TrialErrDir,
								'../Desktop/arvind_workflows/new_dup.xml'])		### call Kepler, specify file name here.
		print ("Submitted Combination # %d out of %d, Trial # %d out of %d: (%s)" % (i+1, num_combs, c, NumOfIterations, vars_str))
		#subprocess.check_call(['sleep','15'])		# do we need to delay?...


print ("---------------")
print ("All jobs have been submitted to Comet.")

