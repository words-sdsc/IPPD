#!/usr/bin/python

'''
extract.py
@author a3rao 
last updated: 08.21.2016

parser for output of '/usr/bin/time -v' 
'''

import sys
import csv 
import subprocess
from os import listdir
from os.path import isfile, join


# check for filename being passed
if len(sys.argv) != 2 :
    print "Usage: python extract.py [directory]"
    sys.exit(1)


# construct a dictionary of column title, and key name in the specified file
params_list = {
		'UserTime' : 'User time (seconds)', 
		'SystemTime' : 'System time (seconds)', 
		'PercentCPU' : 'Percent of CPU this job got', 
		'WallTime' : 'Elapsed (wall clock) time (h:mm:ss or m:ss)', 
		'AvgSharedTxt' : 'Average shared text size (kbytes)', 
		'AvgUnsharedData' : 'Average unshared data size (kbytes)', 
		'AvgStack' : 'Average stack size (kbytes)', 
		'AvgTotal' : 'Average total size (kbytes)', 
		'MaxRSS' : 'Maximum resident set size (kbytes)', 
		'AvgRSS' : 'Average resident set size (kbytes)', 
		'MajorFaults' : 'Major (requiring I\/O) page faults', 
		'MinorFaults' : 'Minor (reclaiming a frame) page faults', 
		'VolCtxtSwitch' : 'Voluntary context switches', 
		'InvolCtxtSwitch' : 'Involuntary context switches', 
		'Swaps' : 'Swaps', 
		'FileSysIn' : 'File system inputs', 
		'FileSysOut' : 'File system outputs', 
		'SocketMsgSent' : 'Socket messages sent', 
		'SocketMsgRecv' : 'Socket messages received', 
		'SignalsDeliv' : 'Signals delivered', 
		'PageSize' : 'Page size (bytes)', 
		'ExitStatus' : 'Exit status', 
		'CommandTimed' : 'Command being timed' 
}


first = True
mypath = sys.argv[1]
profFiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
profFiles.sort()
numFiles = len(profFiles)
myfile =  open ("testing.csv", "w")


for i in range (0, numFiles): 

	filename = profFiles[i] 
	print 'Extracting info from: ' + filename

	# obtain parameter-value pairs embedded in filename
	fileparam_list = filename.split('_')
	fileparam_vallist = []
	fileparam_namelist = []
	i = 0
	while(fileparam_list[i] != 'qsubError'):
		fileparam_namelist.append(fileparam_list[i])
		fileparam_vallist.append(fileparam_list[i+1])
		i += 2

    # get a the list of param keys to extract
	params_keys_list = list(params_list.keys())
	params_keys_list.sort()
	
    # append them to get the header row of params
	csv_header_list = fileparam_namelist + params_keys_list

	# open CSV file, and header row if it is the first time
	wr = csv.writer(myfile, delimiter=",", quoting=csv.QUOTE_NONE, quotechar='')
	if first == True: 
		wr.writerow(csv_header_list)
		first = False


    # get the list of extracted param values from filename
	csv_row_entry = fileparam_vallist

	# delegate to bash script for parsing the current file
	for keytitle in params_keys_list:

		# Use this if python 2.7 or greater
		#value = subprocess.check_output(['./collectData.sh', filename, keytitle]);
	
		# Comet seems to have 2.6.6, so use this
		outerr = subprocess.Popen(['./collectData.sh', sys.argv[1] + '/' + filename, params_list[keytitle] + ": "], stdout=subprocess.PIPE)
		value, dummy = outerr.communicate()
	
		csv_row_entry.append(value)

	wr.writerow(csv_row_entry)


