#!/usr/bin/python

'''
extract.py
Arvind Rao (a3rao@ucsd.edu)
last updated: 02.24.2016

a bit messsy right now, but i do plan to clean it up
'''

import sys
import csv 
import subprocess
from os import listdir
from os.path import isfile, join


# check for filename being passed
if len(sys.argv) == 1 :
	#print "give me the output file name..."
	print "give me a directory name with profile data files..."
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
		'MajorFaults' : 'Major (requiring I/O) page faults', 
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


first = True;
mypath = sys.argv[1]
profFiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
numFiles = len(profFiles)
#print numFiles

# print the file name
profFiles.sort()
#print profFiles
#print('\n')

myfile =  open ("testing.csv", "w")

for i in range (0, numFiles): 

	filename = profFiles[i] 
	print 'Extracting info from: ' + filename


	# obtain file name list
	fileparam_list = filename.split('_')
	fileparam_vallist = []
	fileparam_namelist = []

	# get name of filename params
	i = 0
	while(fileparam_list[i] != 'qsubError'):
		fileparam_namelist.append(fileparam_list[i])
		fileparam_vallist.append(fileparam_list[i+1])
		i += 2

	'''
	# print filename params
	for key in fileparam_namelist:
		print (key + 't'),

	# print column titles
	for key in params_list:
		print (key + '\t'),
	print('\n')
	'''

	params_keys_list = params_list.keys()
	params_keys_list.sort();
	
	csv_header_list = fileparam_namelist + params_keys_list
	#csv_header_list.sort()
	#print(csv_header_list)

	# open CSV file, add header row
	wr = csv.writer(myfile, delimiter=",", quoting=csv.QUOTE_NONE, quotechar='')
	
	if first == True: 
		wr.writerow(csv_header_list)
		first = False


	csv_row_entry = fileparam_vallist
	'''
	for value in fileparam_vallist:
		#print (value + '\t'),
		csv_row_entry.append(value)
	'''


	# get name and values of filename params
	fileparam_list = filename.split('_')
	fileparam_vallist = []
	fileparam_namelist = []
	i = 0
	while(fileparam_list[i] != 'qsubError'):
		fileparam_namelist.append(fileparam_list[i])
		fileparam_vallist.append(fileparam_list[i+1])
		i += 2
	csv_row_entry = fileparam_vallist

	# print extracted value. delegates to bash script for extraction
	for keytitle in params_keys_list:


		#csv_row_entry = []
#		print params_list[keytitle]

		# Use this if python 2.7 or greater
		#value = subprocess.check_output(['./collectData.sh', filename, keytitle]);
	
		# Comet seems to have 2.6.6, so use this
		outerr = subprocess.Popen(['./collectData.sh', sys.argv[1] + '/' + filename, params_list[keytitle] + ": "], stdout=subprocess.PIPE)
		value, dummy = outerr.communicate()
	
		#print (value + '\t'),
		csv_row_entry.append(value)
#	print('\n')
#	print(csv_row_entry)
	wr.writerow(csv_row_entry)
	#subprocess.check_call(['mkdir', '-p', ReadSplitDir])

