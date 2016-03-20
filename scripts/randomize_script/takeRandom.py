'''
takeRandom.py (LS002)
Arvind Rao (a3rao@ucsd.edu)
last updated: 02.24.2016

used for creating randomized input files for MTGA
'''

#!/usr/bin/python

from random import randint
import sys

total_lines = 650829376		## This is number of lines in LS002 input
total_seqs  = 162707344		## A sequence is composed of four lines
#total_toget = 1000000		## Todo?: make compatible to all files

if (len(sys.argv) != 2):
	print "Usage: ./takeRandom [number of lines to get (in Millions)] "
	sys.exit()

toget_str = sys.argv[1]
total_toget = int(toget_str) * 1000000

if (total_toget > total_seqs):
	print "Sequences to get is too large"
	sys.exit()

print "Setting up..." 
used = [False] * total_seqs

print "Randomly choosing lines..."
# Choose selected lines
count = 0
while (count < total_toget):

	chosen = randint(0, total_seqs-1)
	#print chosen
	while (used[chosen] == True):
		#print chosen
		chosen = randint(0, total_seqs-1)

	used[chosen] = True
	count += 1

randfilename1 = "rand1_" + toget_str + "M.seq"
randfile1 = open(randfilename1, "w")
readfile1 = open("seq1.fastq", "r")
print "Read/Writing to " + randfilename1 + "..."
for i, line in enumerate(readfile1):
	if  (used[i/4] == True):
		randfile1.write(line)
randfile1.close()
readfile1.close()


randfilename2 = "rand2_" + toget_str + "M.seq"
randfile2 = open(randfilename2, "w")
readfile2 = open("seq2.fastq", "r")
print "Read/Writing to " + randfilename2 + "..."
for i, line in enumerate(readfile2):
	if  (used[i/4] == True):
		randfile2.write(line)
randfile2.close()
readfile2.close()

print "Done."
