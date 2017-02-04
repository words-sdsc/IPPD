#!/usr/bin/env python

import sys
import csv

# check for filename being passed
if len(sys.argv) != 3 :
    print ('Usage: python timeConvert.py [input.csv] [output.csv]')
    sys.exit(1)

# Converts 'HH:MM:SS' time to seconds
def get_sec(time_str):

	timeSplit = time_str.split(':')
	if len(timeSplit) == 3:
		h, m, s = timeSplit
		return str(int(h) * 3600 + int(m) * 60 + int(s))
	elif len(timeSplit) == 2:
		m, s = timeSplit
		return str(int(m) * 60 + float(s))

	# Returns original string if invalid string
	return time_str	

# Note that this ASSUMES that WALLTIME is the final column... 
# Todo - do this independently of column index
def transform_row(row):
    return row[:-1] + [get_sec(row[-1])]

file_in = sys.argv[1]		# get input filename through argument
file_out = sys.argv[2]		# get output filename through argument

# read row by row, convert the walltime column...
with open(file_in, 'rt') as csv_in, open(file_out, 'wt') as csv_out:
    writer = csv.writer(csv_out)
    writer.writerows(transform_row(row) for row in csv.reader(csv_in))