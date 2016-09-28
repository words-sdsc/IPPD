#!/usr/bin/python
'''
@author a3rao
takeRandom.py - generates randomized input files for MTGA workflow
last updated: 08.30.2016
'''

from random import randint
import sys

# Map of sample names to number of lines
sample_to_lines =   {
    'LS002' :  650829376
}

# argument checks 
if (len(sys.argv) != 5):
    print "Error - Invalid number of arguments."
    print "Usage: ./takeRandom [sample name] [starting number of lines to get (in Millions)] [ending number of lines] [increment/size]"
    print "Example: ./takeRandom LS002 5 40 5"
    sys.exit()

# obtain sequence count through file size
sample = sys.argv[1]                                ## Check dictionary for sample name
if (sample_to_lines.has_key(sample) == False):      ## Exit if cannot find sample
    print "Currently no support for sample " + sample + ", exiting"
    sys.exit()
else: 
    total_lines  = sample_to_lines.get(sample) 
    total_seqs  = total_lines / 4                   ## A sequence is composed of four lines

# range of seqlens to generate 
start_toget = int(sys.argv[2]) * 1000000            ## Starting number
end_toget = int(sys.argv[3]) * 1000000              ## Ending number
incr = int(sys.argv[4]) * 1000000                   ## Increment size
diff = end_toget - start_toget                      ## Calculate number of slots needed
slots = diff / incr + 1

## For now, assume diff will be perfect divisible
#if (end_toget > total_lines):                      ## Check for invalid arg
#   print "Lines to retrieve exceeds number of lines in file, exiting"
#   sys.exit()

# get the size (number of sequences) for each seqlen
sizes = [0] * slots                                 ## Polulate all the sizes slots
for i in range (0, slots): 
    sizes[i] = start_toget + i * incr 

# setting masks for each size
print "Setting up..." 
masks = [None] * slots                  ## Create empty list of lists
for i in range (0, slots):
    used = [False] * total_seqs         ## Add empty list of boolean Falses
    masks[i] = used                     

# creating masks for each size
print "Randomly choosing lines to include for each seqlen..."
for i in range (0, slots):
    print "... for size " + str(sizes[i] / 1000000)
    count = 0
    used = masks[i]                         ## Obtain mask to use 
    total_toget = sizes[i] 
    while (count < total_toget):            ## Choose random numbers without duplicates
    
        chosen = randint(0, total_seqs-1)
        while (used[chosen] == True):
            chosen = randint(0, total_seqs-1)
    
        used[chosen] = True                 ## Mark as used
        #print " " + str(chosen)
        count += 1

# write to file 1 
readfile1 = open("seqfiles.original/seq1.fastq", "r")                 ## Open Readfile
files = [None] * slots                              ## Create/Open files to write to
for i in range (0, slots):
    toget_str = str(sizes[i]/1000000)
    randfilename1 = "rand1_" + toget_str + "M.seq"
    randfile1 = open(randfilename1, "w")
    files[i] = randfile1

print "Read/Writing to file1..."
for j, line in enumerate(readfile1):        ## For each line, check it is in any of masks
    for i in range (0, slots):              ## if in mask, write it to corresponding file
        used = masks[i]
        if (used[j/4] == True):
            randfile1 = files[i]
            randfile1.write(line)

for i in range (0, slots):                  ## Close all opened files
    randfile1 = files[i]
    randfile1.close()
readfile1.close()


# write to file 2
readfile2 = open("seqfiles.original/seq2.fastq", "r")                 ## Open Readfile
files = [None] * slots                              ## Create/Open files to write to
for i in range (0, slots):
    toget_str = str(sizes[i]/1000000)
    randfilename2 = "rand2_" + toget_str + "M.seq"
    randfile2 = open(randfilename2, "w")
    files[i] = randfile2

print "Read/Writing to file2..."
for j, line in enumerate(readfile2):        ## For each line, check it is in any of masks
    for i in range (0, slots):              ## if in mask, write it to corresponding file
        used = masks[i]
        if (used[j/4] == True):
            randfile2 = files[i]
            randfile2.write(line)

for i in range (0, slots):                  ## Close all opened files
    randfile2 = files[i]
    randfile2.close()
readfile2.close()

print "Done."
