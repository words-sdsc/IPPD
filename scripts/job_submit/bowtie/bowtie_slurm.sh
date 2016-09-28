"#!/bin/sh\n"+
"#SBATCH -J " + Jobname + "\n"+
"#SBATCH -A " + Account + "\n"+
"#SBATCH --nodes=" + NumOfNodes + "\n"+
"#SBATCH --export=ALL\n"+
"#SBATCH -t " + Walltime + "\n"+
"#SBATCH -p " + QueueType +"\n"+
"#SBATCH --mail-type=ALL\n"+
"#SBATCH --mail-user=\"" + Mail + "\"\n"+

"cd /scratch/$USER/$SLURM_JOBID"+"\n"+

"mkdir -p " + RemoteToolDir  + "\n"+
"mkdir -p " + RemoteScriptDir  + "\n"+
"mkdir -p " + RemoteRefDir + "\n" +
"mkdir -p " + TrialInputDir  + "\n"+
"mkdir -p " + TrialErrDir + "\n"+
"mkdir -p " + TrialOutputDir + "\n"+
"mkdir -p " + realRemoteErrorDir + "\n"+
"mkdir -p " + realRemoteOutputDir + "\n"+

"cp " + realRemoteDir + "/execFiles/bin/* " + RemoteToolDir + "\n" +
"cp " + realRemoteDir + "/scriptFiles/* " + RemoteScriptDir + "\n" +
"cp -r " + realRemoteDir + "/refFiles/human " + RemoteRefDir + "\n" +
"cp " + realRemoteDir + "/" + SampleName + "/dataFiles.qc/OutputDir/seqlen" + seqlen + "/* " + TrialInputDir + "\n" + 
"cp " + realRemoteDir + "/largeFiles/file.15G " + RemoteProjectDir + "\n" +

"sync"+"\n"+

"numactl --membind=0 --physcpubind=0-" + ppn + " "  +
"/usr/bin/time -v " + RemoteToolDir + "/bowtie -f -k 1 -v 2 -p 16 " + RemoteRefDir + "/human/human " + Input1 + " " + TrialOutputDir + "/human-hit-1" 
+ " 2>> " + TrialErrDir + "/" + JobErrorFile + "_" + "$SLURM_JOBID " 
+ " 1>> " + TrialErrDir + "/" + JobOutputFile + "_" + "$SLURM_JOBID " + "\n"+

"numactl --membind=0 --physcpubind=0-" + ppn + " " +
"/usr/bin/time -v "+RemoteToolDir+"/bowtie -f -k 1 -v 2 -p 16 " + RemoteRefDir + "/human/human " + Input2 + " " + TrialOutputDir + "/human-hit-2"
+ " 2>> " + TrialErrDir + "/" + JobErrorFile + "_" + "$SLURM_JOBID " 
+ " 1>> " + TrialErrDir + "/" + JobOutputFile + "_" + "$SLURM_JOBID " + "\n"+

"cut -f 1 " + TrialOutputDir + "/human-hit-1 > " + TrialOutputDir + "/human-hit-1.ids\n"+
"cut -f 1 " + TrialOutputDir + "/human-hit-2 > " + TrialOutputDir + "/human-hit-2.ids\n"+
RemoteScriptDir + "/fasta_fetch_exclude_pair_ids.pl " 
+ TrialOutputDir + "/human-hit-1.ids " + TrialOutputDir + "/human-hit-2.ids " 
+ TrialInputDir + "/fltd-1 " + TrialInputDir + "/fltd-2 " 
+ TrialOutputDir + "/no-human-1 " + TrialOutputDir + "/no-human-2" + "\n" +

"cp -r " + TrialErrDir + "/* " + realRemoteErrorDir + "/." + "\n" +
"cp -r " + TrialOutputDir + "/* " + realRemoteOutputDir + "/." + "\n"