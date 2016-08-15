"#!/bin/sh\n"+
"#SBATCH -J " + Jobname + "\n"+
"#SBATCH -A " + Account + "\n"+
"#SBATCH --nodes=" + NumOfNodes + "\n"+
"#SBATCH --export=ALL\n"+
"#SBATCH -t " + Walltime + "\n"+
"#SBATCH -p " + QueueType +"\n"+
"#SBATCH --mail-type=ALL\n"+
"#SBATCH --mail-user=\"" + Mail + "\"\n"+

"cd /scratch/$USER/$SLURM_JOBID" + "\n"+

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
"cp " + realRemoteDir + "/dataFiles.dup/OutputDir/seqlen" + seqlen + "/* " + TrialInputDir + "\n" + 
"cp " + realRemoteDir + "/largeFiles/file.15G " + RemoteProjectDir + "\n" +

"sync"+"\n"+

"numactl --membind=0 --physcpubind=0-" + ppn + " " +
"/usr/bin/time -v "+ RemoteToolDir + "/velveth " + TrialOutputDir + "/assembly 55 -shortPaired -separate " + Input1 + " " + Input2
+ " 2>> " + TrialErrDir + "/" + JobErrorFile + "_velveth_" + "$SLURM_JOBID " 
+ " 1>> " + TrialErrDir + "/" + JobOutputFile + "_velveth_" + "$SLURM_JOBID " + "\n"+

"numactl --membind=0 --physcpubind=0-" + ppn + " " +
"/usr/bin/time -v "+ RemoteToolDir + "/velvetg " + TrialOutputDir + "/assembly -exp_cov auto -cov_cutoff auto -min_contig_lgth 100 -ins_length 202" 
+ " 2>> " + TrialErrDir + "/" + JobErrorFile + "_velvetg_" + "$SLURM_JOBID " 
+ " 1>> " + TrialErrDir + "/" + JobOutputFile + "_velvetg_" + "$SLURM_JOBID " + "\n"+

"cp -r " + TrialErrDir + "/* " + realRemoteErrorDir + "/." + "\n" +
"cp -r " + TrialOutputDir + "/* " + realRemoteOutputDir + "/." + "\n"
