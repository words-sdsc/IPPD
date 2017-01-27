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
"cp " + realRemoteDir + "/LS002/dataFiles.qc/InputDir/seqlen" + seqlen + "/* " + TrialInputDir + "\n" + 
"cp " + realRemoteDir + "/largeFiles/file.15G " + RemoteProjectDir + "\n" +

"sync"+"\n"+

"numactl --membind=0 --physcpubind=0-" + ppn + " " +
"/usr/bin/time -v " + RemoteScriptDir + "/NGS-qa-filter-v2.pl -i " + TrialInputDir + "/" + Input1 + " -j " + TrialInputDir + "/" + Input2 + " -e 5 -N 0 -o " + TrialOutputDir 
+ " 2>> " + TrialErrDir + "/" + JobErrorFile + "_script_NGS-qa-filter-v2.pl_" + "$SLURM_JOBID " 
+ " 1>> " + TrialErrDir + "/" + JobOutputFile + "_script_NGS-qa-filter-v2.pl_" + "$SLURM_JOBID " + "\n"+

"cp -r " + TrialErrDir + "/* " + realRemoteErrorDir + "/." + "\n" +
"cp -r " + TrialOutputDir + "/* " + realRemoteOutputDir + "/." + "\n"
