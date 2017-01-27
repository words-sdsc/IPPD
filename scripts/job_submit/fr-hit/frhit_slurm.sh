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
"mkdir -p " + TrialOutputDir + "/seqlen" + "\n"+
"mkdir -p " + realRemoteErrorDir + "\n"+
"mkdir -p " + realRemoteOutputDir + "\n"+

"cp " + realRemoteDir + "/execFiles/bin/* " + RemoteToolDir + "\n" +
"cp " + realRemoteDir + "/scriptFiles/* " + RemoteScriptDir + "\n" +
"cp " + realRemoteDir + "/refFiles/ncbi-genomes/ncbi_genome_2013_0429/ref_all_2013_0429.nr80 " + RemoteRefDir + "\n" +
"cp -r " + realRemoteDir + "/" + SampleName + "/dataFiles.dup/OutputDir/seqlen" + seqlen + "/* " + TrialInputDir + "\n"+ 
"cp " + realRemoteDir + "/largeFiles/file.15G " + RemoteProjectDir + "\n" +

"sync"+"\n"+

"numactl --membind=0 --physcpubind=0-" + ppn + " " + 
"/usr/bin/time -v " + RemoteToolDir + "/fr-hit -T 0 -r 25 -m 60 -c 70 -p 8 -a " + Input1 + "/split-" + SplitNum +
" -d " + RemoteRefDir + "/ref_all_2013_0429.nr80 -o " + TrialOutputDir + "/split-" + SplitNum +"-out" 
+ " 2>> " + TrialErrDir + "/" + JobErrorFile + "_" + "$SLURM_JOBID " 
+ " 1>> " + TrialErrDir + "/" + JobOutputFile + "_" + "$SLURM_JOBID " + "\n"+

"cp -r " + TrialErrDir + "/* " + realRemoteErrorDir + "/." + "\n" +
"cp -r " + TrialOutputDir + "/* " + realRemoteOutputDir + "/." + "\n"
