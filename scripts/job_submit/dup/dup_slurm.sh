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
"mkdir -p " + RemoteRefDir + "\n" +
"mkdir -p " + TrialInputDir  + "\n"+
"mkdir -p " + TrialErrDir + "\n"+
"mkdir -p " + TrialOutputDir + "\n"+
"mkdir -p " + realRemoteErrorDir + "\n"+
"mkdir -p " + realRemoteOutputDir + "\n"+

"mkdir -p " + TrialReadSplitDir + "/split-1" + "\n"+
"mkdir -p " + TrialReadSplitDir + "/split-2" + "\n"+

"cp " + realRemoteDir +"/execFiles/bin/* " + RemoteToolDir + "\n"+
"cp " + realRemoteDir + "/" + SampleName + "/dataFiles.bowtie/OutputDir/seqlen" + seqlen + "/* " + TrialInputDir + "\n" + 
"cp " + realRemoteDir + "/largeFiles/file.15G " + RemoteProjectDir + "\n"+

"sync"+"\n"+

"numactl --membind=0 --physcpubind=0-" + ppn +" "  + 
"/usr/bin/time -v " + RemoteToolDir + "/cd-hit-dup -i " + Input1 + " -i2 " + Input2 + " -o " + TrialOutputDir + "/dup -u 50"
 + " 2>> " + TrialErrDir + "/" + JobErrorFile + "_cd-hit-dup_" + "$SLURM_JOBID " 
 + "1>> " + TrialErrDir + "/" + JobOutputFile + "_cd-hit-dup_" + "$SLURM_JOBID " + "\n"+


"numactl --membind=0 --physcpubind=0-" + ppn +" "  + 
"/usr/bin/time -v " + RemoteToolDir+"/cd-hit-dup-PE-out.pl -i " + Input1 + " -j " + Input2 + " -c " 
		    + TrialOutputDir + "/dup.clstr -o " 
		    + TrialOutputDir + "/uniq-1 -p "
		    + TrialOutputDir + "/uniq-2"
 + " 2>> " + TrialErrDir + "/" + JobErrorFile + "_cd-hit-dup-PE-out.pl_" + "$SLURM_JOBID " +
"1>> " + TrialErrDir + "/" + JobOutputFile + "_cd-hit-dup-PE-out.pl_" + "$SLURM_JOBID "  + "\n"+


"numactl --membind=0 --physcpubind=0-" + ppn +" "  + 
"/usr/bin/time -v " + RemoteToolDir + "/cd-hit-div.pl " + TrialOutputDir + "/uniq-1 " + TrialReadSplitDir + "/split-1/split " + NumOfSplit 
  + " 2>> " + TrialErrDir + "/" + JobErrorFile + "_cd-hit-div.pl_uniq-1_" + "$SLURM_JOBID " +
"1>> " + TrialErrDir + "/" + JobOutputFile + "_cd-hit-div.pl_" + "$SLURM_JOBID " + "\n"+


"numactl --membind=0 --physcpubind=0-" + ppn +" "  + 
"/usr/bin/time -v " + RemoteToolDir + "/cd-hit-div.pl " + TrialOutputDir + "/uniq-2 " + TrialReadSplitDir + "/split-2/split " + NumOfSplit 
  + " 2>> " + TrialErrDir + "/" + JobErrorFile + "_cd-hit-div.pl_uniq" + "$SLURM_JOBID " +
"1>> " + TrialErrDir + "/" + JobOutputFile + "_cd-hit-div.pl_" + "$SLURM_JOBID "  + "\n"+

"cp -r " + TrialErrDir + "/* " + realRemoteErrorDir + "/." + "\n" +
"cp -r " + TrialOutputDir + "/* " + realRemoteOutputDir + "/." + "\n"

