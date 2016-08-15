"#!/bin/sh\n"+
"#SBATCH -J " + Jobname + "\n"+
"#SBATCH -A " + Account + "\n"+
"#SBATCH --nodes=" + NumOfNodes + "\n"+
"#SBATCH --export=ALL\n"+
"#SBATCH -t " + Walltime + "\n"+
"#SBATCH -p " + QueueType +"\n"+
"#SBATCH --mail-type=ALL\n"+
"#SBATCH --mail-user=\"" + Mail + "\"\n"+

[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[

"cd /scratch/$USER/$SLURM_JOBID"+"\n"+

"mkdir -p " + TrialErrDir + "\n"+
"mkdir -p " + TrialOutputDir + "\n"+
"mkdir -p " + RemoteProjectDir  + "/NGS-ann-program" + "\n"+
"mkdir -p " + RemoteProjectDir  + "/NGS-ann-program/BIN" + "\n"+
"mkdir -p " + RemoteProjectDir + "/data" + "\n"+
"mkdir -p " + LocalRefDir + "\n" +

"cp " + realRemoteDir +"/execFiles/bin/* " + RemoteToolDir + "/."+"\n"+
"cp " + realRemoteDir + "/dataFiles/seqlen" + seqlen + "/* " + RemoteProjectDir + "/data/."+"\n"+ 
"cp " + realRemoteDir + "/largeFiles/file.15G " + RemoteProjectDir + "/."+"\n"+

"sync"+"\n"+

]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
"/usr/bin/time -v " + RemoteToolDir + "/bowtie -f -k 1 -v 2 -p 16 " + RemoteRefDir + "/human/human " + Input1 + " " + TrialOutputDir + "/human-hit-1\n"+

"/usr/bin/time -v "+RemoteToolDir+"/bowtie -f -k 1 -v 2 -p 16 "+RemoteRefDir+"/human/human " + Input2 + " " + TrialOutputDir + "/human-hit-2\n"+

"cp "  + TrialErrDir + "/* " + realRemoteOutputDir + "/." + "\n"
