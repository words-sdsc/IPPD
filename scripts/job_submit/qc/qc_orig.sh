"#!/bin/sh\n"+
"#SBATCH -J " + Jobname + "\n"+
"#SBATCH -e " + TrialErrDir + "/" + JobErrorFile + "_" + "%A" + "\n"+
"#SBATCH -o " + TrialErrDir + "/" + JobOutputFile + "_" + "%A" + "\n"+
"#SBATCH -A " + Account + "\n"+
"#SBATCH --nodes=" + NumOfNodes + "\n"+
"#SBATCH --ntasks-per-node=" + ppn +"\n"+
"#SBATCH --export=ALL\n"+
"#SBATCH -t " + Walltime + "\n"+
"#SBATCH -p " + QueueType +"\n"+
"#SBATCH --mem=" +  mem + "\n"+
"#SBATCH --mail-type=ALL\n"+
"#SBATCH --mail-user=\"" + Mail + "\"\n"+

"cd "+RemoteProjectDir+"\n"+
"cd "+SampleName+"\n"+
"mkdir -p " + TrialOutputDir + "\n"+

"/usr/bin/time -v "+RemoteScriptDir+"/NGS-qa-filter-v2.pl -i input/"+Input1+".fastq"+" -j input/"+Input2+".fastq -e 5 -N 0 -o " + TrialOutputDir 