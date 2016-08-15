"#!/bin/sh\n"+
"#PBS -v Catalina_maxhops=None\n"+
"#PBS -v PATH\n"+
"#PBS -M " +mail+"\n"+
"#PBS -q " +queue+"\n"+
"#PBS -N " +jobname+"\n"+
"#PBS -e " +jobErrorFile+"\n"+
"#PBS -o " +jobOutputFile+"\n"+
"#PBS -V\n"+
"#PBS -l nodes="+NumOfNodes+":ppn="+ppn+",walltime="+walltime+",mem="+mem+"\n"+
"#PBS -A " +Account+"\n"+

"cd "+RemoteProjectDir+"\n"+
"cd "+SampleName+"\n"+
"mkdir -p frhit-genome\n"+
"mkdir -p frhit-genome/fr-hit-1\n"+
"time "+RemoteScriptDir+"/ann_batch_run_dir.pl --INDIR1=read-split/split-1 --OUTDIR1=frhit-genome/fr-hit-1 "+RemoteToolDir+"/fr-hit -T 16 -r 25 -m 60 -c 70 -p 8 -a {INDIR1} -d "+RemoteRefDir+"/ncbi-genomes/ref_all_2013_0429.nr80 -o {OUTDIR1}"