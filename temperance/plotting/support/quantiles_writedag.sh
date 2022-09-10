#!/usr/bin/bash

# input

rundir="$1"
tags="$2"
eos_dir_tags="$3"
eos_cs2c2_dir_tags="$4"
eos_per_dir="$5"
which_quantiles="$6"
outtag="$7"
logweight_args="$8"
dont_make_prior="$9"

echo "rundir is"
echo $rundir
echo $tags
if [ -d "$rundir" ]
then
    rm "$rundir"/*
else
    mkdir "$rundir"
fi

# Need all of the lists to match
pretags=($tags)
num_procs=${#pretags[@]}

# compute needed quantities
global_start=0
repodir=$HOME"/ParametricPaper/Analysis"
# output

dagfile="${rundir}/get_quantiles.dag"
logfile="${rundir}/get_quantiles.in"
echo $repodir > $logfile
echo $rundir >> $logfile
echo $obslist >> $logfile

# write sub files

execs=( "get_quantiles" )
args=( "\"\$(Process) '$tags' '$eos_dir_tags' '$eos_cs2c2_dir_tags' '$eos_per_dir' $which_quantiles $outtag '$logweight_args' $dont_make_prior\"" )

for i in $(seq 0 $((${#execs[@]}-1)))
do
        execfile=${execs[$i]}
        subfile="${rundir}/${execfile}.sub"
        arg=${args[$i]}

        echo "universe = vanilla" > $subfile
        echo "executable = $repodir/Utils/Plotting/$execfile.sh" >> $subfile
        echo "arguments = $arg" >> $subfile
        echo "output = $rundir/$execfile.out" >> $subfile
        echo "error = $rundir/$execfile.err" >> $subfile
        echo "log = $rundir/$execfile.log" >> $subfile
        echo "request_disk = 64K" >> $subfile
        echo "getenv = True" >> $subfile
        echo "accounting_group = ligo.dev.o3.cbc.pe.lalinference" >> $subfile
        echo "accounting_group_user = $USER" >> $subfile
        echo "queue $num_procs" >> $subfile
done

# write dag file

echo "# get_quantiles.dag, deploying to $rundir" > $dagfile

job=0
echo "JOB $job $rundir/get_quantiles.sub" >> $dagfile
echo "RETRY $job 1" >> $dagfile

while [ ! -f "$dagfile" ]
do
        sleep 10s
done

condor_submit_dag $dagfile
