#!/bin/bash

# input

dirname=$1
draw_executable=$2
draws_per_dir=$3
dirs_per_proc=$4
num_procs=$5
rundir=$6

# compute needed quantities
# emergency lever to add more draws to an existing dir
global_start=0
repodir=$HOME"/parametric-eos-priors/"
# output

dagfile="${rundir}/solve-tov.dag"
logfile="${rundir}/solve-tov.in"
echo $repodir > $logfile
echo $rundir >> $logfile
echo $obslist >> $logfile

# write sub files

execs=(  "macrofy"  )
args=(  "\"\$(Process) $dirs_per_proc $global_start $dirname $draws_per_dir\"" )

for i in $(seq 0 $((${#execs[@]}-1)))
do
        execfile=${execs[$i]}
        subfile="${rundir}/${execfile}.sub"
        arg=${args[$i]}

        echo "universe = vanilla" > $subfile
        echo "executable = $repodir/bin/$execfile.sh" >> $subfile
        echo "arguments = $arg" >> $subfile
        echo "output = $rundir/$execfile.out" >> $subfile
        echo "error = $rundir/$execfile.err" >> $subfile
        echo "log = $rundir/$execfile.log" >> $subfile
        echo "getenv = True" >> $subfile
        echo "accounting_group = ligo.dev.o3.cbc.pe.lalinference" >> $subfile
        echo "accounting_group_user = $USER" >> $subfile
        echo "queue $num_procs" >> $subfile
done

# write dag file

echo "# solve_tov.dag, deploying to $rundir" > $dagfile

job=0
echo "JOB $job $rundir/macrofy.sub" >> $dagfile
echo "RETRY $job 4" >> $dagfile
while [ ! -f "$dagfile" ]
do
        sleep 10s
done

condor_submit_dag $dagfile
