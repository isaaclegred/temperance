#!/bin/bash
# Maybe this should be a python script instead
RUNDIR=$1
LABEL=$2
PRIOR=$3
EOSPERDIR=$4
DIRPERPROC=$5
NUMDIRS=$6

rm $RUNDIR/*
. make_draw_and_solve.sh "experimental_eos_draw_spectral" "spectral uniform" 100 1 2000 $RUNDIR
