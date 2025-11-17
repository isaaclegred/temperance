#!/bin/bash
eos=$1
outdir=$2
eos_dir="."
#python ../ns-struc/bin/getnsprops $eos -v -p R,M,Mb -m 3e6 -d $eos_dir -o $eos_dir
python ~/Research/universality/bin/integrate-tov $eos_dir"/"$eos 1e12 4.9e15 --outpath $eos_dir"/"macro-"${eos}"  --central-eos-column "baryon_density" --central-eos-column  "pressurec2" --formalism "logenthalpy_MRLambda"
