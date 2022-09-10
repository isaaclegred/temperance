#!/bin/bash
# Use this file when you want to write garbage code
relative_start=0
# If we have access to a lot of procs we can make this smaller
# Total_dirs = num_procs * dirs_per_proc
dirs_to_make=1
global_start=0
dbdir="."
eosperdir=1000
((start_index=relative_start + global_start))
(( total_eos_to_make=dirs_to_make*eosperdir ))
(( initeosnum=start_index*eosperdir*dirs_to_make ))



for eos in $(seq $initeosnum $(($initeosnum+$total_eos_to_make-1)))
do
        echo $eos
        dirnum=$(($eos/$eosperdir))
        printf -v dirlabel "%06d" $dirnum
        printf -v eoslabel "%06d" $eos

        getnsprops eos-draw-$eoslabel.csv -v -p R,M,Mb -m 3e6 -d $dbdir/DRAWmod$eosperdir-$dirlabel/ -o $dbdir/DRAW\
mod$eosperdir-$dirlabel/

        splitbranches macro-draw-$eoslabel.csv -v -d $dbdir/DRAWmod$eosperdir-$dirlabel/ -o $dbdir/DRAWmod$eosperdir-$dirlab\
el/ -f MACROdraw-$eoslabel -t ""

        #plotprops macro-draw-$eoslabel.csv -v -p rhoc,R,Lambda,I,Mb -d $dbdir/DRAWmod$eosperdir-$dirlabel/ -o $dbdir/DRAWmo\
d$eosperdir-$dirlabel/MACROdraw-$eoslabel/ -t draw-$eoslabel
done
