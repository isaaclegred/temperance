RUNDIR="$HOME/parametric-eos-priors/rundrawandsolve_spectralexp/"
rm $RUNDIR/*
. make_draw_and_solve.sh "experimental_eos_draw_spectral" "spectral uniform" 100 1 2000 $RUNDIR
