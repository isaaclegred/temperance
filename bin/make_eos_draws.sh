#! /bin/bash
dir_index=$1
dirs_to_make=$2
eos_per_dir=$3
eos_dir_name=$4
prior_info=$5
global_start=$6

((global_start_dir_index= global_start/eos_per_dir))
echo $global_start_dir_index
((min_index=global_start_dir_index + dir_index*dirs_to_make))
((max_index=min_index + dirs_to_make - 1))

eos_model=$(echo $prior_info| cut  -d " " -f1)
prior_tag=$(echo $prior_info| cut  -d " " -f2)
# Will probably want to add more executables (keep an eye on pathnames (better way to do this?))
model_executable=""
if [[ $eos_model == *"iecewise" ]]; then
    model_executable=$HOME"/parametric-eos-priors/draw_eos_piecewise.py"
elif [[ $eos_model == *"os" ]]; then
    model_executable=$HOME"/parametric-eos-priors/draw_eos_sos.py"     
else 
    model_executable=$HOME"/parametric-eos-priors/draw_eos_spectral.py"
fi
echo "using $model_executable"
mkdir ../eos_draws/$eos_dir_name
cd ../eos_draws/$eos_dir_name

for raw_index in $(seq $min_index $max_index)
do
    # Should just be able to do seq -f %06g $min_index $max_index
    # but for some reason this doesn't work here
    index=$(printf "%06d" $raw_index)
    dir=DRAWmod$eos_per_dir-$index
    mkdir $dir
    cd $dir
    # This is a bit of a problem, I don't really want to 
    # write code for python 2, but much of the code
    # I'm using is incompatible with python 3
    python $model_executable --num-draws $eos_per_dir --dir-index $index --prior-tag $prior_tag
    cd .. 
done
cd ..
