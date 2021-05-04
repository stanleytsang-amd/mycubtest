#!/bin/bash
#input='test.input'
input='iter2-25-cub.size.sort'

progressbar()
{
    bar="##################################################"
    barlength=${#bar}
    n=$(($1*barlength/$2))
    printf "\r[%-${barlength}s (%d%%)] " "${bar:0:n}" "$[$1*100/$total]" 
}

if [ -z "$1" ]
then 
    echo "Please specify the 'rocm' or 'cuda'."
    exit
fi
if [ -z "$2" ]
then 
    echo "Please specify the 'output file name'."
    exit
fi

rm $2.csv

if [ -z "$3" ]
then
    echo "Pickup every #1%th record"
    uniq $input > $input.uniq
    IFS=$'\n' read -d '' -r -a lines < ${input}.uniq
    total=${#lines[@]}
    for i in $(seq 1 $[total/100] ${total});
    do
        ./sort_float_$1.exe 0 ${lines[$i]} ${lines[$i]} 1024 >> $2.csv
        progressbar $i $total
    done
else
    echo "Pickup Top $3 records"
    uniq -c $input > $input.uniq.count
    sort -r -k 1 -n $input.uniq.count > $input.uniq.count.sort
    IFS=$'\n' read -d '' -r -a lines < ${input}.uniq.count.sort
    total=$3
    for i in $(seq 1 ${total});
    do
        IFS=', ' read -r -a data <<< "${lines[$i]}"
        ./sort_float_$1.exe 0 ${data[1]} ${data[1]} 1024 >> $2.csv
        progressbar $i $total
    done
fi
sleep 1





