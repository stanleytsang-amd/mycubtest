#!/bin/bash
#input='test.input'
input='iter2-25-cub.size.sort.uniq'

if [ -z "$1" ]
then 
    echo "Please specify the 'rocm' or 'cuda'."
    exit
fi

rm $1.csv

IFS=$'\n' read -d '' -r -a lines < ${input}

progressbar()
{
    bar="##################################################"
    barlength=${#bar}
    n=$(($1*barlength/$2))
    printf "\r[%-${barlength}s (%d%%)] " "${bar:0:n}" "$[$1*100/$total]" 
}

#i=1
#total=${#lines[@]}
#sleep 1
#echo 'Total # of testcases': $total
#for line in "${lines[@]}"
#do
#    #echo $line
#    ./sort_float_rocm.exe 0 $line $line 1024 >> rocm.csv
#    progressbar $i $total
#    i=$(($i+1))
#done
#echo '\n'
#sleep 1


total=${#lines[@]}
for i in $(seq 1 $[total/100] ${total});
do
#echo $i
    ./sort_float_$1.exe 0 ${lines[$i]} ${lines[$i]} 1024 >> $1.csv
    progressbar $i $total
#    i=$(($i+100))
done
echo '\n'
sleep 1
