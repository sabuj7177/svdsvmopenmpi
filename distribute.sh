#!/bin/bash
filename=$1
n=$2
d=$3
nodes=$4
letters="abcdefghijklmnopqrstuvwxyz"
lines_file=$(( $n / $nodes ))
rm -rf data_part_*
rm -rf partitions
mkdir partitions
cp "$filename" partitions/"$filename"
cd partitions && split -l $lines_file $filename
count=0
term=$(( $nodes - 1))
#
for (( i=0; i<${#letters}; i++ )); do
	for (( k=0; k<${#letters}; k++ )); do
		file_part="x${letters:$i:1}${letters:$k:1}"
		mv "$file_part"  data_part_${count}.csv
		if (( $count == $term)); then
			break 2;
		else
			count=$(( $count + 1))
		fi
	done
done
rm -rf $filename
