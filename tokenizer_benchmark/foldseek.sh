#!/bin/bash
#which foldseek

#foldseek -h
tmp_file="tmp_results.txt"
foldseek easy-search $1 $2 $tmp_file tmpFolder --format-output "ttmscore,lddt" --remove-tmp-files 1 --db-load-mode 2
cat $tmp_file > $3
