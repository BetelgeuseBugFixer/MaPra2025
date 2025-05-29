#!/bin/bash
#which foldseek

#foldseek -h
tmp_file="tmp_results.txt"
foldseek easy-search $1 $2 $3 tmpFolder --format-output "ttmscore,lddt" --remove-tmp-files 1

