#!/bin/bash
#which foldseek

#foldseek -h
foldseek easy-search $1 $2 $3 tmpFolder --format-output "query,target,ttmscore,lddt" --remove-tmp-files 1
