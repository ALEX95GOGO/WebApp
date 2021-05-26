#!/bin/sh
#============ unzip the file ===========


Folder="/home/file/web/"
for file in ${Folder}/*;
do
    file_name=`basename $file`
    cd ${Folder}/$file_name
    (
        for unzip in ${Folder}/$file_name/*.zip;
        do
                unzip_name=`basename $unzip`
                rname="Task 123"
                unzip -o -O gbk  $unzip_name && mv `unzip -l $unzip_name | awk '{if(NR == 4){ print $4}}'` $rname
                # rm -f $unzip_name
        done
    )
done