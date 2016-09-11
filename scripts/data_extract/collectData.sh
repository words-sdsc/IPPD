#!/bin/bash

filename=$1
key=$2

#toreturn=`cat $filename | grep -m 1 "$key" | sed "s@\t\t*$key@@"`
toreturn=`cat $filename | grep -m 1 "$key" | sed "s/$key//g"`
echo -n $toreturn
