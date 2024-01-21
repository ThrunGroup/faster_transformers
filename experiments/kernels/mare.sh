#!/bin/bash
for r in 32 64 128; do
    for c in 1 2 4 8 16 32 64; do
        python /home/zhongming/genghan/mare/mare_profile.py --rank $r --compress $c --record 0
    done
done 