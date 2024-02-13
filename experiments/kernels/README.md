
# Before benchmark
Replace the https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py with one of these 4 util files to benchmark different cases.

ori_util: original

stop_util: don't stop until generating the max_length

time_util: benchmark the token generateion stage

time_stop_util: don't stop until generating the max_length and benchmark the token generateion stage

# Threshold File:Density

threshold_1.json: 0.27

threshold_3.json: 0.6

threshold.json: 0.7

threshold_4.json: 0.8

# Instructions

On-the-fly pruning
```
python inference.py --context_length 64 --generation_length 1089 --sparse_mode 0 --ths_path thresholds_1.json --num_beams 2
```

Dense baseline
```
python inference.py --context_length 64 --generation_length 1089 --sparse_mode -1 --num_beams 2
```

Different beams share the same idx