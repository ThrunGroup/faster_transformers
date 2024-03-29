# python bench_step_inference.py --B 1 --bm 2 --gen 513 --method 0
# python bench_step_inference.py --B 1 --bm 2 --gen 1025 --method 0
# python bench_step_inference.py --B 1 --bm 2 --gen 2049 --method 0
# python bench_step_inference.py --B 1 --bm 3 --gen 513 --method 0
# python bench_step_inference.py --B 1 --bm 3 --gen 1025 --method 0
# python bench_step_inference.py --B 1 --bm 3 --gen 2049 --method 0
# python bench_step_inference.py --B 1 --bm 4 --gen 513 --method 0
# python bench_step_inference.py --B 1 --bm 4 --gen 1025 --method 0
# python bench_step_inference.py --B 1 --bm 4 --gen 2049 --method 0
# python bench_step_inference.py --B 1 --bm 1 --gen 257 --method 0
# python bench_step_inference.py --B 1 --bm 2 --gen 257 --method 0
# python bench_step_inference.py --B 1 --bm 3 --gen 257 --method 0
# python bench_step_inference.py --B 1 --bm 4 --gen 257 --method 0
python bench_dense_inference.py --B 1 --bm 1 --gen 257
python bench_dense_inference.py --B 1 --bm 1 --gen 513
python bench_dense_inference.py --B 1 --bm 1 --gen 1025
python bench_dense_inference.py --B 1 --bm 1 --gen 2049
python /sailhome/zgh23/keep_gpu.py
