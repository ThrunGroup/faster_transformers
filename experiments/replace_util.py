import transformers
from pathlib import Path
import os
import argparse
import subprocess

def replace_ori_with_time(trans_path, time_path):
    cmd = ["cp", time_path, trans_path]
    subprocess.run(cmd)

def backup_ori(backup_path, trans_path):
    cmd = ["cp", trans_path, backup_path]
    subprocess.run(cmd)

def restore_ori(backup_path, trans_path):
    cmd = ["cp", backup_path, trans_path]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/")
    parser.add_argument("--function", type=str, default="None")
    args = parser.parse_args()
    time_path = os.path.join(args.root, 'experiments', 'time_util_nofile.py')
    trans_path = Path(transformers.__file__).parent
    util_path = os.path.join(trans_path, 'generation/utils.py')
    backup_path = os.path.join(args.root, 'experiments', 'ori_util.py')
    if args.function == "backup":
        backup_ori(backup_path, trans_path)
        replace_ori_with_time(trans_path, time_path)
    elif args.function == "restore":
        restore_ori(backup_path, trans_path)
    else:
        print("Please input: backup or restore")