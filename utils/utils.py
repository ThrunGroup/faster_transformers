import torch
import os
import socket
from datetime import datetime


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def set_master_port():
    free_port = find_free_port()
    os.environ["MASTER_PORT"] = str(free_port)


def is_running_deepspeed():
    return "LOCAL_RANK" in os.environ


def ds_print(*args, **kwargs):
    if is_running_deepspeed():
        if is_mainprocess():
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def is_mainprocess():
    return not is_running_deepspeed() or int(os.environ["LOCAL_RANK"]) == 0


def get_datetime(only_date: bool = False):
    now = datetime.now()
    if not only_date:
        return now.strftime("%Y-%m-%d %Hh%Mm%Ss")
    else:
        return now.strftime("%Y-%m-%d")
