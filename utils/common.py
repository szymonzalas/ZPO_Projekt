import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import torch

load_dotenv()
neptune_project=os.getenv("neptune_project")
neptune_key=os.getenv("neptune_key")
neptune_log=False

if neptune_project==None or neptune_key==None:
    print("Neptune environment variables not found.")
else:
    neptune_log=True
    print("Neptune environment variables loaded.")

if not torch.cuda.is_available():
    device=torch.device("cpu")
    print("Current device:", device)
else:
    device=torch.device("cuda")
    print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))

print("Common loaded")

    