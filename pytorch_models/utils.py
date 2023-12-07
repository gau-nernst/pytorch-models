import os

import torch


def torch_hub_download(url: str, subdir: str | None = None) -> str:
    save_dir = torch.hub.get_dir()
    if subdir is not None:
        save_dir = os.path.join(save_dir, subdir)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(url))

    if not os.path.exists(save_path):
        torch.hub.download_url_to_file(url, save_path)
    return save_path
