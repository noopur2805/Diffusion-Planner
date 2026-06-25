"""
NavSim multi-modal dataset wrapping the existing vectorized data with 8 surround
camera images. Each ``__getitem__`` returns the same tuple as
``DiffusionPlannerData`` plus a ``(V, 3, H, W)`` image tensor and a ``(V,)``
mask indicating padded/missing views.

Expected on-disk layout (one entry per sample):
    <data_dir>/<token>.npz                       # existing vectorized scene
    <camera_dir>/<token>/{cam_f0,cam_l0,...}.jpg # 8 views, NavSim naming

The mapping ``token -> npz path`` is given by ``data_list`` (existing convention).
A ``camera_manifest`` JSON file (optional) overrides the default layout by
explicitly listing each view path per token.
"""
import os
import json
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from diffusion_planner.utils.train_utils import openjson, opendata
from diffusion_planner.model.module.camera_encoder import CameraEncoder


_DEFAULT_VIEWS = CameraEncoder.view_names()


class NavSimMultiModalData(Dataset):
    """Vectorized scene + 8 NavSim camera images."""

    def __init__(
        self,
        data_dir: str,
        data_list: str,
        past_neighbor_num: int,
        predicted_neighbor_num: int,
        future_len: int,
        camera_dir: Optional[str] = None,
        camera_manifest: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 480),
        views=_DEFAULT_VIEWS,
        normalize: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len

        self.camera_dir = camera_dir
        self.views = list(views)
        self.image_size = image_size

        self.manifest = None
        if camera_manifest is not None and os.path.exists(camera_manifest):
            with open(camera_manifest, "r") as fh:
                self.manifest = json.load(fh)

        transforms = [T.Resize(image_size), T.ToTensor()]
        if normalize:
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]))
        self.image_transform = T.Compose(transforms)

    def __len__(self):
        return len(self.data_list)

    def _token_from_path(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _load_view(self, token: str, view: str):
        """Return (image_tensor, is_valid)."""
        if self.manifest is not None and token in self.manifest and view in self.manifest[token]:
            path = self.manifest[token][view]
        elif self.camera_dir is not None:
            path = os.path.join(self.camera_dir, token, f"{view}.jpg")
        else:
            return torch.zeros(3, *self.image_size), False

        if not os.path.exists(path):
            return torch.zeros(3, *self.image_size), False
        try:
            img = Image.open(path).convert("RGB")
            return self.image_transform(img), True
        except Exception:
            return torch.zeros(3, *self.image_size), False

    def __getitem__(self, idx):
        rel = self.data_list[idx]
        data = opendata(os.path.join(self.data_dir, rel))

        tup = (
            data['ego_current_state'],
            data['ego_agent_future'],
            data['neighbor_agents_past'][:self._past_neighbor_num],
            data['neighbor_agents_future'][:self._predicted_neighbor_num],
            data['lanes'],
            data['lanes_speed_limit'],
            data['lanes_has_speed_limit'],
            data['route_lanes'],
            data['route_lanes_speed_limit'],
            data['route_lanes_has_speed_limit'],
            data['static_objects'],
        )

        token = self._token_from_path(rel)
        images, valids = [], []
        for v in self.views:
            img, ok = self._load_view(token, v)
            images.append(img)
            valids.append(ok)
        cam_images = torch.stack(images, dim=0)
        cam_mask = torch.tensor([not ok for ok in valids], dtype=torch.bool)

        return tup + (cam_images.numpy().astype(np.float32), cam_mask.numpy())
