"""Mackey-Glass 数据集 + 共享底座 re-export。

``RNN`` / ``create_time_series_dataset`` / ``KL`` 已迁移到根级 ``fgl_common`` 包;
此处 re-export 以保持旧脚本(``from utils.utils import RNN, KL, ...``)向后兼容。

本文件保留 ``MackeyGlass`` 类 —— MG 专属的数据生成器,依赖 ``jitcdde``。
"""
import os as _os
import sys as _sys

import numpy as np
import torch
from jitcdde import jitcdde, y, t, jitcdde_lyap
from torch.utils.data import Dataset

# 让 ``import fgl_common`` 在旧脚本(sys.path 只加了 mackey_glass/)下也能工作
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from fgl_common.models import RNN                       # noqa: E402,F401
from fgl_common.data import create_time_series_dataset  # noqa: E402,F401
from fgl_common.distillation import KL                  # noqa: E402,F401


class MackeyGlass(Dataset):
    """Dataset for the Mackey-Glass task (data generation via jitcdde)."""

    def __init__(self,
                 tau,
                 constant_past,
                 nmg=10,
                 beta=0.2,
                 gamma=0.1,
                 dt=1.0,
                 splits=(8000., 2000.),
                 start_offset=0.,
                 seed_id=0,
                 ):
        """Args:
            tau (float): parameter of the Mackey-Glass equation
            constant_past (float): initial condition for the solver
            nmg (float): parameter of the Mackey-Glass equation
            beta (float): parameter of the Mackey-Glass equation
            gamma (float): parameter of the Mackey-Glass equation
            dt (float): time step length for sampling data
            splits (tuple): data split in time units for training and testing data
            start_offset (float): added offset of the starting point
            seed_id (int): seed for generating function solution
        """
        super().__init__()

        self.tau = tau
        self.constant_past = constant_past
        self.nmg = nmg
        self.beta = beta
        self.gamma = gamma
        self.dt = dt

        self.traintime = splits[0]
        self.testtime = splits[1]
        self.start_offset = start_offset
        self.seed_id = seed_id
        self.maxtime = self.traintime + self.testtime + self.dt

        self.traintime_pts = round(self.traintime / self.dt)
        self.testtime_pts = round(self.testtime / self.dt)
        self.maxtime_pts = self.traintime_pts + self.testtime_pts + 1  # eval one past the end

        self.mackeyglass_specification = [
            self.beta * y(0, t - self.tau) / (1 + y(0, t - self.tau) ** self.nmg) - self.gamma * y(0)
        ]

        self.generate_data()
        self.split_data()

    def generate_data(self):
        """Generate time-series using the provided parameters of the equation."""
        np.random.seed(self.seed_id)
        self.DDE = jitcdde_lyap(self.mackeyglass_specification)
        self.DDE.constant_past([self.constant_past])
        self.DDE.step_on_discontinuities()

        self.mackeyglass_soln = torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)
        lyaps = torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)
        lyaps_weights = torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)
        count = 0
        for time in torch.arange(self.DDE.t + self.start_offset,
                                 self.DDE.t + self.start_offset + self.maxtime,
                                 self.dt, dtype=torch.float64):
            value, lyap, weight = self.DDE.integrate(time.item())
            self.mackeyglass_soln[count, 0] = value[0]
            lyaps[count, 0] = lyap[0]
            lyaps_weights[count, 0] = weight
            count += 1

        self.total_var = torch.var(self.mackeyglass_soln[:, 0], True)
        self.lyap_exp = ((lyaps.T @ lyaps_weights) / lyaps_weights.sum()).item()

    def split_data(self):
        """Generate training and testing indices."""
        self.ind_train = torch.arange(0, self.traintime_pts)
        self.ind_test = torch.arange(self.traintime_pts, self.maxtime_pts - 1)

    def __len__(self):
        return len(self.mackeyglass_soln) - 1

    def __getitem__(self, idx):
        sample = torch.unsqueeze(self.mackeyglass_soln[idx, :], dim=0)
        target = self.mackeyglass_soln[idx + 1, :]
        return sample, target
