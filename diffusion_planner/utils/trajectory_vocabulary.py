"""
Trajectory vocabulary utilities (DreamerAD eq. 9, 20, 21).

Two stages:
    1) ``build_vocabulary``: filter a corpus of GT future trajectories by
       end-state deviation from the *running* GT, then uniform-sample by
       lateral offset to keep a diverse set of K representative trajectories.
    2) ``gaussian_vocab_sample``: at training time, rank vocabulary entries by
       Mahalanobis distance to the policy's mean trajectory and return a
       mixed batch of (a) top-softmax discriminative samples and
       (b) Gaussian-neighborhood exploratory samples.
"""
from typing import Optional

import numpy as np
import torch


def _wrap(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def filter_by_endstate(
    candidates: torch.Tensor,        # (N, T, 3) -> x, y, heading
    gt: torch.Tensor,                # (T, 3) ground-truth reference
    x_thresh: float = 10.0,
    y_thresh: float = 5.0,
    theta_thresh: float = 20.0 * np.pi / 180.0,
) -> torch.Tensor:
    """Return a boolean mask over candidates."""
    dx = (candidates[:, -1, 0] - gt[-1, 0]).abs()
    dy = (candidates[:, -1, 1] - gt[-1, 1]).abs()
    dtheta = _wrap(candidates[:, -1, 2] - gt[-1, 2]).abs()
    return (dx <= x_thresh) & (dy <= y_thresh) & (dtheta <= theta_thresh)


def build_vocabulary(
    candidates: torch.Tensor,
    gt: torch.Tensor,
    K: int = 256,
    x_thresh: float = 10.0,
    y_thresh: float = 5.0,
    theta_thresh: float = 20.0 * np.pi / 180.0,
) -> torch.Tensor:
    """Filter, then uniformly sub-sample by lateral offset to get K trajectories.

    Args:
        candidates: (N, T, 3) tensor of candidate future trajectories.
        gt: (T, 3) reference trajectory used for filtering.
    Returns:
        vocab: (K, T, 3) tensor (K may be less if not enough candidates).
    """
    mask = filter_by_endstate(candidates, gt, x_thresh, y_thresh, theta_thresh)
    kept = candidates[mask]
    if kept.numel() == 0:
        return candidates[:1]  # degenerate fallback

    dy = (kept[:, -1, 1] - gt[-1, 1]).abs()
    order = torch.argsort(dy)
    kept = kept[order]
    if kept.shape[0] <= K:
        return kept
    idx = torch.linspace(0, kept.shape[0] - 1, steps=K).long()
    return kept[idx]


@torch.no_grad()
def gaussian_vocab_sample(
    vocab: torch.Tensor,            # (V, T, 3) shared across batch
    policy_traj: torch.Tensor,      # (B, T, 3) current mean policy trajectory
    g1: int = 8,                    # top-softmax samples for discrimination
    g2: int = 8,                    # neighborhood samples for exploration
    sigma_xy: float = 1.5,
    sigma_h: float = 0.2,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Returns (B, g1 + g2, T, 3) sampled trajectories from the vocabulary.

    Computes Mahalanobis distance between vocabulary entries and the policy
    trajectory under a diagonal Gaussian, then:
        - g1 entries via categorical sampling from softmax(-d / temperature)
        - g2 entries via top-g2 by smallest distance (deterministic neighbourhood)

    The first axis of the output is batch; entries are independently sampled
    per batch element.
    """
    B, T, _ = policy_traj.shape
    V = vocab.shape[0]
    sigma = torch.tensor([sigma_xy, sigma_xy, sigma_h], device=policy_traj.device)

    diff = vocab[None] - policy_traj[:, None]  # (B, V, T, 3)
    diff[..., 2] = _wrap(diff[..., 2])
    d = ((diff / sigma) ** 2).sum(dim=(-1, -2))  # (B, V)

    logits = -d / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    discrim_idx = torch.multinomial(probs, num_samples=g1, replacement=True)  # (B, g1)
    # neighborhood: smallest distance
    nbh_idx = torch.topk(-d, k=g2, dim=-1).indices  # (B, g2)
    idx = torch.cat([discrim_idx, nbh_idx], dim=-1)  # (B, g1+g2)

    gathered = vocab[idx]  # (B, G, T, 3)
    return gathered, idx, d


def total_reward_from_dense(
    dense_rewards: torch.Tensor,    # (B, T_h, K) sigmoid probs in [0,1]
    sigma: Optional[torch.Tensor] = None,             # (B, T_h, K) or (B, T_h)
    metric_weights: Optional[torch.Tensor] = None,    # (B, K)
    horizon_uncertainty_temp: float = 0.0,
    cumulative_uncertainty: bool = False,
    safety_idx=None,                # auto-detect from K when None
    task_idx=None,                  # auto-detect from K when None
    eps: float = 1e-3,
    gate_floor: float = 0.0,        # soften multiplicative safety gates
) -> torch.Tensor:
    """
    DreamerAD eqs. (16-19): log-sigmoid aggregation of safety + log of task sum.

    Extensions:
        ``metric_weights``: per-scene per-metric weights in (B, K). When provided,
            safety log-terms and task probabilities are multiplied by their
            corresponding weight. Pass ``None`` to recover the uniform-weight
            aggregator (default).
        ``sigma`` + ``horizon_uncertainty_temp``: when both are provided, the
            per-horizon contribution is damped by ``1 / (1 + tau_h * sigma_h)``,
            with ``sigma_h`` averaged over metrics if a per-(horizon, metric)
            sigma is given. Mirrors the per-candidate scaling used in GRPO but
            on the horizon axis. ``tau_h = 0`` recovers the original sum.
        ``cumulative_uncertainty``: when True, replace ``sigma_h`` with its
            running maximum along the horizon axis (``cummax``) before damping.
            Enforces a monotone-non-increasing damping factor: if the AD-RM is
            uncertain about horizon h, all later horizons h'>h are damped at
            least as strongly. Has no effect when ``sigma`` is None or
            ``horizon_uncertainty_temp = 0``.
        ``gate_floor``: in [0, 1). Softens the multiplicative safety gates by
            mapping each safety probability ``p`` to ``gate_floor + (1-gate_floor)*p``
            before the log. With ``gate_floor = 0`` (default) this is a no-op and
            recovers the original DreamerAD aggregation; with ``gate_floor > 0`` a
            candidate that fails a gate keeps a non-zero reward instead of
            collapsing to ``log(eps)``, which stabilises GRPO advantage variance
            when adding sparse PDMS gates (e.g. ``ego_is_making_progress``).

    Returns (B,) trajectory-level rewards (sum over horizons).
    """
    K = dense_rewards.shape[-1]
    if safety_idx is None or task_idx is None:
        if K == 8:
            safety_idx = (0, 1, 5, 6) if safety_idx is None else safety_idx  # nc, dac, ddc, mp
            task_idx = (2, 3, 4, 7) if task_idx is None else task_idx        # ttc, ep, comfort, sl
        elif K == 5:
            safety_idx = (0, 1, 2) if safety_idx is None else safety_idx     # nc, dac, ttc
            task_idx = (3, 4) if task_idx is None else task_idx              # ep, comfort
        else:
            raise ValueError(f"safety_idx/task_idx auto-detect needs K in {{5,8}}, got K={K}")
    safety = dense_rewards[..., list(safety_idx)]
    if gate_floor > 0.0:
        safety = gate_floor + (1.0 - gate_floor) * safety
    safety = safety.clamp(min=eps)
    task = dense_rewards[..., list(task_idx)].clamp(min=eps)

    if metric_weights is not None:
        ws = metric_weights[:, None, list(safety_idx)]
        wt = metric_weights[:, None, list(task_idx)]
        L = (ws * safety.log()).sum(dim=-1)
        S = (wt * task).sum(dim=-1).clamp(min=eps).log()
    else:
        L = safety.log().sum(dim=-1)                   # (B, T_h)
        S = task.sum(dim=-1).clamp(min=eps).log()      # (B, T_h)

    per_horizon = L + S                                # (B, T_h)

    if sigma is not None and horizon_uncertainty_temp > 0.0:
        sigma_h = sigma.mean(dim=-1) if sigma.dim() == per_horizon.dim() + 1 else sigma
        if cumulative_uncertainty:
            sigma_h = torch.cummax(sigma_h, dim=-1).values
        per_horizon = per_horizon / (1.0 + horizon_uncertainty_temp * sigma_h)

    return per_horizon.sum(dim=-1)                     # (B,)



class DynamicVocabulary:
    """
    Online vocabulary with utility-based eviction.

    The static ``build_vocabulary`` path is preserved; this is an opt-in
    container for cases where the vocabulary should evolve during training
    (e.g. GRPO adding the policy's own winning candidates and dropping
    stale entries the agent has stopped sampling from).

    Eviction policy:
        score = utility - age_decay * age
    The lowest-score entries are evicted when ``add`` would exceed capacity.
    ``utility`` is updated externally via ``update_utility`` (e.g. with a
    candidate's group advantage); ``age`` is incremented by ``tick`` once per
    GRPO step and reset to zero for any entry touched by ``update_utility``.

    Not a ``nn.Module`` - holds plain tensors so it can be checkpointed by
    saving ``state_dict()`` alongside model weights.
    """

    def __init__(
        self,
        capacity: int,
        T: int,
        traj_dim: int = 3,
        age_decay: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.T = int(T)
        self.traj_dim = int(traj_dim)
        self.age_decay = float(age_decay)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.buffer = torch.zeros(capacity, T, traj_dim, device=self.device)
        self.utility = torch.zeros(capacity, device=self.device)
        self.age = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @property
    def trajectories(self) -> torch.Tensor:
        """Return the currently valid prefix of the buffer."""
        return self.buffer[: self.size]

    def add(
        self,
        trajs: torch.Tensor,                       # (N, T, traj_dim)
        utilities: Optional[torch.Tensor] = None,  # (N,)
    ) -> torch.Tensor:
        """Add trajectories, evicting lowest-score entries if needed.

        Returns the buffer positions where the new trajectories were placed.
        """
        if trajs.dim() != 3 or trajs.shape[1] != self.T or trajs.shape[2] != self.traj_dim:
            raise ValueError(
                f"trajs must have shape (N, {self.T}, {self.traj_dim}), got {tuple(trajs.shape)}"
            )
        N = trajs.shape[0]
        if N > self.capacity:
            # Keep only the top-N candidates by their incoming utility.
            u_in = utilities if utilities is not None else torch.zeros(N, device=self.device)
            keep = torch.topk(u_in, k=self.capacity, largest=True).indices
            trajs = trajs[keep]
            utilities = u_in[keep]
            N = self.capacity

        u = utilities if utilities is not None else torch.zeros(N, device=self.device)

        needed = max(0, (self.size + N) - self.capacity)
        if needed > 0:
            valid_score = self.utility[: self.size] - self.age_decay * self.age[: self.size].float()
            keep_k = self.size - needed
            keep_idx = torch.topk(valid_score, k=keep_k, largest=True).indices
            keep_idx, _ = torch.sort(keep_idx)
            self.buffer[:keep_k] = self.buffer[: self.size][keep_idx]
            self.utility[:keep_k] = self.utility[: self.size][keep_idx]
            self.age[:keep_k] = self.age[: self.size][keep_idx]
            self.size = keep_k

        new_idx = torch.arange(self.size, self.size + N, device=self.device)
        self.buffer[self.size : self.size + N] = trajs
        self.utility[self.size : self.size + N] = u
        self.age[self.size : self.size + N] = 0
        self.size += N
        return new_idx

    def update_utility(self, indices: torch.Tensor, deltas: torch.Tensor) -> None:
        """Bump utility at ``indices`` by ``deltas`` and reset their age."""
        self.utility[indices] = self.utility[indices] + deltas
        self.age[indices] = 0

    def tick(self) -> None:
        """Increment the age of every valid entry by one."""
        if self.size > 0:
            self.age[: self.size] += 1

    def sample(
        self,
        policy_traj: torch.Tensor,
        g1: int = 8,
        g2: int = 8,
        sigma_xy: float = 1.5,
        sigma_h: float = 0.2,
        temperature: float = 1.0,
    ):
        """Sample ``g1 + g2`` trajectories from the current buffer."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty DynamicVocabulary.")
        return gaussian_vocab_sample(
            self.trajectories, policy_traj,
            g1=g1, g2=g2, sigma_xy=sigma_xy, sigma_h=sigma_h, temperature=temperature,
        )

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity, "T": self.T, "traj_dim": self.traj_dim,
            "age_decay": self.age_decay, "size": self.size,
            "buffer": self.buffer[: self.size].clone(),
            "utility": self.utility[: self.size].clone(),
            "age": self.age[: self.size].clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError(f"state size {size} exceeds capacity {self.capacity}")
        self.size = size
        self.buffer[:size] = state["buffer"].to(self.device)
        self.utility[:size] = state["utility"].to(self.device)
        self.age[:size] = state["age"].to(self.device)
