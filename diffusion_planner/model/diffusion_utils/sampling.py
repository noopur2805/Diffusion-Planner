from typing import Dict, Callable, Optional
import torch
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


def dpm_sampler(
        model: torch.nn.Module, 
        x_T, 
        other_model_params: Dict={}, 
        diffusion_steps=10,

        noise_schedule_params: Dict = {},
        model_wrapper_params: Dict = {},
        dpm_solver_params: Dict = {},
        sample_params: Dict = {}
    ):
    
    with torch.no_grad():
        noise_schedule = dpm.NoiseScheduleVP(
            schedule='linear',
            **noise_schedule_params
        )

        model_fn = dpm.model_wrapper(
            model,  # use your noise prediction model here
            noise_schedule,
            model_type=model.model_type,  # or "x_start" or "v" or "score"
            model_kwargs=other_model_params,
            **model_wrapper_params
        )

        dpm_solver = dpm.DPM_Solver(
            model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params) # w.o. dynamic thresholding

        # Steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.
        sample_dpm = dpm_solver.sample(
            x_T,
            steps=diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
            **sample_params
        )

    return sample_dpm


@torch.no_grad()
def shortcut_sampler(
        dit: torch.nn.Module,
        x_T: torch.Tensor,
        cross_c: torch.Tensor,
        route_lanes: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
        n_steps: int = 1,
        correcting_xt_fn: Optional[Callable] = None,
        sde=None,
        eps: float = 1e-3,
    ):
    """
    Multi-step sampler for a DiT trained with Shortcut Forcing.

    The model is queried with the chosen step size ``d = 1/n_steps`` and the
    requested signal level ``t``. It predicts ``x0`` directly (x_start
    parameterization); we then re-noise to the next time level using the
    SDE's ``marginal_prob`` and iterate.

    Args:
        dit: the ``DiT`` instance trained with ``use_shortcut=True``.
        x_T: (B, P, D_flat) initial noise + anchored current state slot.
        n_steps: number of model evaluations. Use 1 for max speed.
        sde: the diffusion SDE (used for marginal_prob during re-noising).
    """
    assert getattr(dit, "use_shortcut", False), "shortcut_sampler requires a DiT trained with use_shortcut=True"
    assert dit.model_type == "x_start", "shortcut_sampler requires diffusion_model_type='x_start'"
    if sde is None:
        sde = dit._sde

    B, P, _ = x_T.shape
    d = torch.full((B,), 1.0 / n_steps, device=x_T.device)

    # walk from t=1 down to t=eps in n_steps equal increments
    ts = torch.linspace(1.0 - eps, eps, n_steps + 1, device=x_T.device)
    xt = x_T

    for i in range(n_steps):
        t_cur = ts[i].expand(B)
        x0_pred = dit(xt, t_cur, cross_c, route_lanes, neighbor_current_mask, d=d).reshape(B, P, -1, 4)

        if i == n_steps - 1:
            xt_full = x0_pred
        else:
            t_next = ts[i + 1].expand(B)
            target_x0 = x0_pred[..., 1:, :]
            mean, std = sde.marginal_prob(target_x0, t_next)
            std = std.view(-1, *([1] * (target_x0.dim() - 1)))
            z = torch.randn_like(target_x0)
            xt_next_future = mean + std * z
            xt_full = torch.cat([x0_pred[:, :, :1, :], xt_next_future], dim=2)

        xt = xt_full.reshape(B, P, -1)
        if correcting_xt_fn is not None:
            xt = correcting_xt_fn(xt, ts[i + 1] if i + 1 < len(ts) else ts[-1], i)

    return xt
