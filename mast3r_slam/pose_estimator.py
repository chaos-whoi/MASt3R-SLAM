"""Optimization based on MASt3R-SLAM tracker.py. Refactored to only focus on 
pair-wise frame pose estimation (unscaled), to use for downstream reconstruction"""

import torch
import lietorch
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber


def lietorch_to_rigid_tensor(T):
    """Convert a lietorch Sim3 to a 4x4 Sim3 tensor, with scale separate."""
    if isinstance(T, lietorch.Sim3):
        scale = T.data.squeeze(0)[-1].clone()
        T.data.squeeze(0)[-1] = 1.0  # Set scale to 1
        return T.matrix().squeeze(0), scale

    else:
        raise TypeError("Input must be a lietorch Sim3 object")
    
def lietorch_to_tensor(T):
    """Convert a lietorch Sim3 to a 4x4 Sim3 tensor."""
    if isinstance(T, lietorch.Sim3):
        return T.matrix().squeeze(0)
    else:
        raise TypeError("Input must be a lietorch Sim3 object")

def solve(cfg, sqrt_info, r, J):
    whitened_r = sqrt_info * r
    robust_sqrt_info = sqrt_info * torch.sqrt(
        huber(whitened_r, k=cfg["huber"])
    )
    mdim = J.shape[-1]
    A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
    b = (robust_sqrt_info * r).view(-1, 1)  # z-h
    H = A.T @ A
    g = -A.T @ b
    cost = 0.5 * (b.T @ b).item()

    L = torch.linalg.cholesky(H, upper=False)
    tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

    return tau_j, cost

def get_points_poses(frame_f, frame_k, idx_f2k, img_size, calib=None, depth_eps=None):
    Xf = frame_f.X_canon
    Xk = frame_k.X_canon

    # Average confidence
    Cf = frame_f.get_average_conf()
    Ck = frame_k.get_average_conf()

    meas_k = None
    valid_meas_k = None

    if calib is not None:
        Xf = constrain_points_to_ray(img_size, Xf[None], calib).squeeze(0)
        Xk = constrain_points_to_ray(img_size, Xk[None], calib).squeeze(0)

        # Setup pixel coordinates
        uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
        uv_k = uv_k.view(-1, 2)
        meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
        # Avoid any bad calcs in log
        valid_meas_k = Xk[..., 2:3] > depth_eps
        meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

    return Xf[idx_f2k], Xk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

def opt_pose_ray_dist_sim3(cfg, Xf, Xk, Qk, valid):
    last_error = 0
    sqrt_info_ray = 1 / cfg["sigma_ray"] * valid * torch.sqrt(Qk)
    sqrt_info_dist = 1 / cfg["sigma_dist"] * valid * torch.sqrt(Qk)
    sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

    # Initialize transform
    T_CkCf = lietorch.Sim3.Identity(1, device=Xf.device, dtype=torch.float32)

    # Precalculate distance and ray for obs k
    rd_k = point_to_ray_dist(Xk, jacobian=False)

    old_cost = float("inf")
    for step in range(cfg["max_iters"]):
        Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
        rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
        # r = z-h(x)
        r = rd_k - rd_f_Ck
        # Jacobian
        J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

        tau_ij_sim3, new_cost = solve(cfg, sqrt_info, r, J)
        T_CkCf = T_CkCf.retr(tau_ij_sim3)

        if check_convergence(
            step,
            cfg["rel_error"],
            cfg["delta_norm"],
            old_cost,
            new_cost,
            tau_ij_sim3,
        ):
            break
        old_cost = new_cost

        if step == cfg["max_iters"] - 1:
            print(f"max iters reached {last_error}")

    return T_CkCf

def opt_pose_calib_sim3(
    cfg, Xf, Xk, Qk, valid, meas_k, valid_meas_k, K, img_size
):
    last_error = 0
    sqrt_info_pixel = 1 / cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
    sqrt_info_depth = 1 / cfg["sigma_depth"] * valid * torch.sqrt(Qk)
    sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

    # Initialize transform
    T_CkCf = lietorch.Sim3.Identity(1, device=Xf.device, dtype=torch.float32)

    old_cost = float("inf")
    for step in range(cfg["max_iters"]):
        Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
        pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
            Xf_Ck,
            K,
            img_size,
            jacobian=True,
            border=cfg["pixel_border"],
            z_eps=cfg["depth_eps"],
        )
        valid2 = valid_proj & valid_meas_k
        sqrt_info2 = valid2 * sqrt_info

        # r = z-h(x)
        r = meas_k - pzf_Ck
        # Jacobian
        J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

        tau_ij_sim3, new_cost = solve(cfg, sqrt_info2, r, J)
        T_CkCf = T_CkCf.retr(tau_ij_sim3)

        if check_convergence(
            step,
            cfg["rel_error"],
            cfg["delta_norm"],
            old_cost,
            new_cost,
            tau_ij_sim3,
        ):
            break
        old_cost = new_cost

        if step == cfg["max_iters"] - 1:
            print(f"max iters reached {last_error}")

    return T_CkCf

def process_matches(idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf,
                    frame_f, frame_k, mast3r_pose_cfg, mast3r_img_shape, calib=None):
    # Get rid of batch dim
    idx_f2k_0 = idx_f2k[0]
    valid_match_k = valid_match_k[0]

    Qk = torch.sqrt(Qff[idx_f2k_0] * Qkf)

    Xf, Xk, Cf, Ck, meas_k, valid_meas_k = get_points_poses(
        frame_f, frame_k, idx_f2k_0, mast3r_img_shape, calib=calib, depth_eps=mast3r_pose_cfg["depth_eps"]
    )
    return Xf, Xk, Cf, Ck, Qk, meas_k, valid_meas_k, valid_match_k


def estimate_mast3r_tf(Xf, Xk, Qk, valid_opt, meas_k, valid_meas_k, 
                       mast3r_pose_cfg, mast3r_img_shape, calib=None):
    """Estimate the transform of frame f with respect to frame k (T_CkCf)."""
    cfg = mast3r_pose_cfg

    try:
        if calib is None:
            T_CkCf = opt_pose_ray_dist_sim3(
                cfg, Xf, Xk, Qk, valid_opt,
            )
        else:
            T_CkCf = opt_pose_calib_sim3(
                cfg, Xf, Xk, Qk, valid_opt, meas_k, valid_meas_k, calib, mast3r_img_shape,
            )
    except Exception as e:
        print(f"Cholesky failed, did not estimate mast3r tf")
        return None
    
    # T_CkCf, scale = lietorch_to_rigid_tensor(T_CkCf)
    return lietorch_to_tensor(T_CkCf)

def construct_initial_tf(T_CkCf_mast3r, T_WCf_meas, T_WCk_meas):
    """First keyframe aligns with T_WCf_meas, with the scale set to the 
    ratio of the translation between T_WCf_meas and T_WCk_meas 
    and the translation in T_CkCf_mast3r

    Frame f is being added as a new keyframe, frame k is the previous 
    frame used as a reference for initialization. This is done to 
    avoid using mono-depth for initialization
    """
    # Remove batch dim 
    T_CkCf_mast3r = T_CkCf_mast3r.squeeze(0)
    T_WCf_meas = T_WCf_meas.squeeze(0)
    T_WCk_meas = T_WCk_meas.squeeze(0)

    T_CkCf_meas = T_WCk_meas.inverse() @ T_WCf_meas

    opt_R = T_WCf_meas[:3, :3]
    opt_T = T_WCf_meas[:3, 3:]

    mast3r_trans = torch.norm(T_CkCf_mast3r[:3, 3:])
    meas_trans = torch.norm(T_CkCf_meas[:3, 3:])
    opt_S = meas_trans / mast3r_trans

    T_WCf_opt = torch.eye(4, device=T_CkCf_mast3r.device, dtype=T_CkCf_mast3r.dtype)
    T_WCf_opt[:3, :3] = opt_R * opt_S
    T_WCf_opt[:3, 3:] = opt_T
    return T_WCf_opt.unsqueeze(0)



    # frame.T_WC = T_WCf

    #     # Use pose to transform points to update keyframe
    #     Xkk = T_CkCf.act(Xkf)
    #     keyframe.update_pointmap(Xkk, Ckf)
    #     # write back the fitered pointmap
    #     self.keyframes[len(self.keyframes) - 1] = keyframe

    #     # Keyframe selection
    #     n_valid = valid_kf.sum()
    #     match_frac_k = n_valid / valid_kf.numel()
    #     unique_frac_f = (
    #         torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
    #     )

    #     new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

    #     # Rest idx if new keyframe
    #     if new_kf:
    #         self.reset_idx_f2k()

    #     return (
    #         new_kf,
    #         [
    #             keyframe.X_canon,
    #             keyframe.get_average_conf(),
    #             frame.X_canon,
    #             frame.get_average_conf(),
    #             Qkf,
    #             Qff,
    #         ],
    #         False,
    #     )