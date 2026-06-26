import sys
import os

ROOT_DIR = os.environ.get("PHYSGM_ROOT", "/root/PhysGM")
gaussian_splatting_path = os.path.join(ROOT_DIR, "gaussian-splatting")
sys.path.insert(0, gaussian_splatting_path)
import imageio_ffmpeg
import argparse
import math
import cv2
import torch
import numpy as np
import json
from tqdm import tqdm
from torchvision import transforms
from PIL import Image


# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.warp_utils import MPMStateStruct, MPMModelStruct
from mpm_solver_warp.mpm_utils import get_float_array_product
from scipy.spatial import cKDTree
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Util
from util.decode_param import *
from util.transformation_utils import *
from util.camera_view_utils import *
from util.render_utils import *

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["FFMPEG_BINARY"] = ffmpeg_path

wp.init()
wp.config.verify_cuda = True


@wp.kernel
def apply_particle_material_arrays(
    state: MPMStateStruct,
    model: MPMModelStruct,
    E_array: wp.array(dtype=float),
    nu_array: wp.array(dtype=float),
    density_array: wp.array(dtype=float),
    material_id_array: wp.array(dtype=int),
):
    p = wp.tid()
    model.E[p] = E_array[p]
    model.nu[p] = nu_array[p]
    state.particle_density[p] = density_array[p]
    model.material_id[p] = material_id_array[p]


def _as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def load_aligned_gaussian_part_ids(part_ids_path, raw_pos, pipeline, source_model_path=None):
    part_ids = np.load(part_ids_path).astype(np.int32)
    raw_np = _as_numpy(raw_pos)
    if len(part_ids) == len(raw_np):
        print(f"Loaded direct Gaussian part ids: {part_ids_path} ({len(part_ids)})")
        return part_ids
    if source_model_path is None:
        raise ValueError(
            f"part id count {len(part_ids)} does not match current Gaussian count {len(raw_np)}; "
            "provide --part_id_source_model_path for nearest-neighbor alignment."
        )
    print(f"Aligning current Gaussians to source part ids via xyz nearest neighbor: {source_model_path}")
    source_gaussians = load_checkpoint(source_model_path)
    source_params = load_params_from_gs(source_gaussians, pipeline)
    source_pos = _as_numpy(source_params["pos"])
    if len(source_pos) != len(part_ids):
        raise ValueError(
            f"source Gaussian count {len(source_pos)} does not match part id count {len(part_ids)}"
        )
    tree = cKDTree(source_pos)
    dist, nn = tree.query(raw_np, k=1, workers=-1)
    print(
        "Gaussian id alignment distance:",
        f"max={float(np.max(dist)):.6g}",
        f"mean={float(np.mean(dist)):.6g}",
        f"p99={float(np.percentile(dist, 99)):.6g}",
    )
    return part_ids[nn].astype(np.int32)


def resolve_unknown_part_ids(surface_pos_np, surface_part_ids):
    ids = surface_part_ids.astype(np.int32).copy()
    unknown = ids < 0
    known = ~unknown
    if np.any(unknown) and np.any(known):
        tree = cKDTree(surface_pos_np[known])
        _, nn = tree.query(surface_pos_np[unknown], k=1, workers=-1)
        ids[unknown] = ids[known][nn]
        print(f"Resolved {int(np.sum(unknown))} unknown Gaussian part ids by nearest known Gaussian")
    return ids


def build_particle_part_ids(transformed_pos, mpm_init_pos, gaussian_part_ids):
    surface_pos_np = _as_numpy(transformed_pos).astype(np.float32)
    particle_pos_np = _as_numpy(mpm_init_pos).astype(np.float32)
    surface_ids = resolve_unknown_part_ids(surface_pos_np, gaussian_part_ids)
    gs_num = len(surface_ids)
    if len(particle_pos_np) < gs_num:
        raise ValueError("MPM particle count is smaller than Gaussian particle count")
    particle_part_ids = np.empty(len(particle_pos_np), dtype=np.int32)
    particle_part_ids[:gs_num] = surface_ids
    extra = len(particle_pos_np) - gs_num
    if extra > 0:
        tree = cKDTree(surface_pos_np)
        dist, nn = tree.query(particle_pos_np[gs_num:], k=1, workers=-1)
        particle_part_ids[gs_num:] = surface_ids[nn]
        print(
            f"Assigned {extra} filled/internal particles by nearest Gaussian part id; "
            f"mean_dist={float(np.mean(dist)):.6g}, p99_dist={float(np.percentile(dist, 99)):.6g}"
        )
    unique, counts = np.unique(particle_part_ids, return_counts=True)
    print("Particle part id counts:", {int(k): int(v) for k, v in zip(unique, counts)})
    return particle_part_ids


MATERIAL_NAME_TO_ID = {
    "jelly": 0,
    "metal": 1,
    "sand": 2,
    "foam": 3,
    "snow": 4,
    "plasticine": 5,
}


def material_to_id(material):
    if isinstance(material, (int, np.integer)):
        mid = int(material)
        if mid not in MATERIAL_NAME_TO_ID.values():
            raise ValueError(f"Unknown material id: {material}")
        return mid
    name = str(material).strip().lower()
    if name not in MATERIAL_NAME_TO_ID:
        raise ValueError(f"Unknown material name: {material}")
    return MATERIAL_NAME_TO_ID[name]


def material_name(material):
    if isinstance(material, (int, np.integer)):
        mid = material_to_id(material)
        for name, value in MATERIAL_NAME_TO_ID.items():
            if value == mid:
                return name
    return str(material).strip().lower()


def load_part_material_table(path, material_params):
    fallback = {
        "name": "fallback_global",
        "material": material_name(material_params.get("material", "jelly")),
        "E": float(material_params.get("E", 1e5)),
        "nu": float(material_params.get("nu", 0.4)),
        "density": float(material_params.get("density", 200.0)),
    }
    table = {}
    if path is None:
        fallback["material_id"] = material_to_id(fallback["material"])
        return table, fallback
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "fallback" in data:
        fallback.update({
            k: data["fallback"][k]
            for k in ("E", "nu", "density", "material")
            if k in data["fallback"]
        })
        if "name" in data["fallback"]:
            fallback["name"] = data["fallback"]["name"]
    fallback["material"] = material_name(fallback["material"])
    fallback["material_id"] = material_to_id(fallback["material"])
    parts = data.get("parts", data)
    for key, value in parts.items():
        if key == "fallback":
            continue
        pid = int(key)
        merged = dict(fallback)
        merged.update(value)
        mat_name = material_name(merged.get("material", fallback["material"]))
        table[pid] = {
            "name": str(merged.get("name", f"part_{pid}")),
            "material": mat_name,
            "material_id": material_to_id(mat_name),
            "E": float(merged["E"]),
            "nu": float(merged["nu"]),
            "density": float(merged["density"]),
            "rigid_project": bool(merged.get("rigid_project", False)),
            "rigid_project_strength": float(merged.get("rigid_project_strength", 1.0)),
            "interface_bond": bool(merged.get("interface_bond", False)),
            "interface_bond_radius": float(merged.get("interface_bond_radius", 0.035)),
            "interface_bond_strength": float(merged.get("interface_bond_strength", 0.75)),
            "interface_bond_velocity_blend": float(merged.get("interface_bond_velocity_blend", 0.75)),
            "interface_bond_max_particles": int(merged.get("interface_bond_max_particles", 25000)),
        }
    return table, fallback


def apply_direct_part_materials(mpm_solver, particle_part_ids, materials_json, material_params, output_path, device):
    table, fallback = load_part_material_table(materials_json, material_params)
    n = len(particle_part_ids)
    E = np.full(n, float(fallback["E"]), dtype=np.float32)
    nu = np.full(n, float(fallback["nu"]), dtype=np.float32)
    density = np.full(n, float(fallback["density"]), dtype=np.float32)
    material_ids = np.full(n, int(fallback["material_id"]), dtype=np.int32)
    summary = {
        "mode": "direct_particle_part_ids",
        "materials_json": materials_json,
        "fallback": fallback,
        "parts": {},
    }
    for pid, params in sorted(table.items()):
        mask = particle_part_ids == pid
        E[mask] = params["E"]
        nu[mask] = params["nu"]
        density[mask] = params["density"]
        material_ids[mask] = params["material_id"]
        summary["parts"][str(pid)] = dict(params, particle_count=int(np.sum(mask)))
    unmatched = sorted(set(int(x) for x in np.unique(particle_part_ids)) - set(table.keys()))
    summary["unmatched_part_ids_using_fallback"] = unmatched
    E_wp = wp.from_numpy(E, dtype=float, device=device)
    nu_wp = wp.from_numpy(nu, dtype=float, device=device)
    density_wp = wp.from_numpy(density, dtype=float, device=device)
    material_id_wp = wp.from_numpy(material_ids, dtype=int, device=device)
    wp.launch(
        kernel=apply_particle_material_arrays,
        dim=mpm_solver.n_particles,
        inputs=[mpm_solver.mpm_state, mpm_solver.mpm_model, E_wp, nu_wp, density_wp, material_id_wp],
        device=device,
    )
    wp.launch(
        kernel=get_float_array_product,
        dim=mpm_solver.n_particles,
        inputs=[mpm_solver.mpm_state.particle_density, mpm_solver.mpm_state.particle_vol, mpm_solver.mpm_state.particle_mass],
        device=device,
    )
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        np.save(os.path.join(output_path, "particle_part_ids.npy"), particle_part_ids)
        with open(os.path.join(output_path, "particle_material_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print("Applied direct per-particle part materials")
    print(json.dumps(summary, indent=2))
    return table, fallback


class RigidPartProjector:
    def __init__(self, init_pos, particle_part_ids, material_table, output_path=None, device="cuda:0"):
        self.groups = []
        part_ids = torch.as_tensor(particle_part_ids, dtype=torch.long, device=device)
        init_pos = init_pos.detach().to(device=device)
        summary = {"mode": "frame_level_translation_shape_projection", "parts": {}}
        for pid, params in sorted(material_table.items()):
            if not bool(params.get("rigid_project", False)):
                continue
            mask = part_ids == int(pid)
            count = int(mask.sum().item())
            if count <= 0:
                continue
            part_init = init_pos[mask].clone()
            init_com = part_init.mean(dim=0)
            self.groups.append(
                {
                    "part_id": int(pid),
                    "name": params.get("name", f"part_{pid}"),
                    "mask": mask,
                    "offsets": part_init - init_com.reshape(1, 3),
                    "count": count,
                    "strength": min(max(float(params.get("rigid_project_strength", 1.0)), 0.0), 1.0),
                }
            )
            summary["parts"][str(pid)] = {
                "name": params.get("name", f"part_{pid}"),
                "particle_count": count,
                "material": params.get("material"),
                "E": params.get("E"),
                "strength": min(max(float(params.get("rigid_project_strength", 1.0)), 0.0), 1.0),
            }
        if output_path is not None and self.groups:
            with open(os.path.join(output_path, "rigid_projection_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        if self.groups:
            print("Rigid part projection enabled")
            print(json.dumps(summary, indent=2))

    def project(self, mpm_solver, device="cuda:0"):
        if not self.groups:
            return
        with torch.no_grad():
            x = mpm_solver.export_particle_x_to_torch().clone()
            v = mpm_solver.export_particle_v_to_torch().clone()
            F = mpm_solver.export_particle_F_to_torch().reshape(-1, 3, 3).clone()
            C = mpm_solver.export_particle_C_to_torch().reshape(-1, 3, 3).clone()
            eye = torch.eye(3, device=x.device, dtype=x.dtype).reshape(1, 3, 3)
            zero = torch.zeros((1, 3, 3), device=x.device, dtype=x.dtype)
            for group in self.groups:
                mask = group["mask"]
                current_com = x[mask].mean(dim=0)
                mean_v = v[mask].mean(dim=0)
                strength = float(group.get("strength", 1.0))
                target_x = current_com.reshape(1, 3) + group["offsets"]
                x[mask] = x[mask] * (1.0 - strength) + target_x * strength
                v[mask] = v[mask] * (1.0 - strength) + mean_v.reshape(1, 3) * strength
                F[mask] = F[mask] * (1.0 - strength) + eye * strength
                C[mask] = C[mask] * (1.0 - strength) + zero * strength
            mpm_solver.import_particle_x_from_torch(x, clone=False, device=device)
            mpm_solver.import_particle_v_from_torch(v, clone=False, device=device)
            mpm_solver.import_particle_F_from_torch(F, clone=False, device=device)
            mpm_solver.import_particle_C_from_torch(C, clone=False, device=device)


class BondedInterfaceProjector:
    def __init__(self, init_pos, particle_part_ids, material_table, output_path=None, device="cuda:0"):
        self.groups = []
        part_ids_np = np.asarray(particle_part_ids, dtype=np.int32)
        init_np = init_pos.detach().cpu().numpy().astype(np.float32)
        init_torch = init_pos.detach().to(device=device)
        part_ids_torch = torch.as_tensor(part_ids_np, dtype=torch.long, device=device)
        support_ids = {
            int(pid)
            for pid, params in material_table.items()
            if bool(params.get("interface_bond", False))
        }
        if not support_ids:
            return
        summary = {"mode": "initial_contact_bond_projection", "parts": {}}
        for pid in sorted(support_ids):
            params = material_table[pid]
            support_idx = np.where(part_ids_np == int(pid))[0]
            if len(support_idx) == 0:
                continue
            candidate_idx = np.where((part_ids_np >= 0) & (part_ids_np != int(pid)))[0]
            if len(candidate_idx) == 0:
                continue
            radius = max(1.0e-5, float(params.get("interface_bond_radius", 0.035)))
            strength = min(max(float(params.get("interface_bond_strength", 0.75)), 0.0), 1.0)
            velocity_blend = min(max(float(params.get("interface_bond_velocity_blend", strength)), 0.0), 1.0)
            max_particles = max(1, int(params.get("interface_bond_max_particles", 25000)))
            tree = cKDTree(init_np[support_idx])
            dist, _ = tree.query(init_np[candidate_idx], k=1, workers=-1)
            order = np.argsort(dist)
            near = candidate_idx[order][dist[order] <= radius]
            if len(near) > max_particles:
                near = near[:max_particles]
            if len(near) == 0:
                continue
            support_mask = part_ids_torch == int(pid)
            bond_mask = torch.zeros(len(part_ids_np), dtype=torch.bool, device=device)
            bond_mask[torch.as_tensor(near, dtype=torch.long, device=device)] = True
            support_init = init_torch[support_mask]
            support_init_com = support_init.mean(dim=0)
            bond_offsets = init_torch[bond_mask].clone() - support_init_com.reshape(1, 3)
            self.groups.append(
                {
                    "part_id": int(pid),
                    "name": params.get("name", f"part_{pid}"),
                    "support_mask": support_mask,
                    "bond_mask": bond_mask,
                    "bond_offsets": bond_offsets,
                    "strength": strength,
                    "velocity_blend": velocity_blend,
                    "count": int(len(near)),
                    "radius": radius,
                }
            )
            bonded_ids, bonded_counts = np.unique(part_ids_np[near], return_counts=True)
            summary["parts"][str(pid)] = {
                "name": params.get("name", f"part_{pid}"),
                "support_particle_count": int(len(support_idx)),
                "bond_particle_count": int(len(near)),
                "radius": radius,
                "strength": strength,
                "velocity_blend": velocity_blend,
                "bonded_part_counts": {str(int(k)): int(v) for k, v in zip(bonded_ids, bonded_counts)},
            }
        if output_path is not None and self.groups:
            with open(os.path.join(output_path, "interface_bond_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        if self.groups:
            print("Bonded interface projection enabled")
            print(json.dumps(summary, indent=2))

    def project(self, mpm_solver, device="cuda:0"):
        if not self.groups:
            return
        with torch.no_grad():
            x = mpm_solver.export_particle_x_to_torch().clone()
            v = mpm_solver.export_particle_v_to_torch().clone()
            for group in self.groups:
                support_mask = group["support_mask"]
                bond_mask = group["bond_mask"]
                if int(support_mask.sum().item()) == 0 or int(bond_mask.sum().item()) == 0:
                    continue
                support_com = x[support_mask].mean(dim=0)
                support_v = v[support_mask].mean(dim=0)
                target = support_com.reshape(1, 3) + group["bond_offsets"]
                strength = float(group["strength"])
                velocity_blend = float(group["velocity_blend"])
                x[bond_mask] = x[bond_mask] * (1.0 - strength) + target * strength
                v[bond_mask] = v[bond_mask] * (1.0 - velocity_blend) + support_v.reshape(1, 3) * velocity_blend
            mpm_solver.import_particle_x_from_torch(x, clone=False, device=device)
            mpm_solver.import_particle_v_from_torch(v, clone=False, device=device)


ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    direct_ply_path = os.path.join(model_path, "point_clouds.ply")
    if os.path.exists(direct_ply_path):
        print(f"Use Gaussian: {direct_ply_path}")
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(direct_ply_path)
        return gaussians

    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--part_ids", type=str, default=None)
    parser.add_argument("--part_id_source_model_path", type=str, default=None)
    parser.add_argument("--part_materials_json", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)

    cam_path = os.path.join(model_path, "cameras.json")
    if not os.path.exists(cam_path):
        default_cam_data = [
            {
                "id": 0,
                "img_name": "0001",
                "width": 800,
                "height": 800,
                "position": [1.0, 1.0, 1.0],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "fy": 400,
                "fx": 400,
            }
        ]
        with open(cam_path, "w", encoding="utf-8") as f:
            json.dump(default_cam_data, f, indent=4)

    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    gaussian_part_ids = None
    if args.part_ids is not None:
        gaussian_part_ids = load_aligned_gaussian_part_ids(
            args.part_ids, init_pos, pipeline, args.part_id_source_model_path
        )

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    if gaussian_part_ids is not None:
        gaussian_part_ids = gaussian_part_ids[mask.detach().cpu().numpy()]

    # print("Applying manual 90-degree clockwise rotation...")

    manual_degree = torch.tensor([90.0])
    manual_axis = [0]
    manual_rot_mat = generate_rotation_matrices(manual_degree, manual_axis)
    init_pos = apply_rotations(init_pos, manual_rot_mat)
    init_cov = apply_cov_rotations(init_cov, manual_rot_mat)

    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]
        if gaussian_part_ids is not None:
            gaussian_part_ids = gaussian_part_ids[mask.detach().cpu().numpy()]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(
        rotated_pos, preprocessing_params["scale"]
    )
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    particle_part_ids = None
    if gaussian_part_ids is not None:
        particle_part_ids = build_particle_part_ids(transformed_pos, mpm_init_pos, gaussian_part_ids)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
    mpm_init_cov[:gs_num] = init_cov
    shs = init_shs
    opacity = init_opacity
    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)
    part_material_table = None
    if particle_part_ids is not None:
        part_material_table, _ = apply_direct_part_materials(
            mpm_solver,
            particle_part_ids,
            args.part_materials_json,
            material_params,
            args.output_path,
            device,
        )

    rigid_projector = None
    bonded_interface_projector = None
    if particle_part_ids is not None and part_material_table:
        rigid_projector = RigidPartProjector(
            mpm_init_pos,
            particle_part_ids,
            part_material_table,
            output_path=args.output_path,
            device=device,
        )
        bonded_interface_projector = BondedInterfaceProjector(
            mpm_init_pos,
            particle_part_ids,
            part_material_table,
            output_path=args.output_path,
            device=device,
        )

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    # breakpoint()
    mpm_solver.finalize_mu_lam()

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )

        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
        if rigid_projector is not None:
            rigid_projector.project(mpm_solver, device=device)
        if bonded_interface_projector is not None:
            bonded_interface_projector.project(mpm_solver, device=device)

        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )

        if args.render_img:
            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)

            rendering, raddi = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            assert args.output_path is not None
            cv2.imwrite(
                os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )

    if args.render_img and args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f'"{ffmpeg_path}" -loglevel error -framerate {fps} -i "{args.output_path}/%04d.png" -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p "{args.output_path}/output.mp4"'
        )
