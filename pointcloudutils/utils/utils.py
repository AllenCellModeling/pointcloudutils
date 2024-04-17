import trimesh
import time
from im2mesh.utils.libmise.mise import MISE
from im2mesh.utils.libmcubes import marching_cubes
import numpy as np
import torch
from skimage.morphology import skeletonize_3d


def skeletonize(img):
    img_data = img
    if isinstance(img, dict):
        img_data = img["image"].detach().cpu().numpy()
    assert isinstance(img_data, np.ndarray)
    skeleton = torch.tensor(skeletonize_3d(img_data[0])).unsqueeze(dim=0).float()
    if isinstance(img, dict):
        img["image"] = skeleton
        return img
    return skeleton


def rescale_image(img):
    """
    Assumes 2 channel image where 0 is raw and 1 is seg
    'Raw' channels are stored with values between 0 and MAX_UINT16,
    where the 0-valued voxels denote the background. This function
    rescales the voxels become min-max scaled (between 0 and 1),
    also subsets to voxels within segmentation
    """

    _MAX_UINT16 = 65535
    img_data = img
    if isinstance(img, dict):
        img_data = img["image"]

    img_data = img_data.astype(np.float32)

    ix = 0
    ix_seg = 1

    raw = np.where(img_data[ix] >= 0, img_data[ix] / (_MAX_UINT16 - 1), 0)
    seg = img_data[ix_seg]
    final_img = torch.tensor(np.where(seg, raw, 0)).unsqueeze(dim=0)
    if isinstance(img, dict):
        img["image"] = final_img
        return img
    return final_img


def generate_from_latent(
    model, z, res=16, c=None, stats_dict={}, points_orig=None, **kwargs
):
    model = model.eval()

    threshold = np.log(0.5) - np.log(1.0 - 0.5)
    upsampling_steps = 3
    t0 = time.time()
    # Compute bounding box size
    box_size = 1 + 0.1
    device = model.device

    # Shortcut
    if upsampling_steps == 0:
        nx = 16
        pointsf = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (nx,) * 3)
        values = eval_points(model, pointsf, z, c, **kwargs).detach().cpu().numpy()
        values = values[0]
        value_grid = values.reshape(nx, nx, nx)
    else:
        mesh_extractor = MISE(res, 3, 0.5)

        points = mesh_extractor.query()

        while points.shape[0] != 0:
            # Query points
            pointsf = torch.FloatTensor(points).to(device)
            # Normalize to bounding box
            pointsf = pointsf / mesh_extractor.resolution
            pointsf = box_size * (pointsf - 0.5)

            if points_orig is not None:
                pointsf = points_orig
            # Evaluate model and update
            values = eval_points(model, pointsf, z, c, **kwargs).detach().cpu().numpy()
            values = values[0]
            values = values.astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        value_grid = mesh_extractor.to_dense()

    # Extract mesh
    stats_dict["time (eval points)"] = time.time() - t0

    mesh = extract_mesh(model, value_grid, z, c, stats_dict=stats_dict)
    return mesh, value_grid


def eval_points(model, p, z, c=None, **kwargs):
    """Evaluates the occupancy values for the points.

    Args:
        p (tensor): points
        z (tensor): latent code z
        c (tensor): latent conditioned code c
    """
    # batch_size = 1
    # p_split = torch.split(p, batch_size)
    # device = model.device
    # occ_hats = []

    # for pi in p_split:
    #     pi = pi.unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         occ_hat = model.decoder['inputs'](pi, z)

    #     occ_hats.append(occ_hat.squeeze(0).detach().cpu())

    # occ_hat = torch.cat(occ_hats, dim=0)
    with torch.no_grad():
        occ_hat = model.decoder["inputs"](p.unsqueeze(dim=0), z)

    return occ_hat


def extract_mesh(model, occ_hat, z, c=None, stats_dict=dict()):
    """Extracts the mesh from the predicted occupancy grid.

    Args:
        occ_hat (tensor): value grid of occupancies
        z (tensor): latent code z
        c (tensor): latent conditioned code c
        stats_dict (dict): stats dictionary
    """
    # Some short hands
    threshold = 0.5
    padding = 0.1
    with_normals = False
    simplify_nfaces = None

    n_x, n_y, n_z = occ_hat.shape
    box_size = 1 + padding
    threshold = np.log(threshold) - np.log(1.0 - threshold)
    # Make sure that mesh is watertight
    t0 = time.time()
    occ_hat_padded = np.pad(occ_hat, 1, "constant", constant_values=-1e6)
    vertices, triangles = marching_cubes(occ_hat_padded, threshold)
    stats_dict["time (marching cubes)"] = time.time() - t0
    # Strange behaviour in libmcubes: vertices are shifted by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
    vertices = box_size * (vertices - 0.5)

    # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
    # mesh_pymesh = fix_pymesh(mesh_pymesh)

    # Estimate normals if needed
    if with_normals and not vertices.shape[0] == 0:
        t0 = time.time()
        normals = estimate_normals(vertices, z, c)
        stats_dict["time (normals)"] = time.time() - t0

    else:
        normals = None

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=normals, process=False)

    # Directly return if mesh is empty
    if vertices.shape[0] == 0:
        return mesh

    # # TODO: normals are lost here
    # if simplify_nfaces is not None:
    #     t0 = time.time()
    #     mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
    #     stats_dict['time (simplify)'] = time.time() - t0

    # # Refine mesh
    # if refinement_step > 0:
    #     t0 = time.time()
    #     self.refine_mesh(mesh, occ_hat, z, c)
    #     stats_dict['time (refine)'] = time.time() - t0

    return mesh


def make_3d_grid(bb_min, bb_max, shape):
    """Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    """
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def estimate_normals(model, vertices, z, c=None):
    """Estimates the normals by computing the gradient of the objective.

    Args:
        vertices (numpy array): vertices of the mesh
        z (tensor): latent code z
        c (tensor): latent conditioned code c
    """
    device = model.device
    vertices = torch.FloatTensor(vertices)
    vertices_split = torch.split(vertices, self.points_batch_size)

    normals = []
    z, c = z.unsqueeze(0), c.unsqueeze(0)
    for vi in vertices_split:
        vi = vi.unsqueeze(0).to(device)
        vi.requires_grad_()
        occ_hat = model.decode(vi, z, c).logits
        out = occ_hat.sum()
        out.backward()
        ni = -vi.grad
        ni = ni / torch.norm(ni, dim=-1, keepdim=True)
        ni = ni.squeeze(0).cpu().numpy()
        normals.append(ni)

    normals = np.concatenate(normals, axis=0)
    return normals
