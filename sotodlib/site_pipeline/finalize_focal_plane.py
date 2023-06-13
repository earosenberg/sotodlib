import os
import sys
from itertools import zip_longest
import argparse as ap
import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation as R
import yaml
import sotodlib.io.g3tsmurf_utils as g3u
from sotodlib.core import AxisManager, metadata, Context
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import util
from sotodlib.coords import focal_plane as fpc

logger = util.init_logger(__name__, "finalize_focal_plane: ")


def _avg_focalplane(fp_dict):
    focal_plane = []
    det_ids = np.array(list(fp_dict.keys()))
    for did in det_ids:
        avg_pointing = np.nanmedian(np.vstack(fp_dict[did]), axis=0)
        focal_plane.append(avg_pointing)
    focal_plane = np.column_stack(focal_plane)

    if np.isnan(focal_plane[:2]).all():
        raise ValueError("All detectors are outliers. Check your inputs")

    return det_ids, focal_plane


def _mk_fpout(det_id, focal_plane):
    outdt = [
        ("dets:det_id", det_id.dtype),
        ("xi", np.float32),
        ("eta", np.float32),
        ("gamma", np.float32),
    ]
    fpout = np.fromiter(zip(det_id, *focal_plane[:3]), dtype=outdt, count=len(det_id))

    return metadata.ResultSet.from_friend(fpout)


def _mk_tpout(shift, scale, shear, rot):
    outdt = [
        ("shift", np.float32),
        ("scale", np.float32),
        ("shear", np.float32),
        ("rot", np.float32),
    ]
    # rot will always have 3 values
    # so we can use to pad the others when we have no pol
    tpout = np.fromiter(
        zip_longest(shift, scale, shear, rot, fillvalue=np.nan), count=3, dtype=outdt
    )

    return metadata.ResultSet.from_friend(tpout)


def get_nominal(focal_plane, config):
    """
    Get nominal pointing from detector xy positions.

    Arguments:

        focal_plane: Focal plane array as generated by _avg_focalplane.

        config: Transformation configuration.
                Nominally config["coord_transform"].

    Returns:

        xi_nominal: The nominal xi values.

        eta_nominal: The nominal eta values.

        gamma_nominal: The nominal gamma values.
    """
    transform_pars = fpc.get_ufm_to_fp_pars(
        config["telescope"], config["slot"], config["config_path"]
    )
    x, y, pol = fpc.ufm_to_fp(
        None, x=focal_plane[3], y=focal_plane[4], pol=focal_plane[5], **transform_pars
    )
    if config["telescope"] == "LAT":
        xi_nominal, eta_nominal, gamma_nominal = fpc.LAT_focal_plane(
            None, config["zemax_path"], x, y, pol, config["rot"], config["tube"]
        )
    elif config["coord_transform"]["telescope"] == "SAT":
        xi_nominal, eta_nominal, gamma_nominal = fpc.SAT_focal_plane(None, x, y, pol)
    else:
        raise ValueError("Invalid telescope provided")

    return xi_nominal, eta_nominal, gamma_nominal


def get_affine(src, dst):
    """
    Get affine transformation between two point clouds.
    Transformation is dst = affine@src + shift

    Arguments:

        src: (ndim, npoints) array of source points.

        dst: (ndim, npoints) array of destination points.

    Returns:

        affine: The transformation matrix.

        shift: Shift to apply after transformation.
    """
    msk = np.isfinite(src).all(axis=0) * np.isfinite(dst).all(axis=0)
    if np.sum(msk) < 7:
        raise ValueError("Not enough finite points to compute transformation")

    M = np.vstack(
        (
            src[:, msk] - np.median(src[:, msk], axis=1)[:, None],
            dst[:, msk] - np.median(dst[:, msk], axis=1)[:, None],
        )
    ).T
    *_, vh = la.svd(M)
    vh_splits = [
        quad for half in np.split(vh.T, 2, axis=0) for quad in np.split(half, 2, axis=1)
    ]
    affine = np.dot(vh_splits[2], la.pinv(vh_splits[0]))

    transformed = affine @ src[:, msk]
    shift = np.median(dst[:, msk] - transformed, axis=1)

    return affine, shift


def decompose_affine(affine):
    """
    Decompose an affine transformation into its components.

    Arguments:

        affine: The affine transformation matrix.

    Returns:

        scale: Array of ndim scale parameters.

        shear: Array of shear parameters.

        rot: Rotation matrix.
             Not currently decomposed in this function because the easiest
             way to do that is not n-dimensional but this rest of this function is.
    """
    # Use the fact that rotation matrix times its transpose is the identity
    no_rot = affine.T @ affine
    # Decompose to get a matrix with just scale and shear
    no_rot = la.cholesky(no_rot).T

    scale = np.diag(no_rot)
    shear = (no_rot / scale[:, None])[np.triu_indices(len(no_rot), k=1)]
    rot = affine @ la.inv(no_rot)

    return scale, shear, rot


def decompose_rotation(rotation):
    """
    Decompose a rotation matrix into its angles.
    This currently won't work on anything higher than 3 dimensions.

    Arguments:

        rotation: (ndim, ndim) rotation matrix.

    Returns:

        angles: Array of rotation angles in radians.
                If the input is 2d then the first 2 angles will be nan.
    """
    ndim = len(rotation)
    if ndim > 3:
        raise ValueError("No support for rotations in more than 3 dimensions")
    if ndim < 2:
        raise ValueError("Rotations with less than 2 dimensions don't make sense")
    if rotation.shape != (ndim, ndim):
        raise ValueError("Rotation matrix should be ndim by ndim")
    _rotation = np.eye(3)
    _rotation[:ndim, :ndim] = rotation
    angles = R.from_matrix(_rotation).as_euler("xyz")

    if ndim == 2:
        angles[:2] = np.nan
    return angles


def main():
    # Read in input pars
    parser = ap.ArgumentParser()

    parser.add_argument("config_path", help="Location of the config file")
    args = parser.parse_args()

    # Open config file
    with open(args.config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Load context
    ctx = Context(config["context"]["path"])
    name = config["context"]["position_match"]
    query = []
    if "query" in config["context"]:
        query = (ctx.obsdb.query(config["context"]["query"])["obs_id"],)
    obs_ids = np.append(config["context"].get("obs_ids", []), query)
    # Add in manually loaded paths
    obs_ids = np.append(obs_ids, config.get("multi_obs", []))
    obs_ids = np.unique(obs_ids)
    if len(obs_ids) == 0:
        raise ValueError("No observations provided in configuration")

    # Build output path
    ufm = config["ufm"]
    append = ""
    if "append" in config:
        append = "_" + config["append"]
    os.makedirs(config["outdir"], exist_ok=True)
    outpath = os.path.join(config["outdir"], f"{ufm}{append}.h5")
    outpath = os.path.abspath(outpath)

    fp_dict = {}
    use_matched = "use_matched" in config and config["use_matched"]
    for obs_id, detmap in zip(obs_ids, config["detmaps"]):
        # Load data
        if os.path.isfile(obs_id):
            logger.info("Loading information from file at %s", obs_id)
            rset = read_dataset(obs_id, "focal_plane")
            _aman = rset.to_axismanager(axis_key="dets:readout_id")
            aman = AxisManager(_aman.dets)
            aman.wrap(name, _aman)
        else:
            logger.info("Loading information from observation %s", obs_id)
            aman = ctx.get_meta(obs_id, dets=config["context"].get("dets", {}))
        if name not in aman:
            logger.warning(
                "\tNo position_match associated with this observation. Skipping."
            )
            continue

        # Put SMuRF band channel in the correct place
        smurf = AxisManager(aman.dets)
        smurf.wrap("band", aman[name].band, [(0, smurf.dets)])
        smurf.wrap("channel", aman[name].channel, [(0, smurf.dets)])
        aman.det_info.wrap("smurf", smurf)

        if detmap is not None:
            g3u.add_detmap_info(aman, detmap)
        have_wafer = "wafer" in aman.det_info
        if not have_wafer:
            logger.error("\tThis observation has no detmap results, skipping")
            continue

        det_ids = aman.det_info.det_id
        x = aman.det_info.wafer.det_x
        y = aman.det_info.wafer.det_y
        pol = aman.det_info.wafer.angle
        if use_matched:
            det_ids = aman[name].matched_det_id
            dm_sort = np.argsort(aman.det_info.det_id)
            mapping = np.argsort(np.argsort(det_ids))
            x = x[dm_sort][mapping]
            y = y[dm_sort][mapping]
            pol = pol[dm_sort][mapping]

        focal_plane = np.column_stack(
            (aman[name].xi, aman[name].eta, aman[name].polang, x, y, pol)
        ).astype(float)
        out_msk = aman[name].outliers
        focal_plane[out_msk, :3] = np.nan

        for di, fp in zip(det_ids, focal_plane):
            try:
                fp_dict[di].append(fp)
            except KeyError:
                fp_dict[di] = [fp]

    if not fp_dict:
        logger.error("No valid observations provided")
        sys.exit()

    # Compute the average focal plane while ignoring outliers
    det_id, focal_plane = _avg_focalplane(fp_dict)
    measured = focal_plane[:3]

    # Get nominal xi, eta, gamma
    nominal = get_nominal(focal_plane, config["coord_transform"])

    # Compute transformation between the two nominal and measured pointing
    if np.isnan(measured[2]).all():
        logger.warning("No polarization data availible, gammas will be nan")
        nominal = nominal[:2]
        measured = measured[:2]
    affine, shift = get_affine(np.vstack(nominal), np.vstack(measured))
    scale, shear, rot = decompose_affine(affine)
    rot = decompose_rotation(rot)

    # Make final outputs and save
    fpout = _mk_fpout(det_id, focal_plane)
    tpout = _mk_tpout(shift, scale, shear, rot)
    write_dataset(fpout, outpath, "focal_plane", overwrite=True)
    write_dataset(tpout, outpath, "pointing_transform", overwrite=True)


if __name__ == "__main__":
    main()
