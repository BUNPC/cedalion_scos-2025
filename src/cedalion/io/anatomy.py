"""Functions for reading and processing anatomical data."""

import os
from typing import Dict, Tuple
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
import xarray as xr

from cedalion.dataclasses import affine_transform_from_numpy

# FIXME
AFFINE_CODES = {
    0: "unknown",  # sform not defined
    1: "scanner",  # RAS+ in scanner coordinates
    2: "aligned",  # RAS+ aligned to some other scan
    3: "talairach",  # RAS+ in Talairach atlas space
    4: "mni",  # RAS+ in MNI atlas space
}


def _get_affine_from_niftii(image: nibabel.nifti1.Nifti1Image):
    """Get affine transformation matrix from NIFTI image.

    Args:
        image (nibabel.nifti1.Nifti1Image): NIFTI image object

    Returns:
        xr.DataArray: Affine transformation matrix
    """
    transform, code = image.get_sform(coded=True)
    if code != 0:
        return affine_transform_from_numpy(
            transform, "ijk", AFFINE_CODES[code], "1", "mm"
        )

    transform, code = image.get_qform(coded=True)
    if code != 0:
        return affine_transform_from_numpy(
            transform, "ijk", AFFINE_CODES[code], "1", "mm"
        )

    transform = image.get_best_affine()

    return affine_transform_from_numpy(transform, "ijk", AFFINE_CODES[0], "1", "mm")


def read_segmentation_masks(
    basedir: str,
    mask_files: Dict[str, str] = {
        "csf": "csf.nii",
        "gm": "gm.nii",
        "scalp": "scalp.nii",
        "skull": "skull.nii",
        "wm": "wm.nii",
    },
) -> Tuple[xr.DataArray, np.ndarray]:
    """Read segmentation masks from NIFTI files.

    Args:
        basedir (str): Directory containing the mask files
        mask_files (Dict[str, str]): Dictionary mapping segmentation types to filenames

    Returns:
        Tuple[xr.DataArray, np.ndarray]:
            - masks (xr.DataArray): Concatenated segmentation masks with a new
              dimension `segmentation_type`.
            - affine (np.ndarray): Affine transformation matrix associated with the
              NIFTI files.
    """
    mask_ids = {seg_type: i + 1 for i, seg_type in enumerate(mask_files.keys())}
    masks = []
    affines = []

    for i, (seg_type, fname) in enumerate(mask_files.items()):
        fpath = os.path.join(basedir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"file '{fpath}' does not exist.")

        f = nibabel.load(fpath)

        volume = f.get_fdata()

        if len(np.unique(volume)) > 2:
            raise ValueError(
                "expected binary mask but found more than two unique "
                f"values in '{fpath}'"
            )

        # mask volume should contain integers stored as floating point numbers.
        # Operations like resampling can introduce small deviations and non-integer
        # mask ids -> round them.
        volume = volume.round(6).astype(np.uint8)

        volume[volume != 0] = mask_ids[seg_type]

        masks.append(
            xr.DataArray(
                volume,
                dims=["i", "j", "k"],
                coords={"segmentation_type": seg_type},
            )
        )

        affine = _get_affine_from_niftii(f)
        affines.append(affine)

    # check that sforms match for all masks
    for i in range(1, len(affines)):
        assert np.all(affines[i] == affines[i])

    masks = xr.concat(masks, dim="segmentation_type")

    # check for voxel that belong to more than one mask # FIXME too strict?
    if (masks > 0).sum("segmentation_type").max() > 1:
        raise ValueError("found voxels with positive entries in more than one mask.")

    affine = affines[0]

    return masks, affine


def cell_coordinates(mask, affine, units="mm"):
    """Get the coordinates of each voxel in the transformed mask.

    Args:
        mask (xr.DataArray): A binary mask of shape (i, j, k).
        affine (np.ndarray): Affine transformation matrix.
        units (str): Units of the output coordinates.

    Returns:
        xr.DataArray: Coordinates of the center of each voxel in the mask.
    """
    # coordinates in voxel space
    i = np.arange(mask.shape[0])
    j = np.arange(mask.shape[1])
    k = np.arange(mask.shape[2])

    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")

    coords = np.stack((ii, jj, kk), -1)  # shape (ni,nj,nk,3)
    transformed = xr.DataArray(
        nibabel.affines.apply_affine(affine, coords),
        dims=["i", "j", "k", "pos"],
        coords={"i": i, "j": j, "k": k},
        attrs={"units": units},
    )

    transformed = transformed.pint.quantify()

    return transformed



def read_parcellations(parcel_file: str | Path) -> pd.DataFrame:
    """Read parcellation labels from a json file.

    Args:
        parcel_file: The parcels file name

    Returns:
        pd.DataFrame: Contains vertices' labels, their appropriate colors
    """
    parcels = pd.read_json(parcel_file)

    parcels = parcels.explode("Vertices")
    parcels["Vertices"] = parcels["Vertices"].astype(int)
    parcels = parcels.sort_values("Vertices")

    parcels["Label"] = parcels["Label"].apply(lambda x: "_".join(x.split(" ")) + "H")

    return parcels
