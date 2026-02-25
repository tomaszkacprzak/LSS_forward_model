from __future__ import annotations
import numpy as np
import glass
from typing import List, Tuple
import pandas as pd
import healpy as hp
import pyccl as ccl
import frogress
import os
from typing import Dict, Mapping, Iterable, Optional, Tuple
import camb
from cosmology import Cosmology
import copy
from .halos import *
import BaryonForge as bfn
import copy



ParamSpec = Iterable[float]  # (low, high, scale)

def add_shells(camb_pars,nside_maps = 1024,missing_shells = None):

    """
    Build CAMB source windows from the given shells, compute their C_ell,
    discretize for HEALPix, and draw ONE correlated realization:
    returns a list of length N (one map per shell).

    Parameters
    ----------
    camb_pars : camb.CAMBparams
        A configured CAMB parameter object (will not be mutated).
    nside_maps : int
        HEALPix Nside for the output maps.
    missing_shells : sequence of (z, W, _)
        Each entry provides the redshift grid and weights for one shell.

    Notes
    -----
    - GLASS expects Cls in *upper-triangle* order:
      (W1xW1, W1xW2, ..., W1xWn, W2xW2, ..., WnxWn).
    - For scalar (spin-0) number-counts fields, keep *all* correlations:
      don't pass `ncorr` to `discretized_cls`/`generate` unless you want to
      intentionally limit correlations to save memory.
    """
    
    lmax = nside_maps*2
    limber=True
    limber_lmin=100
    pars = camb_pars
    
    # set up parameters for angular power spectra
    pars.WantTransfer = True
    pars.WantCls = True
    pars.Want_CMB = True
    #pars.min_l = 1
    pars.set_for_lmax(lmax)
    
    # set up parameters to only compute the intrinsic matter cls
    pars.SourceTerms.limber_windows = limber
    pars.SourceTerms.limber_phi_lmin = limber_lmin
    pars.SourceTerms.counts_density = True
    pars.SourceTerms.counts_redshift = False
    pars.SourceTerms.counts_lensing = False
    pars.SourceTerms.counts_velocity = False
    pars.SourceTerms.counts_radial = False
    pars.SourceTerms.counts_timedelay = False
    pars.SourceTerms.counts_ISW = False
    pars.SourceTerms.counts_potential = False
    pars.SourceTerms.counts_evolve = False
    
    sources = []
    for za, wa, _ in missing_shells:
        s = camb.sources.SplinedSourceWindow(z=za, W=wa)
        sources.append(s)
    pars.SourceWindows = sources
    
    
    n = len(sources)
    cls_ = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)
    
    for i in range(1, n+1):
        if np.any(cls_[f'W{i}xW{i}'] < 0):
            warnings.warn('negative auto-correlation in shell {i}; improve accuracy?')
    
    cc =  [cls_[f'W{i}xW{j}'] for i in range(1, n+1) for j in range(i, 0, -1)]
    
    cls = glass.discretized_cls(cc, nside=nside_maps, lmax=lmax, ncorr=3)
    fields = glass.gaussian_fields(missing_shells)
    
    matter = glass.generate(fields, cls, nside_maps, ncorr=3)
    maps = list(matter)        # length 15, one HEALPix map per shell
    return maps



def load_or_save_updated_params(path_sim, base_params_path, cache_filename, values_to_update=None, overwrite = False):
    """
    Load parameter file if it exists; otherwise create it by updating base values.
    Saves and returns [bpar, sys] where:
      bpar = full dict (base + updates)
      sys  = updated values only
    """
    base = np.load(base_params_path, allow_pickle=True).item()

    if os.path.exists(path_sim+cache_filename):
        if overwrite == False:
            return tuple(np.load(path_sim+cache_filename, allow_pickle=True))

    sys = {}
    if values_to_update:
        for k, v in values_to_update.items():
            sys[k] = float(v)
    bpar = {**base, **sys}
    np.save(path_sim+cache_filename, [bpar, sys])
    return bpar, sys


def draw_params_from_specs(specs):
    """Draw parameter values given specs like {'M_c': (12.5, 15.5, 'lin')}."""

    sysdraw = {}
    for key, (low, high, kind) in specs.items():
        if kind == "log10":
            val = 10 ** np.random.uniform(low, high)
        else:
            val = np.random.uniform(low, high)
        sysdraw[key] = val
    return sysdraw


def baryonify_shell(halos, sims_parameters, counts, bpar, min_z, max_z, nside):
    """
    Apply baryonification to a projected lightcone shell using the Baryonification 2D framework.

    This routine constructs a 2D baryonification displacement model using DMO and DMB profiles,
    interpolates the model across a given redshift shell, and applies it to a halo lightcone
    catalog to compute the baryon-modified density field on a HEALPix shell.

    Parameters
    ----------
    halos : dict
        Dictionary containing halo catalog with keys 'ra', 'dec', 'z', and 'M'.
        Halos should be in physical units (M in Msun/h, z is redshift).

    sims_parameters : dict
        Dictionary of cosmological parameters including:
            - 'Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'w0'

    counts : np.ndarray
        2D HEALPix map of halo counts or projected mass on the shell (same `nside`).

    bpar : dict
        Dictionary of baryonification model parameters. Must include at least:
            - profile parameters for DMO and DMB models
            - 'epsilon_max': maximum smoothing scale in Mpc

    min_z, max_z : float
        Redshift bounds of the shell to which baryonification will be applied.

    nside : int
        HEALPix resolution parameter of the shell.

    Returns
    -------
    density_baryonified : np.ndarray
        HEALPix map of the baryonified overdensity field:
            δ = ρ / ⟨ρ⟩ - 1
    """

    DMO = bfn.Profiles.DarkMatterOnly(**bpar)
    DMB = bfn.Profiles.DarkMatterBaryon(**bpar)
    PIX = bfn.HealPixel(nside)
    DMO = bfn.ConvolvedProfile(DMO, PIX)
    DMB = bfn.ConvolvedProfile(DMB, PIX)

    Displacement = bfn.Profiles.Baryonification2D(
        DMO, DMB, cosmo=cosmo, epsilon_max=bpar['epsilon_max']
    )
    Displacement.setup_interpolator(
        z_min=min_z, z_max=max_z, N_samples_z=2, z_linear_sampling=True,
        R_min=1e-4, R_max=300, N_samples_R=2000, verbose=True
    )

    cdict = {
        'Omega_m': sims_parameters['Omega_m'],
        'sigma8': sims_parameters['sigma_8'],
        'h': sims_parameters['h'],
        'n_s': sims_parameters['n_s'],
        'w0': sims_parameters['w0'],
        'Omega_b': sims_parameters['Omega_b'],
    }

    shell = bfn.utils.LightconeShell(map=counts, cosmo=cdict)

    mask_z = (halos['z'] > min_z) & (halos['z'] < max_z)

    halos_ = bfn.utils.HaloLightConeCatalog(
        halos['ra'][mask_z],
        halos['dec'][mask_z],
        halos['M'][mask_z],
        halos['z'][mask_z],
        cosmo=cdict
    )

    Runners = bfn.Runners.BaryonifyShell(
        halos_, shell, epsilon_max=bpar['epsilon_max'], model=Displacement, verbose=True
    )

    baryonified_shell = Runners.process()
    if np.sum(baryonified_shell)>0:
        density_baryonified = baryonified_shell / np.mean(baryonified_shell) - 1
    else:
        return baryonified_shell
    return density_baryonified


def make_density_maps(shells_info,path_simulation,path_output,nside_maps,shells,camb_pars):
    """
    Generate or load downgraded Healpy maps of the density contrast for each simulation shell.

    This function computes the density contrast δ = (n / <n>) - 1 from particle counts 
    stored in parquet files for each shell defined in `shells_info`. It then downgrades 
    the resolution of each map to `nside_maps` using Healpy's `ud_grade` and saves the result 
    to disk. If the maps already exist, they are loaded instead of recomputed.

    Parameters
    ----------
    shells_info : dict
        Dictionary containing metadata for each shell, including a . Step' key that holds 
        the list of simulation timesteps (should be sortable as integers).
    path_simulation : str
        Path to the directory containing the simulation outputs, where each file is named 
        'particles_<step>_4096.parquet'.
    nside_maps : int
        Desired Nside resolution for the output Healpy maps.

    Returns
    -------
    delta : np.ndarray
   .     Array of downgraded Healpy maps of the density contrast for each shell. Shape is 
        (number_of_shells, 12 * nside_maps**2).
    """

    delta= []
    missing_shells = []
    for iii in frogress.bar(range(len(shells_info['Step']))):
        try:            
            step = shells_info['Step'][::-1][iii]

            try:
                path = path_simulation + '/particles_{0}_4096.parquet'.format(int(step))
                counts = np.array(pd.read_parquet(path)).flatten()
            except:
                path = path_simulation + '/run.{:05d}.lightcone.npy'.format(int(step))
                counts = np.load(path)*1.                
            nside_original = hp.npix2nside(counts.size) 
            
            if np.sum(counts) == 0:
                delta.append(hp.ud_grade(counts*1.0,nside_out=nside_maps))
            else:
                if nside_maps != nside_original:
                    d = counts/np.mean(counts)-1
                    alm = hp.map2alm(d,lmax = nside_maps*2)

                    #deconvolve original window function - 
                    p = hp.sphtfunc.pixwin(nside_original)
                    alm_scaled = hp.almxfl(alm, 1/p[: nside_maps*2])
                    d_filtered = hp.alm2map(alm_scaled,nside= nside_maps,pixwin=True)
                    delta.append(d_filtered)
                else:
                    d = counts/np.mean(counts)-1
                    delta.append(d)
        except:
            print ('missing shell -  ',step)
            missing_shells.append(step)

    # Add missing shells --------------------
    try:
        if len(missing_shells)>0:
            missing_shells = [shells[::-1][np.where(shells_info['Step'] == int(i))[0][0]] for i in missing_shells]
            density_to_be_added = add_shells(camb_pars,nside_maps = nside_maps,missing_shells = missing_shells)
    
            for d in density_to_be_added:
                delta.append(d)  
    except:
        print ('failed adding shells at high redshift ',missing_shells )
    delta = np.array(delta)
    np.save(path_output,delta)
    return delta




    

def convert_to_pix_coord(ra, dec, nside=1024):
    """
    Converts RA,DEC to hpix coordinates
    """
    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    return pix



def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048):
    """
    Convert shear (γ1, γ2) to E/B-mode convergence maps on the sphere (HEALPix).

    All inputs are HEALPix maps at `nside`. Masking is applied before the spin-2 transform.

    Returns
    -------
    E_map, B_map : np.ndarray
        E- and B-mode convergence maps.
    almsE : np.ndarray
        E-mode alm coefficients.
    alms : tuple(np.ndarray)
        (T, E, B) alm triple returned by healpy.map2alm with pol=True.
    """

    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!


    ell, emm = hp.Alm.getlm(lmax=lmax)


    almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
    almsB = alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5

    almsE[ell == 0] = 0.0
    almsB[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    almsB[ell == 1] = 0.0



    almssm = [alms[0], almsE, almsB]


    kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
    E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
    B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)

    return E_map, B_map, almsE, alms


def shift_nz(z, nz, z_rebinned, delta_z=0.0, renorm="source"):
    """
    Evaluate the shifted n(z - delta_z) on z_rebinned.
    
    Parameters
    ----------
    z : array-like
        Original redshift grid (must be monotonic).
    nz : array-like
        n(z) sampled on `z`. Can be normalized or not.
    z_rebinned : array-like
        Target redshift grid where you want the shifted n(z) evaluated.
    delta_z : float, optional
        Shift to apply (positive shifts the distribution to higher observed z).
    renorm : {"none", "source", "target"}, optional
        - "none": no renormalization.
        - "source": scale to preserve the original integral ∫ n(z) dz over the original z grid.
        - "target": scale to unit integral over z_rebinned after shifting.
    
    Returns
    -------
    nz_shifted : np.ndarray
        The shifted n(z) evaluated on z_rebinned.
    """
    z = np.asarray(z)
    nz = np.asarray(nz)
    zr = np.asarray(z_rebinned)

    # Query positions: n(z - delta_z) evaluated at zr  -> sample original at (zr - delta_z)
    z_query = zr - delta_z

    # Linear interpolation with zero outside support
    nz_shifted = np.interp(z_query, z, nz, left=0.0, right=0.0)

    if renorm is not None and renorm != "none":
        if renorm == "source":
            area_src = np.trapz(nz, z)
            area_new = np.trapz(nz_shifted, zr)
            # match the original area (if nz was normalized, this keeps it normalized)
            if area_new > 0:
                nz_shifted *= (area_src / area_new)
        elif renorm == "target":
            area_new = np.trapz(nz_shifted, zr)
            if area_new > 0:
                nz_shifted /= area_new
        else:
            raise ValueError("renorm must be one of {'none','source','target'}")

    return nz_shifted


def apply_nz_shifts_and_build_shells(
    z_rebinned,
    nz_all,
    dz_values,
    shells_info,
    renorm="source",
    samples_per_shell=100,
):
    """
    Apply redshift shifts to pre-rebinned n(z) arrays, normalize them,
    and build lensing shells.

    Parameters
    ----------
    z_rebinned : array
        Redshift grid (already rebinned / defined upstream).
    nz_all : list of arrays
        List of n(z) arrays (one per tomographic bin, including total if desired).
        Must be on the same z_rebinned grid.
    dz_values : list or np.ndarray
        List of delta-z shifts for each tomographic bin.
    shells_info : dict
        Output of recover_shell_info().
    renorm : str, optional
        Passed to shift_nz(); controls normalization mode.
    samples_per_shell : int, optional
        Number of samples per shell when building the windows.

    Returns
    -------
    nz_shifted : np.ndarray
        Shifted and normalized n(z) for each tomographic bin.
    shells, steps, zeff, ngal_glass : tuple
        Outputs from build_shell_windows_and_partitions().
    """
    nz_shifted = []
    for ix, dz in enumerate(dz_values):
        nz_rebinned = nz_all[ix]
        shifted = shift_nz(
            z=z_rebinned,
            nz=nz_rebinned,
            z_rebinned=z_rebinned,
            delta_z=dz,
            renorm=renorm,
        )
        shifted /= np.trapz(shifted, z_rebinned)
        nz_shifted.append(shifted)
    nz_shifted = np.array(nz_shifted)

    shells, steps, zeff, ngal_glass = build_shell_windows_and_partitions(
        shells_info=shells_info,
        redshift=z_rebinned,
        nz=nz_shifted,
        samples_per_shell=samples_per_shell,
    )

    return nz_shifted, shells, steps, zeff, ngal_glass
    

def F_nla(z, om0, A_ia, rho_c1, eta=0.0, z0=0.0, cosmo=None):
    """
    Nonlinear linear-alignment (NLA) amplitude F(z).

    Parameters
    ----------
    z : array-like
        Redshift(s).
    om0 : float
        Present-day matter density parameter Ω_m.
    A_ia : float
        Intrinsic-alignment amplitude.
    rho_c1 : float
        Normalization constant (often 0.0134 for IA in some conventions).
    eta : float, optional
        Redshift evolution exponent (default 0.0).
    z0 : float, optional
        Pivot redshift for evolution (default 0.0).
    cosmo : pyccl.Cosmology
        CCL cosmology; used to compute the growth factor D(a).

    Returns
    -------
    ndarray
        F(z) = -A_ia * rho_c1 * Ω_m * [(1+z)/(1+z0)]^eta / D(z).
    """
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    if cosmo is None:
        raise ValueError("Pass a pyccl.Cosmology as `cosmo` to use CCL growth.")
    D = ccl.growth_factor(cosmo, a)  # normalized to 1 at a=1
    return -A_ia * rho_c1 * om0 * ((1 + z) / (1 + z0))**eta / D


def IndexToDeclRa(index, nside, nest=False):
    """
    Convert HEALPix pixel index to (Dec, RA) in degrees.

    Parameters
    ----------
    index : array-like or int
        HEALPix pixel index/indices.
    nside : int
        HEALPix NSIDE.
    nest : bool, optional
        If True, assume NESTED ordering; else RING.

    Returns
    -------
    dec, ra : ndarray
        Declination and Right Ascension in degrees.
    """
    theta, phi = hp.pixelfunc.pix2ang(nside, index, nest=nest)
    return -np.degrees(theta - np.pi/2.0), np.degrees(phi)


def rotate_and_rebin(pix_, nside_maps, rot, delta_=0.0):
    """
    Apply a deterministic rotation/mirror to pixel centers and rebin to `nside_maps`.

    This mimics your map-rotation scheme:
      - rot=0: 0°
      - rot=1: 180°
      - rot=2: 90° + mirror
      - rot=3: 270° + mirror
    You can add a small extra angle `delta_` (degrees).

    Parameters
    ----------
    pix_ : array-like
        HEALPix pixel indices at NSIDE = 2 * nside_maps (RING).
    nside_maps : int
        Target NSIDE for the rebinned output.
    rot : int
        Rotation code in {0, 1, 2, 3}.
    delta_ : float, optional
        Additional rotation angle in degrees.

    Returns
    -------
    pix_rebinned : ndarray
        HEALPix pixel indices at NSIDE = nside_maps (RING) after the transform.
    """
    if rot not in (0, 1, 2, 3):
        raise ValueError("rot must be one of {0, 1, 2, 3}.")

    # per-rot settings
    angle_by_rot = [0, 180, 90, 270]   # degrees
    flip_by_rot  = [False, False, True, True]

    ang = angle_by_rot[rot] + delta_
    flip = flip_by_rot[rot]

    rotu = hp.rotator.Rotator(rot=[ang, 0, 0], deg=True)

    # original directions at NSIDE = 2*nside_maps
    alpha, delta = hp.pix2ang(nside_maps * 2, pix_)
    rot_alpha, rot_delta = rotu(alpha, delta)

    if flip:
        rot_alpha = np.pi - rot_alpha  # mirror in alpha

    # back to pixels (still at 2*nside_maps)
    pix_hi = hp.ang2pix(nside_maps * 2, rot_alpha, rot_delta)

    # convert to (Dec, RA) and then to target NSIDE pixels
    dec__, ra__ = IndexToDeclRa(pix_hi, nside_maps * 2)
    return convert_to_pix_coord(ra__, dec__, nside=nside_maps)

def unrotate_map(rotated_map, nside_maps, rot, delta_=0.0):
    """
    Invert the rotate_and_rebin operation at map level.

    Parameters
    ----------
    rotated_map : array-like, shape (hp.nside2npix(nside_maps),)
        The map AFTER your rotate_and_rebin-style transformation (defined on nside_maps).
    nside_maps : int
        Target NSIDE of the (unrotated) map you want back.
    rot : int
        Same 'rot' used in rotate_and_rebin (0..3).
    delta_ : float, optional
        Same delta_ used in rotate_and_rebin (degrees added to angle_by_rot[rot]).

    Returns
    -------
    unrotated_map : np.ndarray
        Map sampled to undo the rotation/mirroring.
    """
    # Forward settings (same as rotate_and_rebin)
    angle_by_rot = [0, 180, 90, 270]   # degrees
    flip_by_rot  = [False, False, True, True]

    ang  = angle_by_rot[rot] + delta_
    flip = flip_by_rot[rot]

    rotu = hp.rotator.Rotator(rot=[ang, 0, 0], deg=True)

    npix = hp.nside2npix(nside_maps)
    # Pixel centers of the *unrotated* target grid
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside_maps, pix)          # (theta, phi)

    # Apply the *forward* transform to find where to sample from in the rotated map
    rot_theta, rot_phi = rotu(theta, phi)
    if flip:
        rot_theta = np.pi - rot_theta                 # same flip as forward op (its own inverse)

    # Pixels in the rotated map corresponding to those directions
    src_pix = hp.ang2pix(nside_maps, rot_theta, rot_phi)

    # Pull sampling (no holes); if you need smoothing, consider bilinear/neighbor averaging
    unrotated_map = np.asarray(rotated_map)[src_pix]
    return unrotated_map



def build_shells(shells_info: dict,samples_per_shell: int = 100):
    steps_rev = shells_info["Step"][::-1]
    z_near_rev = shells_info["z_near"][::-1]
    z_far_rev = shells_info["z_far"][::-1]

    shells: List[glass.shells.RadialWindow] = []
    zeff_list = []
    steps_list = []

    for step, zmin, zmax in zip(steps_rev, z_near_rev, z_far_rev):
        za = np.linspace(float(zmin), float(zmax), samples_per_shell)
        wa = np.ones_like(za)
        zeff = 0.5 * (float(zmin) + float(zmax))
        shells.append(glass.shells.RadialWindow(za, wa, zeff))
        steps_list.append(int(step))
        zeff_list.append(zeff)

    steps = np.asarray(steps_list, dtype=int)
    zeff_array = np.asarray(zeff_list, dtype=float)
    return shells, steps, zeff_array

     

def build_shell_windows_and_partitions(
    shells_info: dict,
    redshift: np.ndarray,
    nz: np.ndarray,
    samples_per_shell: int = 100,
) -> Tuple[List[glass.shells.RadialWindow], np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct GLASS radial windows for each simulation shell and compute
    tomographic partitions (per shell) given n(z) for multiple bins.

    Parameters
    ----------
    shells_info : dict
        Output of `recover_shell_info`. Must contain 'Step', 'z_near', 'z_far'.
    redshift : (Nz,) ndarray
        Redshift grid matching the second axis of `nz`.
    nz : (Ntomo, Nz) ndarray
        n(z) per tomographic bin sampled on `redshift`.
        It need not be normalized; `glass.shells.partition` only uses relative weights.
    samples_per_shell : int, optional
        Number of linearly spaced z-samples to use inside each [zmin, zmax] window.

    Returns
    -------
    shells : list[glass.shells.RadialWindow]
        One RadialWindow per shell (ordered low→high z to match GLASS expectations).
    steps : (Nshell,) ndarray of int
        Integer step index per shell (aligned with `shells`).
    zeff_array : (Nshell,) ndarray of float
        Effective redshift per shell (midpoint).
    ngal_glass : (Ntomo, Nshell) ndarray
        Fractional counts per tomographic bin per shell from `glass.shells.partition`.
        Each row integrates (approximately) to 1 over shells if `nz` is normalized.
    """
    # Reverse once (your inputs were high→low; GLASS prefers low→high)
    steps_rev = shells_info["Step"][::-1]
    z_near_rev = shells_info["z_near"][::-1]
    z_far_rev = shells_info["z_far"][::-1]

    shells: List[glass.shells.RadialWindow] = []
    zeff_list = []
    steps_list = []

    for step, zmin, zmax in zip(steps_rev, z_near_rev, z_far_rev):
        za = np.linspace(float(zmin), float(zmax), samples_per_shell)
        wa = np.ones_like(za)
        zeff = 0.5 * (float(zmin) + float(zmax))
        shells.append(glass.shells.RadialWindow(za, wa, zeff))
        steps_list.append(int(step))
        zeff_list.append(zeff)

    steps = np.asarray(steps_list, dtype=int)
    zeff_array = np.asarray(zeff_list, dtype=float)

    # Partition n(z) into the shell windows: shape (Ntomo, Nshell)
    ngal_glass = np.array([glass.shells.partition(redshift, nz_i, shells)
                           for nz_i in nz], dtype=float)

    return shells, steps, zeff_array, ngal_glass


# def compute_lensing_fields(density, shells, camb_pars, nside_maps, *,
#                            do_kappa=True, do_shear=True, do_IA=False):
#     """
#     Compute kappa, shear, and/or intrinsic-alignment shear fields
#     from a set of density shells using glass.lensing.

#     Parameters
#     ----------
#     density : list of arrays
#         Density maps per shell.
#     shells : list
#         Corresponding shell window definitions.
#     camb_pars : dict
#         CAMB parameters (for Cosmology.from_camb).
#     nside_maps : int
#         Healpix nside for the maps.
#     do_kappa, do_shear, do_IA : bool
#         Control which fields are returned.

#     Returns
#     -------
#     dict
#         Dictionary containing requested fields among
#         {"kappa", "gamma", "IA_shear"}.
#     """
#     cosmo = Cosmology.from_camb(camb_pars)
#     results = {}
    
#     if do_kappa or do_shear:
#         conv = glass.lensing.MultiPlaneConvergence(cosmo)
#         kappa_list, gamma_list = [], []
#         for ss in frogress.bar(range(len(density))):
#             conv.add_window(density[ss], shells[ss])
#             kappa = copy.deepcopy(conv.kappa)
#             if do_kappa:
#                 kappa_list.append(kappa)
#             if do_shear:
#                 gamma_list.append(glass.lensing.from_convergence(kappa, lmax=nside_maps*3-1, shear=True))
#         if do_kappa:
#             results["kappa"] = np.array(kappa_list)
#         if do_shear:
#             results["gamma"] = np.array(gamma_list)

#     if do_IA:
#         IA_list = []
#         for ss in frogress.bar(range(len(density))):
#             IA_list.append(
#                 glass.lensing.from_convergence(
#                     density[ss] - np.mean(density[ss]),
#                     lmax=nside_maps*3-1,
#                     shear=True
#                 )
#             )
#         results["IA_shear"] = np.array(IA_list)

#     return results

def compute_lensing_fields(density, shells, camb_pars, nside_maps, *,
                           do_kappa=True, do_shear=True, do_IA=False):
    """
    Compute kappa, shear, and/or intrinsic-alignment shear fields
    from a set of density shells using glass.lensing.

    Parameters
    ----------
    density : list of arrays
        Density maps per shell.
    shells : list
        Corresponding shell window definitions.
    camb_pars : dict
        CAMB parameters (for Cosmology.from_camb).
    nside_maps : int
        Healpix nside for the maps.
    do_kappa, do_shear, do_IA : bool
        Control which fields are returned.

    Returns
    -------
    dict
        Dictionary containing requested fields among
        {"kappa", "gamma", "IA_shear"}.
    """
    cosmo = Cosmology.from_camb(camb_pars)
    results = {}
    
    if do_kappa or do_shear:
        conv = glass.lensing.MultiPlaneConvergence(cosmo)

        for ss in frogress.bar(range(len(density))):
            conv.add_window(density[ss], shells[ss])
            kappa = copy.deepcopy(conv.kappa)

            if do_kappa:
                if ss == 0:
                    results["kappa"] = np.zeros((len(density),) + kappa.shape, dtype=kappa.dtype)
                results["kappa"][ss] = kappa
            
            if do_shear:
                gamma = glass.lensing.from_convergence(kappa, lmax=nside_maps*3-1, shear=True)
                if ss == 0:
                    results["gamma"] = np.zeros((len(density),len(gamma),) + gamma[0].shape, dtype=gamma[0].dtype)
                results["gamma"][ss] = gamma

    if do_IA:
        
        for ss in frogress.bar(range(len(density))):
            
            IA = glass.lensing.from_convergence(
                    density[ss] - np.mean(density[ss]),
                    lmax=nside_maps*3-1,
                    shear=True
                )

            if ss == 0:
                results["IA_shear"] = np.zeros((len(density),len(IA),) + IA[0].shape, dtype=IA[0].dtype)
            results["IA_shear"][ss] = IA
        
    return results


def integrate_field(ngal_glass,field):
    nside_maps = hp.npix2nside(len(field[0]))
    field_tomo = np.zeros((len(ngal_glass),nside_maps**2*12))
    for tomo in range(len(ngal_glass)):
        for i in range(len(field)):
             field_tomo[tomo,:] += ngal_glass[tomo,i] * field[i]  
    return field_tomo


def load_and_baryonify_gower_st_shells(
    path_simulation,
    sims_parameters,
    cosmo_bundle,
    baryons,
    nside_maps,
    shells_info,
    shells,
    overwrite_baryonified_shells = False,
    dens_path = None,
    tsz_path = None
):
    """
    Load or create baryonified (or normal) GowerSt2 density shells.

    Returns
    -------
    density : np.ndarray
        Array of density shells.
    label_baryonification : str
        'baryonified' or 'normal'
    """
    
    if baryons["enabled"]:

        baryons.setdefault('no_calib', True)
        
        bpar, sys = load_or_save_updated_params(path_simulation,baryons['base_params_path'],baryons['filename_new_params'],baryons['values_to_update'], overwrite = False)
        label_baryonification = "baryonified"

        halo_catalog_path = os.path.join(path_simulation, "halo_catalog.parquet")
        if tsz_path is None:
            tsz_path = os.path.join(path_simulation, f"tsz_{nside_maps}.npy")
        if dens_path is None:
            dens_path = os.path.join(path_simulation, f"delta_b_{nside_maps}.npy")


        # --- Create halo catalog if missing
        if not os.path.exists(halo_catalog_path):
            print("Creating halo light cone...")
            save_halocatalog(
                shells_info,
                sims_parameters,
                max_redshift=baryons['max_z_halo_catalog'],
                halo_snapshots_path=path_simulation,
                catalog_path=halo_catalog_path,
            )

        # --- Create baryonified density (and tSZ if needed)
        if overwrite_baryonified_shells and os.path.exists(dens_path):
            try:
                os.remove(dens_path)
            except:
                pass
        if not os.path.exists(dens_path) or (baryons['do_tSZ'] and not os.path.exists(tsz_path)):
            
            halos = load_halo_catalog(
                halo_catalog_path,
                cosmo_bundle['colossus_params'],
                sims_parameters,
                baryons['mass_cut'],
                no_calib = baryons['no_calib'],
            )
            
            make_tsz_and_baryonified_density(
                path_simulation,
                sims_parameters,
                cosmo_bundle['cosmo_pyccl'],
                halos,
                bpar,
                nside_maps,
                shells_info,
                dens_path,
                tsz_path,
                baryons['do_tSZ'],
                shells,
                cosmo_bundle['pars_camb'],
                baryons['mass_cut'],
            )

        density = np.load(dens_path, allow_pickle=True)

    else:
        label_baryonification = "normal"
        if dens_path is None:
            dens_path = os.path.join(path_simulation, f"delta_{nside_maps}.npy")

        if not os.path.exists(dens_path):
            density = make_density_maps(
                shells_info,
                path_simulation,
                dens_path,
                nside_maps,
                shells,
                cosmo_bundle['pars_camb'],
            )
        else:
            density = np.load(dens_path, allow_pickle=True)

    return density, label_baryonification


def make_tsz_and_baryonified_density(
    path_simulation: str,
    sims_parameters: dict,
    cosmo_pyccl,
    halos: dict,
    bpar: dict,
    nside_maps: int,
    shells_info: dict,
    dens_path: str,
    tsz_path: str,
    do_tSZ: bool,
    shells,
    camb_pars,
    min_mass: float = 13,                      # Msun/h threshold after FoF->SO
    njobs: int = 16,
    particles = None,
):

    """
    Build (if missing) tSZ map and baryonified density shells, saving them to disk.
    Returns the density array for downstream lensing.

    Outputs
    -------
    tsz file:  path_simulation + f"/tsz_{nside_maps}.npy"
    dens file: path_simulation + f"/density_b_{nside_maps}_{noise_rel}.npy"
    """
  
    # ---------- tSZ ----------
    if do_tSZ:
        if not os.path.exists(tsz_path):
            print ('creating a tSZ map --')
            mask = (halos['M'] > 10**min_mass)
            cdict = {
                "Omega_m": sims_parameters["Omega_m"],
                "sigma8": sims_parameters["sigma_8"],
                "h": sims_parameters["h"],
                "n_s": sims_parameters["n_s"],
                "w0": sims_parameters["w0"],
                "Omega_b": sims_parameters["Omega_b"],
            }
    
            halos_ = bfn.utils.HaloLightConeCatalog(
                halos["ra"][mask], halos["dec"][mask], halos['M'][mask], halos["z"][mask], cosmo=cdict
            )
    
            Gas = bfn.Profiles.Gas(**bpar)
            DMB = bfn.Profiles.DarkMatterBaryon(**bpar, twohalo=0 * bfn.Profiles.TwoHalo(**bpar))
            PRS = bfn.Profiles.Pressure(gas=Gas, darkmatterbaryon=DMB)
            PRS = PRS * (1 - bfn.Profiles.Thermodynamic.NonThermalFrac(**bpar))
            PRS = bfn.Profiles.ThermalSZ(PRS)
            PRS = bfn.Profiles.misc.ComovingToPhysical(PRS, factor=-3)
            Pix = bfn.utils.HealPixel(NSIDE=nside_maps)
            PRS = bfn.utils.ConvolvedProfile(PRS, Pix)
            PRS = bfn.utils.TabulatedProfile(PRS, cosmo_pyccl)
    
            zmin, zmax = float(halos["z"].min()), float(halos["z"].max())
            PRS.setup_interpolator(
                z_min=zmin, z_max=zmax, N_samples_z=10, z_linear_sampling=True,
                R_min=1e-4, R_max=300, N_samples_R=2000, verbose=True
            )
    
            shell = bfn.utils.LightconeShell(np.zeros(hp.nside2npix(nside_maps)), cosmo=cdict)
            runner = bfn.Runners.PaintProfilesShell(halos_, shell, epsilon_max=bpar["epsilon_max"], model=PRS, verbose=True)
            painted_shell = bfn.utils.SplitJoinParallel(runner, njobs=njobs).process()
            np.save(tsz_path, painted_shell)

            
            
    # ---------- Baryonified density shells ----------
    if not os.path.exists(dens_path):
        print ('baryonifying shells --')
        density = []
        steps = shells_info["Step"][::-1]
        z_near = shells_info["z_near"][::-1]
        z_far = shells_info["z_far"][::-1]

        missing_shells = []
        for i in frogress.bar(range(len(steps))):
            try:
                step = steps[i]
                zmin = float(z_near[i]) + (1e-6 if i == 0 else 0.0)
                zmax = float(z_far[i])
    
                # shell thickness for projection cutoff
                chi = ccl.comoving_radial_distance
                shell_thickness = chi(cosmo_pyccl, 1.0 / (1.0 + zmax)) - chi(cosmo_pyccl, 1.0 / (1.0 + zmin))
                bpar["proj_cutoff"] = float(shell_thickness / 2.0)
    
                DMO = bfn.Profiles.DarkMatterOnly(**bpar)
                DMB = bfn.Profiles.DarkMatterBaryon(**bpar)


                Pix = bfn.utils.HealPixel(NSIDE=nside_maps)
                DMO = bfn.ConvolvedProfile(DMO, Pix)
                DMB = bfn.ConvolvedProfile(DMB, Pix)

            
                Displacement = bfn.Profiles.Baryonification2D(DMO, DMB, cosmo=cosmo_pyccl, epsilon_max=bpar["epsilon_max"])
    
                try:
                    Displacement.setup_interpolator(
                        z_min=zmin, z_max=zmax, N_samples_z=2, z_linear_sampling=True,
                        R_min=1e-4, R_max=300, N_samples_R=2000, verbose=True
                    )
                except Exception:
                    Displacement.setup_interpolator(
                        z_min=zmin, z_max=zmax, N_samples_z=2, z_linear_sampling=True,
                        R_min=1e-9, R_max=2000, N_samples_R=4000, verbose=True
                    )

                if particles is None:
                    # read it from Gower St format ---------
                    try:
                        path = path_simulation + '/particles_{0}_4096.parquet'.format(int(step))
                        counts = np.array(pd.read_parquet(path)).flatten()
                    except:
                        path = path_simulation + '/run.{:05d}.lightcone.npy'.format(int(step))
                        counts = np.load(path)*1.                
                    
                    nside_original = hp.npix2nside(counts.size) 
                    p = hp.sphtfunc.pixwin(nside_original)
                    alm = hp.map2alm(counts,lmax = nside_maps*2)
                    alm_scaled = hp.almxfl(alm, 1/p[: nside_maps*2])
                    counts = hp.alm2map(alm_scaled,nside= nside_maps,pixwin=True)*(nside_original/nside_maps)**2

                   
                    
                else:
                    # use provided particl counts
                    counts = copy.deepcopy(particles[i])
                    

                mask_z = (halos["z"] > zmin) & (halos["z"] < zmax)


                if len(halos["z"][mask_z])>1:
                    cdict = {
                        "Omega_m": sims_parameters["Omega_m"],
                        "sigma8": sims_parameters["sigma_8"],
                        "h": sims_parameters["h"],
                        "n_s": sims_parameters["n_s"],
                        "w0": sims_parameters["w0"],
                        "Omega_b": sims_parameters["Omega_b"],
                    }
                    halos_ = bfn.utils.HaloLightConeCatalog(
                        halos["ra"][mask_z], halos["dec"][mask_z], halos["M"][mask_z], halos["z"][mask_z], cosmo=cdict
                    )
        
                    shell = bfn.utils.LightconeShell(map=counts, cosmo=cdict)
                    runner = bfn.Runners.BaryonifyShell(halos_, shell, epsilon_max=bpar["epsilon_max"], model=Displacement, verbose=True)
                    baryonified_shell = runner.process()
        
                    if np.mean(baryonified_shell) != 0:
                        density_b = (baryonified_shell / np.mean(baryonified_shell)) - 1.0
                    else:
                        density_b = 0.0 * baryonified_shell
                    density.append(density_b)

                else:
                    if np.mean(counts) != 0:
                        density_b = (counts / np.mean(counts)) - 1.0
                        density.append(density_b)
                    else:
                        density_b = 0.0 * baryonified_shell
                        density.append(density_b)
                        
                    
            except:
                print ('generate step ',step)
                missing_shells.append(step)
                
        if len(missing_shells)>0:
 
            missing_shells = [shells[::-1][np.where(shells_info['Step'] == int(i))[0][0]] for i in missing_shells]       
            density_to_be_added = add_shells(camb_pars,nside_maps = nside_maps,missing_shells = missing_shells)
            for d in density_to_be_added:
                density.append(d)   
        density = np.asarray(density, dtype=np.float32)
        np.save(dens_path, density)






def cmb_lensing_from_glass_plus_gaussian(
    zeff_glass,
    density,
    shells,
    theory,
    cosmo_bundle,
    nside_maps,
    z_max_born=2.5,
    diagnostics_lmax=2000,
    nonlinear=True,
    seed=None,
):
    """
    Build a CMB lensing convergence map by:
      (1) multi-plane convergence from simulated shells up to z_max_born using GLASS
      (2) adding the missing high-z contribution as a Gaussian realization drawn from theory

    Parameters
    ----------
    zeff_glass : array-like
        Effective redshift per shell/window (same ordering as `density` and `shells`).
    density : sequence of array-like
        Overdensity maps per shell (HEALPix, length = 12*nside_maps^2).
    shells : sequence of glass.shells.RadialWindow (or compatible)
        Windows corresponding to each shell.
    theory : object
        Your theory helper with:
          - theory.results.conformal_time(0), theory.results.tau_maxvis,
          - theory.results.redshift_at_comoving_radial_distance(chi),
          - theory.set_Wcmb(),
          - theory.cl_kk_log(nonlinear=..., zmax=...).
    cosmo_bundle : dict-like
        Must contain 'pars_camb' (CAMBparams) used to build glass.Cosmology.
    nside_maps : int
        HEALPix NSIDE of the maps.
    z_max_born : float
        Max redshift up to which simulated shells are included as “low-z” part.
    diagnostics_lmax : int
        Multipole up to which diagnostics ratio is returned (uses hp.anafast output).
    nonlinear : bool
        Passed to theory.cl_kk_log.
    seed : int or None
        Random seed for synfast (high-z Gaussian completion).

    Returns
    -------
    cmb_lensing_map : np.ndarray
        Final kappa map = kappa_lowz + Gaussian completion, shape (12*nside_maps^2,).
    diagnostics : np.ndarray
        Ratio (Cl_measured / (Cl_theory_lowz * pixwin^2)) for ell < diagnostics_lmax.
    meta : dict
        Useful bookkeeping: imax, z_cmb, z_cut_used, cl_z_max, cl_full, cl_missing.
    """
    zeff_glass = np.asarray(zeff_glass, dtype=float)
    npix = 12 * nside_maps**2

    if len(density) != len(shells) or len(density) != len(zeff_glass):
        raise ValueError("density, shells, and zeff_glass must have the same length.")

    # --- CMB distance/redshift from CAMB results
    chi_cmb = theory.results.conformal_time(0) - theory.results.tau_maxvis
    z_cmb = float(theory.results.redshift_at_comoving_radial_distance(chi_cmb))

    # --- choose last shell index included in the low-z part
    idx = np.where(zeff_glass > z_max_born)[0]
    if len(idx) == 0:
        raise ValueError(f"No shell has zeff > z_max_born={z_max_born}. Increase z_max_born?")
    imax = int(idx[0])

    # --- GLASS multipane convergence up to z_max_born, then propagate to source at z_cmb
    cosmo = Cosmology.from_camb(cosmo_bundle["pars_camb"])
    conv = glass.lensing.MultiPlaneConvergence(cosmo)

    for ss in range(imax + 1):
        m = np.asarray(density[ss], dtype=np.float64)
        if m.size != npix:
            raise ValueError(
                f"density[{ss}] has size {m.size}, expected {npix} for nside={nside_maps}."
            )
        conv.add_window(m, shells[ss])

    conv.add_plane(np.zeros(npix, dtype=np.float64), z_cmb)
    kappa_lowz = np.asarray(conv.kappa, dtype=np.float64)

    # --- theory spectra: low-z truncated and full to CMB
    theory.set_Wcmb()
    z_cut_used = float(zeff_glass[imax])  # keep consistent with your current convention

    cl_z_max = np.asarray(theory.cl_kk_log(nonlinear=nonlinear, zmax=z_cut_used), dtype=float)
    cl_full  = np.asarray(theory.cl_kk_log(nonlinear=nonlinear, zmax=z_cmb), dtype=float)

    # --- diagnostics: compare measured low-z Cl to theory low-z Cl (incl pixel window)
    cl_ref = hp.anafast(kappa_lowz)
    ell_max_diag = min(diagnostics_lmax, len(cl_ref), len(cl_z_max), len(hp.pixwin(nside_maps)) )

    cl_pix = hp.pixwin(nside_maps)
    diagnostics = cl_ref[:ell_max_diag] / (cl_z_max[:ell_max_diag] * cl_pix[:ell_max_diag] ** 2)

    # --- missing high-z contribution (clip negatives to avoid synfast issues)
    cl_missing = cl_full - cl_z_max
    cl_missing = np.clip(cl_missing, 0.0, None)

    # healpy synfast expects C_ell starting at ell=0
    if cl_missing[0] != 0.0:
        # Your spectra may start at ell=1; force ell=0 to zero safely
        cl_missing = np.asarray(cl_missing)
        if cl_missing.shape[0] > 0:
            cl_missing[0] = 0.0

    if seed is not None:
        np.random.seed(seed)

    kappa_hi = hp.synfast(cl_missing, nside_maps, new=True)
    cmb_lensing_map = kappa_lowz + kappa_hi

    meta = dict(
        imax=imax,
        z_cmb=z_cmb,
        chi_cmb=float(chi_cmb),
        z_cut_used=z_cut_used,
        cl_z_max=cl_z_max,
        cl_full=cl_full,
        cl_missing=cl_missing,
    )

    return cmb_lensing_map, diagnostics, meta
