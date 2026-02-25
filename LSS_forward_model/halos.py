import os
import gc
import numpy as np
import pandas as pd
import healpy as hp
import pyccl as ccl
import copy
from scipy.interpolate import CubicSpline
from typing import Optional, Dict
import frogress
from colossus.cosmology import cosmology as colossus_cosmology
from scipy.optimize import root_scalar
from scipy.interpolate import RegularGridInterpolator
import BaryonForge as bfn
import astropy.io.fits as fits
from astropy.cosmology import wCDM, Flatw0waCDM
import astropy.units as u

def save_halocatalog(shells_info, sims_parameters, max_redshift = 1.5, halo_snapshots_path = '' , catalog_path = 'halo_catalog.parquet',log10_mass_limit = None):

    """
    Generate a lightcone halo catalog by assembling halo snapshots across redshift shells, 
    replicating boxes to fill the shell volume, and saving the final catalog to a FITS file.

    Parameters
    ----------
    shells_info : dict
        Dictionary containing shell definitions with keys like 'cmd_near', 'cmd_far', 'Step', etc.
        Typically output from `recover_shell_info_`.

    sims_parameters : dict
        Dictionary of cosmological parameters and simulation metadata. Must contain:
            - 'Omega_cdm', 'Omega_b', 'h', 'sigma_8', 'n_s', 'm_nu' (in eV), etc.
            - 'dBoxSize Mpc/h' : Box size in comoving Mpc/h units.

    max_redshift : float, optional
        Upper redshift limit for including halos in the lightcone (default is 1.5).

    catalog_path : str, optional
        Path to save the resulting halo catalog in FITS format.

    Notes
    -----
    - Halos are replicated in a cubic grid until they cover the spherical shell between cmd_near and cmd_far.
    - Box replication ensures volume completeness for the shell.
    - Uses `pyccl` to compute comoving distances for redshift-distance interpolation.
    - Outputs a FITS file containing positions, redshifts, masses, and optional angular coordinates.

    Output
    ------
    A binary FITS table saved at `catalog_path`, containing:
        - pix_16384_ring: HEALPix pixel index
        - log_M: Halo mass (in Msun/h, scaled by 1000)
        - R: Halo half-mass radius (in kpc/h, scaled by 1000)
        - redshift: Scaled by 10000
    """  


    def load_snapshot(path_base, c_, Lbox_Mpc, sims_parameters,log10_mass_limit = None):
        """
        Loads halo data from a specified path based on the mode.
    
        :param path_base: Base path to the data
        :param c_: Configuration index
        :param mode: Mode of the data ('rockstar' or other)
        :param f_mass: Mass factor
        Lbox_Mpc: Lbox in Mpc
        :return: Dictionary containing halo data
        """

        Om0 = float(sims_parameters["Omega_m"])
        w0  = float(sims_parameters["w0"])
        try:
            wa  = float(sims_parameters["wa"])
        except:
            wa = 0.
        H0  = float(sims_parameters["h"]) * 100.0
        Lbox_Mpc = float(sims_parameters["dBoxSize Mpc/h"]) / float(sims_parameters["h"])
        cosmo = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa) if abs(wa) > 0 else wCDM(H0=H0, Om0=Om0, w0=w0, Ode0=1-Om0)
        Lbox = Lbox_Mpc * u.Mpc
        f_mass = (Lbox**3 * cosmo.critical_density(0).to(u.Msun/u.Mpc**3)).value  # Msun
        


        c__ = f'{int(c_):03}'
    
        p = f'{path_base}run.00{c__}.fofstats.parquet'
        p0 = f'{path_base}/fof/run.{int(c_):05d}.fofstats.0'
        pkd_halo_dtype = np.dtype([("rPot", ("f4", 3)), ("minPot", "f4"), ("rcen", ("f4", 3)),
                                   ("rcom", ("f4", 3)), ("cvom", ("f4", 3)), ("angular", ("f4", 3)),
                                   ("inertia", ("f4", 6)), ("sigma", "f4"), ("rMax", "f4"),
                                   ("fMAss", "f4"), ("fEnvironDensity0", "f4"),
                                   ("fEnvironDensity1", "f4"), ("rHalf", "f4")])
    
        parquet_ = True
        columns_to_read = ['halo_center', 'rmax', 'log10M']
        try:
            halos = pd.read_parquet(p, columns=columns_to_read)
        except:
            try:
    
                halos_ = np.fromfile(p0, count=-1, dtype=pkd_halo_dtype)
                halo_center = Lbox_Mpc * (halos_["rPot"]  + halos_["rcen"] + 0.5)
                rmax = Lbox_Mpc * halos_["rMax"]
                log10M = np.log10((halos_['fMAss'] * f_mass))
    
                halos = dict()
                halos["halo_center"] = halo_center.tolist()
                halos["rmax"] =  (rmax* 1000).astype('uint16')
                halos["log10M"] = (log10M* 1000).astype('uint16')
    
                del halos_
                
            except:
                print ('failed ',p)
                sd_
        if log10_mass_limit is not None:
            if min(halos['log10M'])/1000<10:    
                # fiducial covariance, likely off by a factor 1000
                mask = (halos['log10M']/1000 + 4) > log10_mass_limit
                halos = halos[mask]
            else:
                mask = (halos['log10M']/1000)  > log10_mass_limit
                halos = halos[mask]
                        
        centers = np.array([x for x in np.array(halos['halo_center'])])
        M = np.array([x for x in np.array(halos['log10M'])])
        rmax = np.array([x for x in np.array(halos['rmax'])])
    
    
        output = {
            'x': centers[:, 0],
            'y': centers[:, 1],
            'z': centers[:, 2],
            'rhalf': rmax,
            'M': M 
        }
    
                
        return output    

    def may_intersect_sphere(x_i, y_i, z_i, Lbox_Mpc, d_min, d_max):
        """
        Determine whether a replicated simulation box intersects with a spherical shell.
    
        Parameters
        ----------
        x_i, y_i, z_i : int
            Replication indices along each axis. The box is translated by (x_i, y_i, z_i) * Lbox_Mpc.
    
        Lbox_Mpc : float
            Side length of the simulation box in comoving Mpc.
    
        d_min : float
            Inner radius of the spherical shell (in Mpc).
    
        d_max : float
            Outer radius of the spherical shell (in Mpc).
    
        Returns
        -------
        intersects : bool
            True if the box may intersect the shell between d_min and d_max.
    
        min_distance : float
            Minimum possible distance from the origin to any point in the box.
    
        max_distance : float
            Maximum possible distance from the origin to any point in the box.
    
        Notes
        -----
        This function assumes the box is axis-aligned and cubic. It uses the box's 
        center and diagonal to conservatively estimate the range of distances covered 
        by the box and checks for overlap with the target spherical shell.
        """
    # Center of the box after translation
        center_x = x_i * Lbox_Mpc + Lbox_Mpc / 2
        center_y = y_i * Lbox_Mpc + Lbox_Mpc / 2
        center_z = z_i * Lbox_Mpc + Lbox_Mpc / 2
        center = np.array([center_x, center_y, center_z])
    
        # Distance from the origin to the center of the box
        center_distance = np.linalg.norm(center)
        
        # Radius of the sphere that contains the entire box (half-diagonal of the box)
        half_diagonal = np.sqrt(3) * (Lbox_Mpc / 2)
    
        # Calculate the minimum and maximum distances any point in the box could be from the origin
        min_distance = max(0, center_distance - half_diagonal)
        max_distance = center_distance + half_diagonal
    
        # Check if there's any overlap between the box and the sphere range
        return (min_distance <= d_max and max_distance >= d_min),min_distance,max_distance


    

    cosmo = ccl.Cosmology(Omega_c = sims_parameters['Omega_cdm'], Omega_b = sims_parameters['Omega_b'], 
                          h =  sims_parameters['h'], sigma8 = sims_parameters['sigma_8'], 
                          n_s = sims_parameters['n_s'], #w0 = cosmological_parameters['w0'], wa = cosmological_parameters['wa'], 
                          m_nu = [sims_parameters['m_nu']/3,sims_parameters['m_nu']/3,sims_parameters['m_nu']/3],mass_split='equal',
                          matter_power_spectrum='linear')
    z_hr = np.linspace(0, 10, 5001)
    d_hr = ccl.comoving_radial_distance(cosmo, 1./(1+z_hr))

    interpolated_distance_to_redshift = CubicSpline(d_hr, z_hr)
    interpolated_redshift_to_distance = CubicSpline(z_hr, d_hr)

    max_step_halocatalog = len(shells_info['z_far'])-int(shells_info['Step'][[shells_info['z_far']<max_redshift][0]][0])+1

    Lbox_Mpc = sims_parameters['dBoxSize Mpc/h']/ sims_parameters['h']



    
    
    # Initialize the final catalog dictionary
    final_cat = {
        'pix_16384_ring' : [],
        'x': [],
        'y': [],
        'z': [],
        'M': [],
        'redshift': [],
        'R': [],
        'ra': [],
        'dec': [],
        'redshift_hr': [],
        
        
        
    }
    
    count = 0
   
    # Iterate through each step in the halo catalog
    for i_ in frogress.bar(np.arange(0, max_step_halocatalog)):
        i = len(shells_info['Step']) - i_ - 1
        d_min = shells_info['cmd_near'][i]
        d_max = shells_info['cmd_far'][i]
        step = shells_info['Step'][i]

        # Load the snapshot data for the current step
        output_ = load_snapshot(halo_snapshots_path, step, Lbox_Mpc, sims_parameters,log10_mass_limit = log10_mass_limit)


        replicas_max = np.ceil(d_max / Lbox_Mpc).astype(int)
        replicas_min = np.ceil(d_min / Lbox_Mpc).astype(int)


        count_i = 0
        add = 0

        f = 1.0
        
        final_cat_x = []
        final_cat_y = []
        final_cat_z = []
        final_cat_M = []
        final_cat_R = []
        final_cat_redshift = []
        # Iterate through replicas
        for x_i in range(-replicas_max, replicas_max ):
            for y_i in range(-replicas_max , replicas_max ):
                for z_i in range(-replicas_max , replicas_max ):
                    may_intersect, close_box,far_box = may_intersect_sphere(x_i, y_i, z_i, Lbox_Mpc ,d_min, d_max)
                        
                    if may_intersect:
                            new_x = output_['x'] + x_i * Lbox_Mpc
                            new_y = output_['y'] + y_i * Lbox_Mpc
                            new_z = output_['z'] + z_i * Lbox_Mpc
                            r = np.sqrt(new_x**2 + new_y**2 + new_z**2)
                            mask = (r >= d_min) & (r < d_max)
                            if np.any(mask):
                                final_cat_x.append(new_x[mask])
                                final_cat_y.append(new_y[mask])
                                final_cat_z.append(new_z[mask])
                                final_cat_M.append(output_['M'][mask])
                                final_cat_R.append(output_['rhalf'][mask])
                                final_cat_redshift.append(interpolated_distance_to_redshift(r[mask]))
                            add += 1
         
                            
                            

        # Append collected data from this step to the final catalog
        if add>0:
            final_cat['pix_16384_ring'].append(hp.pixelfunc.vec2pix(8192 * 2, np.concatenate(final_cat_x), np.concatenate(final_cat_y), np.concatenate(final_cat_z), nest=False))
            final_cat['M'].append(np.concatenate(final_cat_M) )
            final_cat['R'].append(np.concatenate(final_cat_R) )
            final_cat['redshift'].append(np.concatenate(final_cat_redshift) * 10000)

    for key in final_cat:
        if isinstance(final_cat[key], list):
            final_cat[key] = np.concatenate(final_cat[key]) if final_cat[key] else np.array([])

    # Save the final catalog to a FITS file
    if os.path.exists(catalog_path):
        os.remove(catalog_path)

    print ('Done assemblying')
    
    fits_f = dict()
    fits_f['pix_16384_ring'] = (final_cat['pix_16384_ring']).astype('uint32')
    fits_f['log_M'] = (final_cat['M']).astype('uint16')  # this is Msun/h ---
    fits_f['R'] = (final_cat['R']).astype('uint16')
    fits_f['redshift'] = (final_cat['redshift']).astype('uint16')
    
    df = pd.DataFrame(fits_f)

    # Save it as a Parquet file
    df.to_parquet(catalog_path, index=False)

    # Clean up
    del final_cat
    gc.collect()



def reconstruct_inertia_tensor_halo_snapshot(df):
    """
    Reconstructs the original 6-component inertia tensor from
    inertia_auto and inertia_cross columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'inertia_auto' and 'inertia_cross' columns.

    Returns
    -------
    inertia : np.ndarray of shape (N, 6)
        Reconstructed inertia tensor per halo, in the order:
        [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    """

    # Convert stored lists back to arrays
    inertia_auto_log = np.stack(df['inertia_auto'].apply(np.array).to_numpy())
    inertia_cross_scaled = np.stack(df['inertia_cross'].apply(np.array).to_numpy())

    # Invert transformation for auto components: Ixx, Iyy, Izz
    principal_moments = 10**(inertia_auto_log.astype(float) / 1000) / 1e20
    Ixx, Iyy, Izz = principal_moments[:, 0], principal_moments[:, 1], principal_moments[:, 2]

    # Invert transformation for cross components: Ixy, Ixz, Iyz
    cross = inertia_cross_scaled.astype(float) / 10000 - 1
    Ixy = cross[:, 0] * np.sqrt(Ixx * Iyy)
    Ixz = cross[:, 1] * np.sqrt(Ixx * Izz)
    Iyz = cross[:, 2] * np.sqrt(Iyy * Izz)

    # Stack into (N, 6) array in the original order
    inertia = np.stack([Ixx, Ixy, Ixz, Iyy, Iyz, Izz], axis=1)
    return inertia

def reconstruct_angular_halo_snapshot(df):
    """
    Reconstructs the original angular vectors from the 'angular' column
    in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with an 'angular' column containing transformed int16 triplets.

    Returns
    -------
    angular : np.ndarray of shape (N, 3)
        Reconstructed angular vectors.
    """
    angular_encoded = np.stack(df['angular'].apply(np.array).to_numpy()).astype(float)
    
    # Invert the transformation:
    # x_original = sign * (10**(abs(x_encoded)/1000) - 1) / 1e20
    sign = np.sign(angular_encoded)
    magnitude = (10**(np.abs(angular_encoded) / 1000) - 1) / 1e20
    angular = sign * magnitude
    return angular


def recover_halo_mass(df):
    """
    Recover mass in M_sun/h from df['log10M'].

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'log10M' column (uint16, scaled).

    Returns
    -------
    mass : np.ndarray
        Mass in M_sun/h.
    """

    if 'log10M' in df.columns:
        mass_col = 'log10M'
    elif 'log_M' in df.columns:
        mass_col = 'log_M'
    else:
        raise KeyError("No mass column found (expected 'log10M' or 'log_M').")

    vals = df[mass_col].to_numpy()

    # detect whether there's a missing factor of 1000
    if np.min(vals) / 1000 < 10:
        # old sims: values were 1000x too large
        return 10**(vals / 1000. + 4)
    else:
        # new sims: already correct
        return 10**(vals / 1000.)

def recover_halo_redshift(df):
    """
    Recover redshift from scaled df['redshift'] values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'redshift' column (uint16, scaled by 10000).

    Returns
    -------
    z : np.ndarray
        Redshift values as float.
    """
    return df['redshift'].to_numpy() / 10000.
    
def recover_halo_centers(df):
    """
    Recover halo centers from df['halo_center'] as a (N, 3) numpy array in Mpc.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'halo_center' column (lists of 3 floats).

    Returns
    -------
    centers : np.ndarray of shape (N, 3)
        Halo center coordinates in Mpc.
    """
    return np.array(df['halo_center'].to_list())

def recover_fof_radius(df):
    """
    Recover FOF radius from df['rmax'] column, returning values in Mpc.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'rmax' column (uint16, stored in kpc).

    Returns
    -------
    rmax_mpc : np.ndarray
        FOF radius in Mpc.
    """
    return df['rmax'].to_numpy() / 1000.


def recover_halo_radec(df, nest=False):
    """
    Recover declination and right ascension from HEALPix pixel index.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'pix_16384_ring' column (HEALPix pixel index).

    nest : bool, optional
        If True, assumes nested ordering. Default is RING ordering (False).

    Returns
    -------
    dec, ra : np.ndarray
        Declination and Right Ascension in degrees.
    """
    theta, phi = hp.pix2ang(16384, df['pix_16384_ring'].to_numpy(), nest=nest)
    dec = np.degrees(0.5 * np.pi - theta)
    ra = np.degrees(phi)
    return ra, dec


def load_halo_catalog(
    path_halo_cat: str,
    colossus_pars: [Dict[str, float]],
    sims_parameters: Dict[str, float],
    halo_catalog_log10mass_cut: float,
    no_calib: True,
):
    """
    Load a halo catalog, convert FoF masses to M200c (via Colossus-based calibrator),
    apply a log10(M) cut, and return a compact dict.

    Parameters
    ----------
    path_halo_cat : str
        path to the halo catalog
    sims_parameters : dict
        Cosmology dict with at least Omega_m, Omega_b, h, sigma_8, n_s.
    halo_catalog_log10mass_cut : float
        Keep halos with log10(M/Msun h^-1) > cut after FoF->SO conversion.
    colossus_pars : dict, optional
        Parameters to register/select a Colossus cosmology (e.g., {"Om0":..., "Ob0":..., "H0":..., "sigma8":..., "ns":...}).
        If provided, a 'my_cosmo' entry is added/selected.


    Returns
    -------
    halos : dict of np.ndarray
        Keys: 'M' (M200c, Msun/h), 'z', 'ra', 'dec'.
    """
    # (Optional) configure Colossus cosmology
    colossus_cosmology.addCosmology('my_cosmo', colossus_pars)
    colossus_cosmology.setCosmology('my_cosmo')

    # Load catalog
    df = pd.read_parquet(path_halo_cat, engine="pyarrow")

    # Recover base fields
    M_fof = recover_halo_mass(df)          # Msun/h
    z     = recover_halo_redshift(df)
    ra, dec = recover_halo_radec(df)

    # Convert FoF -> SO (M200c); important for tSZ, neutral for large-scale baryonification
    if no_calib:
        print ('no calib')
        M200c = copy.deepcopy(M_fof)
    else:
        print ('extra calib [fof -> SO]')
        interpolator_calib = build_fof_to_m200c_interpolator(
            sims_parameters, colossus_cosmology)
        coords = np.column_stack([np.log10(M_fof), z])
        M200c  = interpolator_calib(coords)

    # Apply mass cut -----------------------------------
    mask = (np.log10(M200c) > halo_catalog_log10mass_cut)

    halos = {
        "M":   M200c[mask],
        "z":   z[mask],
        "ra":  ra[mask],
        "dec": dec[mask],
    }
    return halos


def mu(c):
    """
    Helper for NFW profiles: μ(c) = ln(1+c) − c/(1+c).

    Parameters
    ----------
    c : float or array-like
        Concentration parameter (>0).

    Returns
    -------
    float or ndarray
        μ(c).
    """
    return np.log(1 + c) - c / (1 + c)


def enclosed_overdensity(c, b=0.2, nc=0.652960):
    """
    Effective enclosed overdensity matched to a FoF linking length.

    Implements Eq. (10) of More et al. (2011), giving the mean overdensity
    inside the FoF boundary for an NFW halo of concentration `c`.

    Parameters
    ----------
    c : float or array-like
        NFW concentration.
    b : float, optional
        FoF linking length (in units of mean inter-particle separation). Default 0.2.
    nc : float, optional
        Numerical factor from More+2011 (≈ 0.652960).

    Returns
    -------
    float or ndarray
        Overdensity Δ_FoF/ρ̄ − 1 (i.e., 〈ρ〉/ρ̄ − 1).
    """
    return 3 * nc * b**-3 * mu(c) * (1 + c)**2 / c**2 - 1


def c_zhao2009(M, z):
    """
    Zhao et al. (2009) concentration–mass relation for 200c halos.

    Parameters
    ----------
    M : float or array-like
        Halo mass (Msun/h) defined at 200c.
    z : float or array-like
        Redshift.

    Returns
    -------
    float or ndarray
        Concentration c_200c.
    """
    M_pivot = 1e14  # Msun/h
    A, B, C = 4.67, -0.11, -1.0
    return A * (M / M_pivot)**B * (1 + z)**C


def m_nfw(r, rs, rho_s):
    """
    Enclosed mass for an NFW profile at radius r.

    Parameters
    ----------
    r : float or array-like
        Radius (same distance units as rs).
    rs : float
        NFW scale radius.
    rho_s : float
        NFW scale density.

    Returns
    -------
    float or ndarray
        M(<r) with units of rho_s * rs^3.
    """
    x = r / rs
    return 4 * np.pi * rho_s * rs**3 * (np.log(1 + x) - x / (1 + x))


def c_of_delta(delta, M_fof, z=0):
    """
    Placeholder c–M relation (Duffy et al. 2008-ish) for a given overdensity.

    Parameters
    ----------
    delta : float
        Target overdensity label (unused in this simplified stub).
    M_fof : float
        FoF mass (Msun/h).
    z : float, optional
        Redshift.

    Returns
    -------
    float
        Approximate concentration.

    Notes
    -----
    This is a rough stub (tuned to 200c); for accurate work, use a proper
    concentration model and/or conversion routine.
    """
    A, B, C = 5.71, -0.084, -0.47  # typical for 200c
    return A * (M_fof / 2e12)**B * (1 + z)**C


def infer_overdensity_from_fof(M_fof, M_fof_corr, b=0.2, z=0, tol=1e-3,
                               conc_corr_factor=True, cosmo='planck15'):
    """
    Infer the effective enclosed overdensity for a FoF halo (More+2011 approach).

    Steps:
      1) Get c_200c(M, z) from Colossus (Ishiyama+21 model), optionally scaled.
      2) Convert FoF linking length b to an effective FoF overdensity using
         the NFW mapping (More+2011 Eq. 10).

    Parameters
    ----------
    M_fof : float
        Original FoF mass (Msun/h). (Not used directly; kept for API symmetry.)
    M_fof_corr : float
        Corrected FoF mass used to evaluate the concentration (Msun/h).
    b : float, optional
        FoF linking length (default 0.2).
    z : float, optional
        Redshift.
    tol : float, optional
        Convergence tolerance for the fixed-point iteration on Δ (rarely needs many iters).
    conc_corr_factor : bool, optional
        If True, multiply Colossus c(M) by 0.6*z + 0.8 (empirical tweak).
    cosmo : str or dict, optional
        Colossus cosmology name or definition (default 'planck15').

    Returns
    -------
    Delta_eff : float
        Effective enclosed overdensity (w.r.t. mean density), i.e., 〈ρ〉/ρ̄ − 1.
    c_200c : float
        Concentration used (at 200c).

    Raises
    ------
    RuntimeError
        If the iteration fails to converge.
    """
    # get concentration from Colossus

    from colossus.halo import concentration

    corr = (0.6 * z + 0.8) if conc_corr_factor else 1.0
    c = corr * concentration.concentration(M_fof_corr, '200c', z, model='ishiyama21')

    # fixed-point iteration (usually converges in ≤ a few steps)
    Delta_guess = 200.0
    for _ in range(10):
        Delta_new = enclosed_overdensity(c, b=b)
        if abs(Delta_new - Delta_guess) < tol:
            return Delta_new, c
        Delta_guess = Delta_new

    raise RuntimeError("Overdensity iteration did not converge.")


def fof_to_mdelta(M_fof, Delta_fof, c_fof, Delta_target=200, ref_density='mean',
                  Omega_m=0.3, H0=70.0):
    """
    Convert FoF mass M_FoF to a spherical-overdensity mass M_Δ (NFW assumption).

    Treat the FoF halo as an NFW with concentration c_fof and mean enclosed density
    Δ_fof * ρ_ref_FoF (here taken as Δ_fof * ρ_m). Solve for R_Δ such that
    〈ρ(<R_Δ)〉 = Δ_target * ρ_ref, then return M_Δ = M(<R_Δ).

    Parameters
    ----------
    M_fof : float
        FoF mass (Msun/h).
    Delta_fof : float
        Enclosed overdensity of the FoF halo (w.r.t. mean matter density, ρ_m).
    c_fof : float
        Concentration of the FoF halo (R_FoF / r_s).
    Delta_target : float, optional
        Desired spherical overdensity (e.g., 200).
    ref_density : {"mean","crit"}, optional
        Reference density for the target mass definition (ρ_m or ρ_c).
    Omega_m : float, optional
        Matter density parameter at z relevant for the conversion (default 0.3).
    H0 : float, optional
        Hubble constant [km/s/Mpc] for ρ_c calculation (default 70).

    Returns
    -------
    M_delta : float
        Spherical-overdensity mass M_Δ_target (Msun/h).

    Raises
    ------
    RuntimeError
        If the root finding for R_Δ fails.

    Notes
    -----
    Units: G is taken as 4.30091e-9 Mpc⋅Msun⁻¹⋅(km/s)² so that
           ρ_c = 3 H0² / (8πG) has units Msun/Mpc³ when H0 is in km/s/Mpc.
    """
    # critical and reference densities
    G = 4.30091e-9  # Mpc * Msun^-1 * (km/s)^2
    rho_crit = 3.0 * (H0**2) / (8.0 * np.pi * G)  # Msun / Mpc^3
    rho_m = Omega_m * rho_crit
    rho_ref = rho_m if ref_density == 'mean' else rho_crit

    # FoF radius from mean-density definition (Δ_fof w.r.t. ρ_m)
    R_fof = (3.0 * M_fof / (4.0 * np.pi * Delta_fof * rho_m))**(1.0 / 3.0)

    # NFW params
    rs = R_fof / c_fof
    rho_s = M_fof / (4.0 * np.pi * rs**3 * mu(c_fof))

    # Solve for R_Δ: 〈ρ(<R)〉 = Δ_target * ρ_ref
    def mean_density_minus_target(r):
        return m_nfw(r, rs, rho_s) / ((4.0 / 3.0) * np.pi * r**3) - Delta_target * rho_ref

    # robust bracket (r must be > 0 and < few x R_fof)
    a, b = 1e-6 * rs, 10.0 * R_fof
    sol = root_scalar(mean_density_minus_target, bracket=[a, b], method='brentq')
    if not sol.converged:
        raise RuntimeError("Root finding for R_delta failed.")

    R_delta = sol.root
    M_delta = m_nfw(R_delta, rs, rho_s)
    return M_delta


def build_fof_to_m200c_interpolator(
    sims_parameters: dict,
    cosmo_colossus,
    logM_grid=(12.0, 15.5, 200),
    z_grid=(0.0, 1.5, 200),
):
    """
    Build or load a RegularGridInterpolator mapping (log10 M_fof, z) -> M_200c.
    Use `cache_path` (.npz) to save/reuse the expensive grid.
    """
    from colossus.halo import concentration  # local import to keep it optional

    lmin, lmax, nL = logM_grid
    zmin, zmax, nZ = z_grid
    lg = np.linspace(lmin, lmax, nL)
    zg = np.linspace(zmin, zmax, nZ)


    logM_mesh, z_mesh = np.meshgrid(lg, zg, indexing="ij")
    M_mesh = 10.0**logM_mesh
    M200c_grid = np.empty_like(M_mesh)

    # NOTE: this nested loop is heavy; consider parallelizing or precomputing offline.
    for i in range(logM_mesh.shape[0]):
        for j in range(logM_mesh.shape[1]):
            M_ = M_mesh[i, j]
            z_ = z_mesh[i, j]
            Delta_fof, c_fof = infer_overdensity_from_fof(  # your function from halos.py
                M_, M_, z=z_, conc_corr_factor=True, cosmo=cosmo_colossus
            )
            M200c_grid[i, j] = fof_to_mdelta(               # your function from halos.py
                M_,
                Delta_fof,
                c_fof,
                Delta_target=200,
                ref_density="crit",
                Omega_m=sims_parameters["Omega_m"],
                H0=sims_parameters["h"] * 100.0,
            )

    return RegularGridInterpolator((lg, zg), M200c_grid, bounds_error=False, fill_value=None)



def load_Flamingo_halo_catalog(path_catalog,sims_parameters,cosmo_bundle, type_cat = 'fof',halo_catalog_log10mass_cut = 13, max_z_halo_catalog = 1., no_calib = True):

    colossus_cosmology.addCosmology('my_cosmo', cosmo_bundle['colossus_params'])
    colossus_cosmology.setCosmology('my_cosmo')
    
    cat = fits.open(path_catalog)
    if type_cat == 'fof':

        if no_calib:
            M200c = cat[1].data['M']
        else:
            interpolator_calib = build_fof_to_m200c_interpolator(
            sims_parameters, colossus_cosmology)
            coords = np.column_stack([(np.log10(cat[1].data['M'])), cat[1].data['z']])
            M200c = interpolator_calib(coords)
    else:
        M200c = cat[1].data['M']

    z_all   = cat[1].data['z']
    ra_all  = cat[1].data['ra']
    dec_all = cat[1].data['dec']

    mask = (np.log10(M200c) > halo_catalog_log10mass_cut) & (z_all<max_z_halo_catalog)

    halos = {
        "M":   M200c[mask],
        "z":   z_all[mask],
        "ra":  ra_all[mask],
        "dec": dec_all[mask],
    }
    return halos
