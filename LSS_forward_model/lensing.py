import numpy as np
from typing import Sequence, Tuple
import pandas as pd  # only for typing/self; safe to remove if not needed
import pyccl as ccl  
from .maps import F_nla,convert_to_pix_coord,rotate_and_rebin
import healpy as hp



def addSourceEllipticity(
    self: "pd.DataFrame",
    es: "pd.DataFrame | np.ndarray",
    es_colnames: Sequence[str] = ("e1", "e2"),
    rs_correction: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Compose intrinsic source ellipticities with shear to obtain observed ellipticities.

    Uses the complex reduced-shear mapping:
        e = (e_s + g) / (1 + g* · e_s)          (exact, reduced-shear form)
    If `rs_correction=False`, uses the linearized approximation:
        e ≈ e_s + g

    Parameters
    ----------
    self : pandas.DataFrame
        Table with per-object shear components in columns "shear1", "shear2".
    es : DataFrame or array-like
        Intrinsic ellipticities for each row in `self`. If a DataFrame, it must
        have columns named by `es_colnames`; if a NumPy structured/record array,
        fields with those names must exist.
    es_colnames : (str, str), optional
        Names of the intrinsic ellipticity columns (default: ("e1", "e2")).
    rs_correction : bool, optional
        If True, apply the exact reduced-shear denominator (default True).
        If False, use the linear approximation e = e_s + g.
    inplace : bool, optional
        If True, overwrite self["shear1"], self["shear2"]; otherwise return (e1, e2).

    Returns
    -------
    (e1, e2) : tuple of ndarray, or None
        Observed ellipticity components. Returns None if `inplace=True`.

    Notes
    -----
    - Conventions: e = e1 + i e2, g = g1 + i g2, and the denominator uses g* (complex conjugate).
    - Assumes `len(self) == len(es)`.
    """
    # Safety check
    assert len(self) == len(es), "Length of `es` must match number of rows in `self`."

    # Build complex intrinsic ellipticity and shear
    if isinstance(es, pd.DataFrame):
        e1s = es[es_colnames[0]].to_numpy()
        e2s = es[es_colnames[1]].to_numpy()
    else:
        # array-like (could be structured array or 2D array)
        try:
            e1s = np.asarray(es[es_colnames[0]])
            e2s = np.asarray(es[es_colnames[1]])
        except Exception:
            es_arr = np.asarray(es)
            if es_arr.ndim == 2 and es_arr.shape[1] >= 2:
                e1s, e2s = es_arr[:, 0], es_arr[:, 1]
            else:
                raise ValueError("`es` must provide two components matching `es_colnames`.")
    es_c = e1s + 1j * e2s

    g = np.asarray(self["shear1"]) + 1j * np.asarray(self["shear2"])

    # Compose
    e = es_c + g
    if rs_correction:
        e = (es_c + g) / (1.0 + np.conjugate(g) * es_c)

    if inplace:
        self["shear1"] = e.real
        self["shear2"] = e.imag
        return None
    else:
        return e.real, e.imag


def apply_random_rotation(e1_in: np.ndarray, e2_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply independent random spin-2 rotations to ellipticities (e1, e2).

    Draws a random phase θ ~ U(0, 2π) per object and rotates:
        [e1']   [ cos θ   sin θ] [e1]
        [e2'] = [-sin θ   cos θ] [e2]
    (θ here already includes the spin-2 factor; equivalently use φ and set θ=2φ.)

    Parameters
    ----------
    e1_in, e2_in : array-like
        Input ellipticity components of equal length.

    Returns
    -------
    (e1_out, e2_out) : tuple of ndarray
        Rotated ellipticity components.

    Notes
    -----
    - For reproducibility, consider controlling the RNG outside this function
      (e.g., pass precomputed angles or set NumPy’s seed before calling).
    """
    e1_in = np.asarray(e1_in)
    e2_in = np.asarray(e2_in)
    if e1_in.shape != e2_in.shape:
        raise ValueError("e1_in and e2_in must have the same shape.")

    # Random spin-2 phase per object
    rot_angle = np.random.random(size=e1_in.shape) * 2.0 * np.pi
    c = np.cos(rot_angle)
    s = np.sin(rot_angle)

    e1_out = e1_in * c + e2_in * s
    e2_out = -e1_in * s + e2_in * c
    return e1_out, e2_out


def make_WL_sample(ngal_glass, zeff_glass, cosmo_bundle, sims_parameters, nside_maps, fields, cats_Euclid, SC_corrections =None,do_catalog = False, include_SC = True, compact_savings = False):
    if include_SC:
        corr_variance_array =  [  SC_corrections['corr_variance_fit'][tomo](sims_parameters['bias_sc'][tomo])        for tomo in range(len(ngal_glass))]
        coeff_kurtosis_array = [  SC_corrections['coeff_kurtosis_fit'][tomo](sims_parameters['bias_sc'][tomo])       for tomo in range(len(ngal_glass))]
        A_corr_array = [  SC_corrections['A_corr_fit'][tomo](sims_parameters['bias_sc'][tomo])                       for tomo in range(len(ngal_glass))]
    else:
        corr_variance_array = np.ones(len(ngal_glass))
        coeff_kurtosis_array = np.zeros(len(ngal_glass))
        A_corr_array  = np.ones(len(ngal_glass))
        sims_parameters['bias_sc'] = np.zeros(len(ngal_glass))
    
    
    
    
    kappa_tot  = np.zeros((len(ngal_glass), 12*nside_maps**2))
    g1_tot     = np.zeros((len(ngal_glass), 12*nside_maps**2))
    g2_tot     = np.zeros((len(ngal_glass), 12*nside_maps**2))
    d_tot      = np.zeros((len(ngal_glass), 12*nside_maps**2))
    
    
    # load each lightcone output in turn and add it to the simulation
    # note: I added a -sign to gamma to match data conventions later
    for tomo in range(len(ngal_glass)):
        for i in (range(len(fields['gamma']))):       
            C1 = 5e-14
            rho_crit0_h2 = ccl.physical_constants.RHO_CRITICAL
            rho_c1 = C1 * rho_crit0_h2
            IA_f = F_nla(z=zeff_glass[i],
             om0=sims_parameters['Omega_m'],
             A_ia=sims_parameters['A_IA'], rho_c1=rho_c1, eta=sims_parameters['eta_IA'], z0=0.67,
             cosmo=cosmo_bundle['cosmo_pyccl'])
            
            g1_tot[tomo] += ngal_glass[tomo,i] * (-fields['gamma'][i][0].real-fields['IA_shear'][i][0].real*IA_f) * (1 + sims_parameters['bias_sc'][tomo] * fields['density'][i])
            g2_tot[tomo] += ngal_glass[tomo,i] * (-fields['gamma'][i][0].imag-fields['IA_shear'][i][0].imag*IA_f) * (1 + sims_parameters['bias_sc'][tomo] * fields['density'][i])
            d_tot[tomo]  += ngal_glass[tomo,i] * (1 + sims_parameters['bias_sc'][tomo] * fields['density'][i] )
       
    
    sims_parameters.setdefault('rot', 0)
    sims_parameters.setdefault('delta_rot', 0)

    maps_sim = dict()
    for tomo in range(len(ngal_glass)):
        maps_sim[tomo] = dict()



        pix_ = convert_to_pix_coord(cats_Euclid[tomo]['ra'],cats_Euclid[tomo]['dec'], nside=nside_maps*2)
        pix = rotate_and_rebin(pix_, nside_maps, sims_parameters['rot'], delta_=sims_parameters['delta_rot'])
             

        # source clustering term ~
        f = 1./np.sqrt(d_tot[tomo])
        f = f[pix]
    
    
        n_map = np.zeros(hp.nside2npix(nside_maps))
        n_map_sc = np.zeros(hp.nside2npix(nside_maps))
    
                        
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    
    
        n_map_sc[unique_pix] += np.bincount(idx_rep, weights=cats_Euclid[tomo]['w']/f**2)
        n_map[unique_pix] += np.bincount(idx_rep, weights=cats_Euclid[tomo]['w'])
    
        g1_ = g1_tot[tomo][pix]
        g2_ = g2_tot[tomo][pix]
    
    
        es1,es2 = apply_random_rotation(cats_Euclid[tomo]['e1']/f, cats_Euclid[tomo]['e2']/f)
        es1_ref,es2_ref = apply_random_rotation(cats_Euclid[tomo]['e1'], cats_Euclid[tomo]['e2'])
        es1a,es2a = apply_random_rotation(cats_Euclid[tomo]['e1']/f, cats_Euclid[tomo]['e2']/f)
    
    
        #x1_sc,x2_sc = addSourceEllipticity({'shear1':g1_,'shear2':g2_},{'e1':es1,'e2':es2},es_colnames=("e1","e2"))
    
    
        e1r_map = np.zeros(hp.nside2npix (nside_maps))
        e2r_map = np.zeros(hp.nside2npix (nside_maps))
        e1r_map0 = np.zeros(hp.nside2npix(nside_maps))
        e2r_map0 = np.zeros(hp.nside2npix(nside_maps))
        e1r_map0_ref = np.zeros(hp.nside2npix(nside_maps))
        e2r_map0_ref = np.zeros(hp.nside2npix(nside_maps))
        g1_map = np.zeros(hp.nside2npix(nside_maps))
        g2_map = np.zeros(hp.nside2npix(nside_maps))
    
        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
    
    
        e1r_map[unique_pix] += np.bincount(idx_rep, weights=es1*cats_Euclid[tomo]['w'])
        e2r_map[unique_pix] += np.bincount(idx_rep, weights=es2*cats_Euclid[tomo]['w'])
    
        e1r_map0[unique_pix] += np.bincount(idx_rep, weights=es1a*cats_Euclid[tomo]['w'])
        e2r_map0[unique_pix] += np.bincount(idx_rep, weights=es2a*cats_Euclid[tomo]['w'])
    
        e1r_map0_ref[unique_pix] += np.bincount(idx_rep, weights=es1_ref*cats_Euclid[tomo]['w'])
        e2r_map0_ref[unique_pix] += np.bincount(idx_rep, weights=es2_ref*cats_Euclid[tomo]['w'])
    
    
        mask_sims = n_map_sc != 0.
        e1r_map[mask_sims]  = e1r_map[mask_sims]/(n_map_sc[mask_sims])
        e2r_map[mask_sims] =  e2r_map[mask_sims]/(n_map_sc[mask_sims])
        e1r_map0[mask_sims]  = e1r_map0[mask_sims]/(n_map_sc[mask_sims])
        e2r_map0[mask_sims] =  e2r_map0[mask_sims]/(n_map_sc[mask_sims])
        e1r_map0_ref[mask_sims]  = e1r_map0_ref[mask_sims]/(n_map[mask_sims])
        e2r_map0_ref[mask_sims] =  e2r_map0_ref[mask_sims]/(n_map[mask_sims])
    
    
    
        var_ =  e1r_map0_ref**2+e2r_map0_ref**2
    
    
        #'''
        e1r_map   *= 1/(np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_))
        e2r_map   *= 1/(np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_))
        e1r_map0  *= 1/(np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_))
        e2r_map0  *= 1/(np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_))
    
    
    
        
        
        #'''
        g1_map[unique_pix] += np.bincount(idx_rep, weights= g1_*cats_Euclid[tomo]['w'])
        g2_map[unique_pix] += np.bincount(idx_rep, weights= g2_*cats_Euclid[tomo]['w'])
    
    
    
        g1_map[mask_sims]  = g1_map[mask_sims]/(n_map_sc[mask_sims])
        g2_map[mask_sims] =  g2_map[mask_sims]/(n_map_sc[mask_sims])

        if compact_savings:
            e1_ = ((g1_map* sims_parameters['dm'][tomo] +e1r_map0))[mask_sims]
            e2_ = ((g2_map* sims_parameters['dm'][tomo] +e2r_map0))[mask_sims]
            g1_ = g1_map[mask_sims]
            g2_ = g2_map[mask_sims]
            e1n_ = ( e1r_map)[mask_sims]
            e2n_ = ( e2r_map)[mask_sims]
            idx_ = np.arange(len(mask_sims))[mask_sims]
        
            maps_sim[tomo] =     {'g1_map':g1_,'g2_map':g2_,'e1':e1_,'e2':e2_,'e1n':e1n_,'e2n':e2n_,
                                    'idx_':idx_}

            
        else:
            e1_ = ((g1_map* sims_parameters['dm'][tomo] +e1r_map0))#[mask_sims]
            e2_ = ((g2_map* sims_parameters['dm'][tomo] +e2r_map0))#[mask_sims]
            e1n_ = ( e1r_map)#[mask_sims]
            e2n_ = ( e2r_map)#[mask_sims]
           # idx_ = np.arange(len(mask_sims))[mask_sims]
        
            maps_sim[tomo] =     {'g1_map':g1_map,'g2_map':g2_map,'e1':e1_,'e2':e2_,'e1n':e1n_,'e2n':e2n_,
                                    'e1r_map0_ref':e1r_map0_ref,
                                    'e2r_map0_ref':e2r_map0_ref,
                                    'var_':var_}     
    
        if do_catalog:
    
            cats_sim = dict()
            # make a catalog ---------------------------------------------------------------------------------------------------------------------------------
            SC_per_pixel_correction_noise  = f**2/((np.sqrt(A_corr_array[tomo]*corr_variance_array[tomo])) * np.sqrt((1+coeff_kurtosis_array[tomo]*var_)))[pix]
            
            # the f**2 applied to g1,g2 is the normalisation missing in the g1_tot,g2_tot ---------------------------------------------------
            e1_SC = sims_parameters['dm'][tomo]*g1_*f**2+es1a*SC_per_pixel_correction_noise
            e2_SC = sims_parameters['dm'][tomo]*g2_*f**2+es2a*SC_per_pixel_correction_noise
            #e1_SC,e2_SC = addSourceEllipticity({'shear1':g1_,'shear2':g2_},{'e1':es1a*SC_per_pixel_correction_noise,'e2':es2a*SC_per_pixel_correction_noise},es_colnames=("e1","e2"))
            cats_sim[tomo] =  {'ra':cats_Euclid[tomo]['ra'],'dec':cats_Euclid[tomo]['dec'],'e1':e1_SC,'e2':e2_SC,'w':cats_Euclid[tomo]['w']}
        
        else:
            cats_sim = None
            
    return maps_sim, cats_sim