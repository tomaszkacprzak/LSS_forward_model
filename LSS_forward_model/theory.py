import numpy as np
import camb
from camb import model
from copy import deepcopy
from scipy.interpolate import RegularGridInterpolator
import warnings
import frogress
import copy

warnings.filterwarnings(
    "ignore",
    message=r"EuclidEmulator2 emulates the non-linear correction.*"
)

# optional EuclidEmulator2
try:
    import euclidemu2 as ee2
    _EMU_OK = True
except Exception:
    _EMU_OK = False


class LimberTheory:
    """
    Compute Limber C_ell for nn, gg, ng, nk, gk, kk using an existing CAMBparams.

    Kernels:
      - W_gal (per tomographic n(z) and bias)
      - W_shear (per tomographic n(z))
      - W_cmb  (CMB lensing kernel; both 'short' z<=~12 and full-to-recombination)

    Nonlinear matter power:
      - 'euclidemu'  -> linear*boost (z<=2.5), else MEAD(2020); falls back to MEAD if emulator unusable
      - 'mead'       -> CAMB HMCode/Mead 2020
      - 'halofit'    -> CAMB Takahashi halofit

    All P(k,z) here are in (Mpc/h)^3, k in h/Mpc (standard LSS).
    """

    def __init__(self, pars: camb.CAMBparams, lmax=4000,
                 nonlinear='euclidemu', zgrid_max_lss=12.0, nz_lss=2401,
                 zgrid_max_kk=1100.0, nz_kk=140, feedback=0,kmax = 500):




        AccuracyBoost=1; max_eta_k=7000; lens_potential_accuracy=2;
       # AccuracyBoost=3; max_eta_k=50000; lens_potential_accuracy=4;
        
        pars.set_for_lmax(lmax=lmax,lens_potential_accuracy=lens_potential_accuracy, max_eta_k=max_eta_k)
        pars.set_accuracy(AccuracyBoost=AccuracyBoost,lSampleBoost=1, lAccuracyBoost=1)
        
        self.pars_in = pars
        self.lmax = int(lmax)
        self.nonlinear_req = nonlinear.lower()
        camb.set_feedback_level(feedback)

        # --- CAMB results (do not mutate user's params) ---
        self._pars_lin = copy.copy(pars)
        self._pars_lin.NonLinear = model.NonLinear_none
        self.results = camb.get_results(self._pars_lin)

        # basic numbers
        self.h = self._pars_lin.H0 / 100.0
        self.Omb = self._pars_lin.ombh2 / self.h**2
        self.Omc = self._pars_lin.omch2 / self.h**2
        self.Onu = getattr(self._pars_lin, "omnuh2", 0.0) / self.h**2
        self.Omm = self.Omb + self.Omc + self.Onu
        self.As  = self._pars_lin.InitPower.As
        self.ns  = self._pars_lin.InitPower.ns
        self.w0  = getattr(self._pars_lin.DarkEnergy, "w", -1.0)
        self.wa  = getattr(self._pars_lin.DarkEnergy, "wa", 0.0)
        self.mnu = float(self.Onu * 93.14 * self.h**2)  # ~eV

        # emulator params
        self._emu_params = dict(As=self.As, ns=self.ns, Omb=self.Omb, Omm=self.Omm,
                                h=self.h, mnu=self.mnu, w=self.w0, wa=self.wa)

        # --- P(k,z) interpolators ---
        # linear (wide z for kk)
        self._pk_lin = camb.get_matter_power_interpolator(
            self._pars_lin, nonlinear=False, hubble_units=True, k_hunit=True,
            kmax=kmax, zmax=zgrid_max_lss, zmin=0.0,
            var1=model.Transfer_tot, var2=model.Transfer_tot
        )
        # MEAD fallback (robust)
        self._pars_mead = copy.copy(pars)
        self._pars_mead.NonLinear = model.NonLinear_both
        self._pars_mead.NonLinearModel.set_params(halofit_version="mead")
        self._pk_mead = camb.get_matter_power_interpolator(
            self._pars_mead, nonlinear=True, hubble_units=True, k_hunit=True,
            kmax=kmax, zmax=zgrid_max_lss, zmin=0.0,
            var1=model.Transfer_tot, var2=model.Transfer_tot
        )
        # optional halofit
        self._pk_halofit = None
        if self.nonlinear_req == "halofit":
            self._pars_halofit = copy.copy(pars)
            self._pars_halofit.NonLinear = model.NonLinear_both
            self._pars_halofit.NonLinearModel.set_params(halofit_version="takahashi")
            self._pk_halofit = camb.get_matter_power_interpolator(
                self._pars_halofit, nonlinear=True, hubble_units=True, k_hunit=True,
                kmax=kmax, zmax=zgrid_max_lss, zmin=0.0,
                var1=model.Transfer_tot, var2=model.Transfer_tot
            )

        # Decide if we’ll use emulator
        self._use_emu = (self.nonlinear_req == "euclidemu") and _EMU_OK
        if self._use_emu:
            try:
                _ = ee2.get_boost(self._emu_params, 0.5, np.asarray([0.1]))
                print ('using euclid emu')
            except Exception:
                self._use_emu = False  # cosmology out of domain
        self._emu_interp = None         # RegularGridInterpolator for boost(z,k)
        self._emu_grid = None           # (Zgrid, Kgrid) used to build the interpolator
        # --- z, chi grids (Mpc) ---
        # LSS grid (~0..12)
        self.zs = np.linspace(0.0, float(zgrid_max_lss), int(nz_lss))
        self.chi  = self.results.comoving_radial_distance(self.zs)           # Mpc
        self.dchi = (self.chi[2:] - self.chi[:-2]) * 0.5
        self.zs   = self.zs[1:-1]
        self.chi  = self.chi[1:-1]
        self.a    = 1.0 / (1.0 + self.zs)
        self.chih = self.chi * self.h          # Mpc/h
        self.dchih= self.dchi * self.h

        # KK full grid to recombination (coarser)
        self.zs_kk = np.linspace(0.0, float(zgrid_max_kk), int(nz_kk))
        self.chi_kk  = self.results.comoving_radial_distance(self.zs_kk)     # Mpc
        self.dchi_kk = (self.chi_kk[2:] - self.chi_kk[:-2]) * 0.5
        self.zs_kk   = self.zs_kk[1:-1]
        self.chi_kk  = self.chi_kk[1:-1]
        self.a_kk    = 1.0 / (1.0 + self.zs_kk)
        self.chih_kk = self.chi_kk * self.h    # Mpc/h
        self.dchih_kk= self.dchi_kk * self.h






        #log version
        self.chistarlog = (self.results.conformal_time(0)- self.results.tau_maxvis)
        self.chislog    = np.linspace(0,self.chistarlog,400)
        self.zslog      = self.results.redshift_at_comoving_radial_distance(self.chislog)
        self.chislog    = self.results.comoving_radial_distance(self.zslog)
        self.dchislog   = (self.chislog[2:]-self.chislog[:-2])/2.
        self.dzslog     = (self.zslog[2:]-self.zslog[:-2])/2.
        self.chislog    = (self.chislog[1:-1])
        self.zslog      = (self.zslog[1:-1])
        self.alog       = 1./(1.+self.zslog)
        self.dzlog      = self.zslog[1]-self.zslog[0]
        gethzlog        = np.vectorize(self.results.h_of_z)
        self.hzlog      = gethzlog(self.zslog) # convert everything in units of Mpc/h

        self.chishlog    = self.chislog*self.h # Mpc/h now
        self.chistarhlog = self.chistarlog*self.h
        self.dchishlog   = self.dchislog*self.h


        
        

        # CMB source distance
        self.chi_star = self.results.conformal_time(0.0) - self.results.tau_maxvis  # Mpc
        self.chi_star_h = self.chi_star * self.h                                     # Mpc/h

        # kernel prefactor (matches your previous code style)
        c = 299792458.0  # m/s
        self.fc = 1.5 * (1000.0 * 100.0)**2 * self.Omm / (c**2)  # dimensionless factor in your convention

        # placeholders
        self.Wgal   = None  # (Nchi, Nbin)
        self.Wshear = None  # (Nchi, Nbin)
        self.Wcmb   = None  # (Nchi,) z<=~12
        self.Wcmb_full = None  # (Nchi_kk,) 0..z*
        self.Wcmba  = None
        self.Wcmba_full = None

    # ---------------- power spectrum backends ----------------

    def _P_lin(self, z, k):
        return self._pk_lin.P(z, k, grid=False)

    def _P_mead(self, z, k):
        return self._pk_mead.P(z, k, grid=False)

    def _P_halofit(self, z, k):
        if self._pk_halofit is None:
            return self._P_mead(z, k)
        return self._pk_halofit.P(z, k, grid=False)

    def _P_euclidemu_or_mead(self, z, k):
        """
        If emulator usable and z<=2.5: P = boost * P_lin; else MEAD.
        z, k can be scalars or arrays (broadcasted to (Nz, Nk)).
        """
        z = np.atleast_1d(np.asarray(z, float))
        k = np.atleast_1d(np.asarray(k, float))

        # start with MEAD everywhere (robust fallback)
        out = self._P_mead(z, k)

        if not self._use_emu:
            return out

        #mask = (z <= 2.5)
        #if not mask.any():
        #    return out

        #z_try = z[mask]
        pk_lin = self._P_lin(z, k)

        # Loop over z to get boosts
        # ee2.get_boost returns (kgrid, boost) matching input k
        for i, zz in enumerate(np.atleast_1d(z)):
            try:
                _, b = ee2.get_boost(self._emu_params, float(zz), k)
                if z.ndim == 0:
                    out = pk_lin * b
                elif k.ndim == 0:
                    out[mask][i] = pk_lin[i] * b
                else:
                    out[mask][i, :] = pk_lin[i] * b
            except Exception:
                pass  # keep MEAD fallback for that z
        return out

    EMU_K_MIN = 8.73e-3  # h/Mpc
    EMU_K_MAX = 9.41     # h/Mpc
    def _emu_grid_boost(self, z_grid, k_grid):
        """
        One batched EuclidEmulator2 call to fill a 2D table B[z,k].
        Handles dict/tuple/list/ndarray outputs; always returns (Nz, Nk).
        """
        obj = ee2.get_boost(self._emu_params, np.asarray(z_grid), np.asarray(k_grid))
    
        # Peel "(k_eval, boosts)" tuples to the 'boosts' payload
        if isinstance(obj, tuple) and len(obj) == 2:
            obj = obj[1]
    
        # Case A: dict mapping 0..Nz-1 -> array(Nk)
        if isinstance(obj, dict):
            # Sort keys to preserve z order used by the emulator
            rows = [np.asarray(obj[i], dtype=float) for i in sorted(obj.keys())]
            B = np.vstack(rows)
    
        # Case B: list/tuple of Nz arrays, or a single array (Nk,) if Nz==1
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise RuntimeError("EMU returned empty list/tuple for boosts")
            rows = [np.atleast_1d(np.asarray(row, dtype=float)) for row in obj]
            # If it's actually a single (Nk,) for Nz==1, make it (1, Nk)
            if len(rows) == 1:
                B = rows[0][None, :]
            else:
                # Ensure all rows have same length Nk
                Nk0 = rows[0].size
                if any(r.size != Nk0 for r in rows):
                    raise RuntimeError("EMU boost rows have inconsistent lengths")
                B = np.vstack(rows)
    
        # Case C: ndarray
        else:
            B = np.asarray(obj, dtype=float)
            if B.ndim == 1:
                # Single-z result -> make it (1, Nk)
                B = B[None, :]
            elif B.ndim == 2:
                pass
            else:
                # Try to coerce weird shapes
                B = B.reshape(len(z_grid), -1)
    
        # Final orientation check/transpose if needed
        Nz, Nk = len(z_grid), len(k_grid)
        if B.shape == (Nz, Nk):
            return B
        if B.shape == (Nk, Nz):
            return B.T
        # Last-resort reshape if sizes match (avoids IndexError on odd returns)
        if B.size == Nz * Nk:
            return B.reshape(Nz, Nk)
    
        raise RuntimeError(f"EMU boost grid has unexpected shape {B.shape}; expected {(Nz, Nk)}")
        
    def _ensure_emu_cache(self, lmax=None, Nz=80, Nk=160):
        """
        Build (and cache) a 2D boost(z,k) interpolator covering all k used up to lmax,
        restricted to the emulator domain in both z (<=2.5) and k.
        """
        if not self._use_emu:
            return
        if lmax is None:
            lmax = self.lmax
    
        # Effective k-range we actually need for Limber
        kmin_need = (1.0 + 0.5) / np.max(self.chih)      # h/Mpc  (ℓ=1)
        kmax_need = (lmax + 0.5) / np.min(self.chih)     # h/Mpc
    
        kmin = kmin_need
        kmax = kmax_need
        if not np.isfinite(kmin) or not np.isfinite(kmax) or kmin >= kmax:
            # Nothing useful in-domain; skip EMU
            self._use_emu = False
            return
    
        zmax = min([10,np.max(self.zs)])
        z_grid = np.linspace(0.0, zmax, Nz)
        k_grid = np.geomspace(kmin, kmax, Nk)
    
        # Skip rebuild if cache already matches
        if (self._emu_grid is not None and
            np.array_equal(self._emu_grid[0], z_grid) and
            np.array_equal(self._emu_grid[1], k_grid)):
            return
    
        # Build boost grid once
        B = self._emu_grid_boost(z_grid, k_grid)
    
        # Cache interpolator; set fill_value=np.nan to detect out-of-domain later
        self._emu_interp = RegularGridInterpolator(
            (z_grid, k_grid), B, bounds_error=False, fill_value=np.nan
        )
        self._emu_grid = (z_grid, k_grid)
        
    # ---------------- kernels ----------------

    def set_Wgal(self, nz_len: np.ndarray, bias: np.ndarray):
        """
        nz_len: array of shape (N, 1+Nbins): first col is z, others are n(z) (not necessarily normalized).
        bias:   array of shape (Nbins,) with constant bias per bin (or effective bias).
        """
        zgrid = np.asarray(nz_len[:, 0])
        nb = nz_len.shape[1] - 1
        if nb != bias.size:
            raise ValueError("bias length must match number of nz bins")
        W = np.zeros((self.zs.size, nb), dtype=float)
        for i in range(nb):
            nz = np.interp(self.zs, zgrid, nz_len[:, i+1], left=0.0, right=0.0)
            # Normalize to ∫dz n(z) = 1 (safe even if already normalized)
            norm = np.trapz(nz, self.zs)
            if norm > 0:
                nz /= norm
            # Limber weight for density contrast: n(z)*b * dz/dχ (your convention)
            W[:, i] = nz * bias[i] * (self.zs[1] - self.zs[0]) / self.dchih
        self.Wgal = W
        return W

    def set_Wshear(self, nz_src: np.ndarray):
        """
        nz_src: array of shape (N, 1+Nbins): first col z, others n(z) per source bin.
        Implements your (discrete) standard WL kernel on the LSS grid in Mpc/h units.
        """
        zgrid = np.asarray(nz_src[:, 0])
        nb = nz_src.shape[1] - 1
        Wg = np.zeros((self.zs.size, nb), dtype=float)

        for j in range(nb):
            nz = np.interp(self.zs, zgrid, nz_src[:, j+1], left=0.0, right=0.0)
            norm = np.trapz(nz, self.zs)
            if norm > 0:
                nz /= norm
            # discrete accumulation for lensing efficiency
            for i in range(self.zs.size):
                idx = np.where(self.zs >= self.zs[i])[0]
                if idx.size == 0:
                    continue
                # ∑ n(z_s) (dz/dχ) [(χ_s - χ)/χ_s] dχ   (your convention)
                tmp = np.sum(nz[idx] * (self.zs[1]-self.zs[0]) / self.dchih[idx] *
                             (self.chih[idx] - self.chih[i]) / self.chih[idx] * self.dchih[idx])
                Wg[i, j] = self.fc * self.chih[i] / self.a[i] * tmp
        self.Wshear = Wg
        return Wg

    def set_Wcmb(self):
        """
        CMB lensing kernels on both the short LSS grid and a long grid up to recombination.
        """
        # short grid (z<=~12)
        Wcmb = (self.chi_star_h - self.chih) / (self.chih**2 * self.chi_star_h)
        Wcmba = self.fc / self.a * (self.chi_star_h - self.chih) / self.chi_star_h * self.chih
        self.Wcmb, self.Wcmba = Wcmb, Wcmba

        # long grid for kk (to z*)
        Wcmb_full = (self.chi_star_h - self.chih_kk) / (self.chih_kk**2 * self.chi_star_h)
        Wcmba_full = self.fc / self.a_kk * (self.chi_star_h - self.chih_kk) / self.chi_star_h * self.chih_kk
        self.Wcmb_full, self.Wcmba_full = Wcmb_full, Wcmba_full
        return Wcmb, Wcmba

    # ---------------- C_ell (Limber) ----------------

    @staticmethod
    def _ell_array(lmax):
        ell = np.arange(lmax + 1, dtype=int)
        ell[0] = 1  # avoid l=0 exactly in k=(l+1/2)/χ
        return ell

    # --- replace your _cl_matrix with this version (works for all observables) ---
    def _cl_matrix(self, WA, WB, chi_h, dchi_h, z, lmax, nonlinear=True, show_progress=False):
        """
        Compute C_ell for two sets of kernels WA, WB (both [Nchi, Nbins]) with one
        P(z,k) vector per ℓ, then a single matmul over bins.
        Uses EMU cache (boost table) when nonlinear='euclidemu'.
        """
        Nchi = chi_h.size
        NbA  = WA.shape[1]
        NbB  = WB.shape[1]
        out  = np.zeros((NbA, NbB, lmax + 1), dtype=float)
    
        pref = dchi_h / (chi_h**2)  # [Nchi]
  
        # Decide backend once
        using_emu = nonlinear and self._use_emu and (self.nonlinear_req == "euclidemu")
        if using_emu:
            self._ensure_emu_cache(lmax=lmax)  # builds once if needed
    
        # Choose CAMB interpolator for non-EMU path (linear/mead/halofit)
        def _PK_nonemu(nonlinear_flag):
            if not nonlinear_flag:
                return self._pk_lin
            if self.nonlinear_req == "halofit" and (self._pk_halofit is not None):
                return self._pk_halofit
            return self._pk_mead
    
        for L in frogress.bar(range(1, lmax + 1)):
            k_vec = (L + 0.5) / chi_h  # [Nchi]
    
            if not nonlinear:
                P_vec = self._pk_lin.P(z, k_vec, grid=False)
            elif not using_emu:
                PK = _PK_nonemu(True)
                P_vec = PK.P(z, k_vec, grid=False)
            else:
                # EMU path: combine pk_lin * boost (in-domain) with MEAD elsewhere
                pk_lin_vec  = self._pk_lin.P(z, k_vec, grid=False)
                P_mead_vec  = self._pk_mead.P(z, k_vec, grid=False)
    
                # evaluate boost on all (z, k) pairs from cache
                pts = np.column_stack([z, k_vec])  # (Nchi, 2)
                boost = self._emu_interp(pts)      # may be NaN outside domain
    
                # valid where cache covers; else use MEAD
                valid = np.isfinite(boost)
                P_vec = P_mead_vec.copy()
                P_vec[valid] = pk_lin_vec[valid] * boost[valid]
    
            w = pref * P_vec                 # [Nchi]
            out[:, :, L] = WA.T @ (w[:, None] * WB)
    
        return out    
    # ---- exposed builders rewritten to use _cl_matrix ----
    
    def cl_nn(self, nonlinear=True, lmax=None, show_progress=False):
        if self.Wgal is None:
            raise RuntimeError("Call set_Wgal(...) first.")
        if lmax is None: lmax = self.lmax
        return self._cl_matrix(self.Wgal, self.Wgal, self.chih, self.dchih, self.zs,
                               lmax, nonlinear=nonlinear, show_progress=show_progress)
    
    def cl_gg(self, nonlinear=True, lmax=None, show_progress=False):
        if self.Wshear is None:
            raise RuntimeError("Call set_Wshear(...) first.")
        if lmax is None: lmax = self.lmax
        return self._cl_matrix(self.Wshear, self.Wshear, self.chih, self.dchih, self.zs,
                               lmax, nonlinear=nonlinear, show_progress=show_progress)
    
    def cl_ng(self, nonlinear=True, lmax=None, show_progress=False):
        if self.Wgal is None or self.Wshear is None:
            raise RuntimeError("Call set_Wgal(...) and set_Wshear(...) first.")
        if lmax is None: lmax = self.lmax
        return self._cl_matrix(self.Wgal, self.Wshear, self.chih, self.dchih, self.zs,
                               lmax, nonlinear=nonlinear, show_progress=show_progress)
    
    def cl_nk(self, nonlinear=True, lmax=None, show_progress=False):
        if self.Wgal is None or self.Wcmba is None:
            raise RuntimeError("Call set_Wgal(...) and set_Wcmb() first.")
        if lmax is None: lmax = self.lmax
        WB = self.Wcmba[:, None]  # treat CMB kernel as single-column "bin"
        out = self._cl_matrix(self.Wgal, WB, self.chih, self.dchih, self.zs,
                              lmax, nonlinear=nonlinear, show_progress=show_progress)
        # shape (Nb,1,lmax+1)
        return out
    
    def cl_gk(self, nonlinear=True, lmax=None, show_progress=False):
        if self.Wshear is None or self.Wcmba is None:
            raise RuntimeError("Call set_Wshear(...) and set_Wcmb() first.")
        if lmax is None: lmax = self.lmax
        WB = self.Wcmba[:, None]
        out = self._cl_matrix(self.Wshear, WB, self.chih, self.dchih, self.zs,
                              lmax, nonlinear=nonlinear, show_progress=show_progress)
        return out  # (Nb,1,lmax+1)
    
    def cl_kk(self, nonlinear=True, use_full_grid=True, lmax=None, show_progress=False):
        if self.Wcmb is None:
            raise RuntimeError("Call set_Wcmb() first.")
        if lmax is None: lmax = self.lmax
        if use_full_grid:
            W = self.Wcmb_full[:, None]  # single "bin"
            out = self._cl_matrix(W, W, self.chih_kk, self.dchih_kk, self.zs_kk,
                                  lmax, nonlinear=nonlinear, show_progress=show_progress)
        else:
            W = self.Wcmb[:, None]
            out = self._cl_matrix(W, W, self.chih, self.dchih, self.zs,
                                  lmax, nonlinear=nonlinear, show_progress=show_progress)
        # reshape to (1,1,lmax+1) for consistency
        return out.reshape(1, 1, lmax + 1)





    def cl_kk_log(self, nonlinear=True, lmax=None, zmax=1100.0):

        if lmax is None:
            lmax = self.lmax
    
        # --- chi grid: uniform in χ from 0..χ* ; central difference to drop endpoints
    
        # Weyl kernel on this grid (your old Wcmblog)
        Wcmblog       = ((self.chistarhlog-self.chishlog)/(self.chishlog**2*self.chistarhlog))
        Wcmbalog      = self.fc/self.alog*(self.chistarhlog-self.chishlog)/(self.chistarhlog)*self.chishlog
        self.Wcmblog  = Wcmblog  # Using Weyl 
        self.Wcmbalog = Wcmbalog # Not using Weyl
    
        kmax = 500
    
        # Weyl P(k,z) interpolator
        if self.nonlinear_req == "halofit":
            PKw = camb.get_matter_power_interpolator(
            self._pars_halofit, nonlinear=nonlinear, hubble_units=True, k_hunit=True,
            kmax=float(kmax), zmax=zmax, zmin=0.0,
            var1=model.Transfer_Weyl, var2=model.Transfer_Weyl
            )
        else:
            PKw = camb.get_matter_power_interpolator(
            self._pars_mead, nonlinear=nonlinear, hubble_units=True, k_hunit=True,
            kmax=float(kmax), zmax=zmax ,zmin=0.0,
            var1=model.Transfer_Weyl, var2=model.Transfer_Weyl
            )
        
        # main loop over ℓ
        out = np.zeros(lmax + 1, dtype=float)

        if zmax==0: zmax=1100
        idx = np.where( (self.zslog<zmax) )[0]        
        for l in range(1,lmax+1):
            k     = (l+0.5)/self.chishlog[idx]
            pp    = PKw.P(self.zslog[idx],k,grid=False)
            out[l] = np.dot(self.dchishlog[idx]*self.Wcmblog[idx]*self.Wcmblog[idx]/k**4/self.h**4,pp)*(l*(l+1))**2

        return out