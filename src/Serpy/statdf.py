import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import map_coordinates, convolve
from numba import njit
from typing import Tuple, Optional
from tqdm import tqdm


def _generate_band_coords(angle_rad: float, length: int, band_thickness: int = 1, step: float = 1.0):
    """
    Generate relative (Y, X) coordinates for an oriented band within a local window.

    This function defines the sampling coordinates of a thin band (or “profile”) oriented
    at a given angle `angle_rad`, centered on the current pixel within a square window
    of size `2 * length + 1`. These coordinates are later used to interpolate image
    intensities along this orientation, in order to compute directional local statistics
    such as mean and standard deviation.

    Parameters
    ----------
    angle_rad : float
        Orientation angle of the band, in radians.
        0 corresponds to a horizontal band (along X-axis), and π/2 to a vertical one.
    length : int
        Half-length of the band (in pixels), defining the window extent along its main direction.
    band_thickness : int, optional
        Thickness of the band (in pixels) along the perpendicular direction.
        Values greater than 1 create a multi-pixel-wide “strip” instead of a single line.
        Default is 1.
    step : float, optional
        Sampling step along the longitudinal axis of the band.
        Smaller values (<1) enable oversampling for smoother interpolation.
        Default is 1.0.

    Returns
    -------
    Y, X : ndarray
        Two 2D arrays of the same shape giving the relative row (`Y`) and column (`X`)
        coordinates of the oriented band.  
        These coordinates are centered around (0, 0) and should be shifted by the local
        window center before being passed to `scipy.ndimage.map_coordinates`.
    """
    
    # This vector defines how the band extends in its main direction.
    dir_x = np.cos(angle_rad)
    dir_y = np.sin(angle_rad)
    # This vector points 90° counterclockwise from (dir_x, dir_y),
    # and is used to control the band thickness (width across its normal direction).
    norm_x = -np.sin(angle_rad)
    norm_y =  np.cos(angle_rad)
    
    #Each point is given by (x, y) = t \cdot \vec{d} + o \cdot \vec{n}
    t = np.arange(-length, length + step, step, dtype=np.float32)
    offsets = np.linspace(-(band_thickness - 1) / 2, (band_thickness - 1) / 2, band_thickness, dtype=np.float32)
    X = t[None, :] * dir_x + offsets[:, None] * norm_x
    Y = t[None, :] * dir_y + offsets[:, None] * norm_y
    return Y, X

def compute_directional_stats(data: np.ndarray, angles: np.ndarray, window_size: int = 9, band_thickness: int = 1, step: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local directional statistics (mean and standard deviation)
    of an image within a sliding window along multiple orientations.

    This function extracts local intensity profiles inside each window
    oriented at several discrete angles, and computes their mean and
    standard deviation. The result is a set of orientation-resolved
    statistics used to estimate directional visibility or dark-field signals.

    Parameters
    ----------
    data : np.ndarray
        2D input image (e.g., reference or sample intensity map).
    angles : np.ndarray
        Array of orientation angles (in radians) along which directional
        profiles are computed.
    window_size : int, optional
        Size of the local square window (must be odd).
        Default is 9 (i.e., a 9×9 window).
    band_thickness : int, optional
        Number of parallel lines across the band (controls the local
        thickness of the directional region). Default is 1.
    step : float, optional
        Sampling step along the band direction. Smaller steps increase
        sampling density and smoothness. Default is 0.5.

    Returns
    -------
    mean_result : np.ndarray
        3D array of shape (H, W, N_ang + 1).
        - index 0 stores the isotropic local mean (average over the full window).
        - index  1..N_ang store the directional means for each angle.
    std_result : np.ndarray
        3D array of shape (H, W, N_ang + 1).
        - index 0 stores the isotropic local standard deviation.
        - index 1..N_ang store the directional standard deviations.
    """
    
    data = np.asarray(data, dtype=np.float32)
    
    pad = window_size // 2
    padded = np.pad(data, pad, mode='reflect')
    
    windows = sliding_window_view(padded, (window_size, window_size))
    
    H, W = data.shape
    N_ang = len(angles)
    
    mean_result = np.zeros((H, W, N_ang + 1), dtype=np.float32)
    std_result  = np.zeros((H, W, N_ang + 1), dtype=np.float32)
    
    # Compute isotropic (non-directional) mean and std for index 0
    mean_result[..., 0] = np.mean(windows, axis=(-2, -1))
    std_result[..., 0]  = np.std(windows, axis=(-2, -1))
    
    center = window_size // 2
    flat_windows = windows.reshape(-1, window_size, window_size)

    for k, angle_rad in enumerate(tqdm(angles, desc="Computing directional stats", unit="angle")):

        # Generate relative coordinates of the oriented band (within the window)
        Yrel, Xrel = _generate_band_coords(angle_rad, center, band_thickness, step=step)
        coords = np.stack([Yrel + center, Xrel + center], axis=0).reshape(2, -1)
        
        # For each window, interpolate pixel values along the band at subpixel precision
        # order=3 → cubic interpolation; mode='reflect' → smooth edges
        interpolated = np.array([
            map_coordinates(win, coords, order=3, mode='reflect')
            for win in flat_windows
        ], dtype=np.float32)
        
        # Compute mean and std of intensity profiles within each band
        profile_means = interpolated.mean(axis=1).reshape(H, W)
        profile_stds  = interpolated.std(axis=1).reshape(H, W)
        mean_result[..., k + 1] = profile_means
        std_result[...,  k + 1] = profile_stds
    return mean_result, std_result


def _df(std_r, std_s, T):
    """
    Compute the local dark-field (DF) signal from reference and sample statistics.

    The dark-field quantifies the reduction of local visibility (or contrast)
    in the presence of a scattering sample. It is derived from the ratio of
    local standard deviations (sample vs. reference) and corrected by the
    transmission factor `T`.

    Parameters
    ----------
    std_r : np.ndarray
        Local standard deviation of the reference image (no sample).
    std_s : np.ndarray
        Local standard deviation of the sample image (with scattering).
    T : np.ndarray
        Transmission map, defined as mean_sample / mean_reference.

    Returns
    -------
    np.ndarray
        Dark-field signal, dimensionless, with values in [0, 1] in most cases.
        - A value near 0 → minimal scattering (visibility preserved).
        - A value near 1 → strong scattering (visibility strongly reduced).

    Notes
    -----
    The formula implements:
        DF = | 1 - (std_s / std_r) * (1 / T) |
    """
    
    return np.abs(1.0 - (std_s / (std_r + 1e-12)) * (1.0 / (T + 1e-12)))

def compute_df(std_ref: np.ndarray, std_sample: np.ndarray, mean_ref: np.ndarray, mean_sample: np.ndarray):
    """
    Compute the directional dark-field (DF) and transmission maps
    from local statistics of reference and sample images.

    This function combines local standard deviations and mean values
    (previously obtained through directional statistical analysis)
    to estimate both:
      - the **transmission** map (attenuation-related contrast), and
      - the **dark-field** map (visibility reduction caused by scattering).

    Parameters
    ----------
    std_ref : np.ndarray
        Local standard deviation of the reference image (no sample).
    std_sample : np.ndarray
        Local standard deviation of the sample image (with scattering).
    mean_ref : np.ndarray
        Local mean of the reference image.
    mean_sample : np.ndarray
        Local mean of the sample image.

    Returns
    -------
    Df_result : np.ndarray
        Local dark-field signal computed from standard deviations and transmission.
        Values are typically within [0, 1], representing the relative scattering strength.
    transmission : np.ndarray
        Local transmission map, defined as `mean_sample / mean_ref`.
    """
    # Compute the local transmission map (sample intensity normalized by reference)
    transmission = mean_sample / (mean_ref + 1e-12)
    
    Df_result = _df(std_ref, std_sample, transmission)
    
    # Replace non-finite values (NaN or inf) with a small constant to ensure numerical stability
    Df_result = np.where(~np.isfinite(Df_result), 1e-6, Df_result)
    transmission = np.where(~np.isfinite(transmission), 1e-6, transmission)
    
    return Df_result, transmission

@njit
def _fit_ellipse_numba(x, y):
    """
    Fit a conic (ellipse) to 2D points using the direct least-squares method.

    The conic has the algebraic form:
        a x^2 + b x y + c y^2 + d x + f y + g = 0
    with the ellipse constraint 4ac - b^2 > 0 enforced via the generalized
    eigenproblem using a constraint matrix.

    Parameters
    ----------
    x, y : 1D arrays (float32)
        Point coordinates on the polar DF plot (already projected into x/y).

    Returns
    -------
    coeffs : (6,) float32
        Conic coefficients [a, b, c, d, f, g]. NaNs if fitting fails.

    Notes
    -----
    D1 = [x^2, x y, y^2], D2 = [x, y, 1]
    S1 = D1^T D1, S2 = D1^T D2, S3 = D2^T D2
    T = - S3^{-1} S2^T,  M = S1 + S2 T
    C is the ellipse-constraint matrix in the quadratic terms basis.
    Solve eig(M C^{-1}) and select the vector v with 4 v0 v2 - v1^2 > 0.
    """
    
    
    D1 = np.empty((x.shape[0], 3), dtype=np.float32)
    D2 = np.empty((x.shape[0], 3), dtype=np.float32)
    for k in range(x.shape[0]):
        D1[k, 0] = x[k] * x[k]   # x^2
        D1[k, 1] = x[k] * y[k]   # x y
        D1[k, 2] = y[k] * y[k]   # y^2
        D2[k, 0] = x[k]          # x
        D2[k, 1] = y[k]          # y
        D2[k, 2] = 1.0           # 1
        
    # Scatter matrices    
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    
    try:
        # Eliminate linear terms: T maps quadratic coeffs -> linear/constant coeffs
        T = -np.linalg.inv(S3) @ S2.T

        # Quadratic block after elimination
        M = S1 + S2 @ T

        # Constraint matrix C for ellipse (in quadratic term basis [a, b, c])
        C = np.array(((0, 0, 2),
                      (0, -1, 0),
                      (2, 0, 0)), dtype=np.float32)

        # Solve generalized eigenproblem via C^{-1} M
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)

        # Pick eigenvector that satisfies the ellipse condition 4ac - b^2 > 0
        for idx in range(3):
            v = eigvec[:, idx]  # [a, b, c]
            if 4 * v[0] * v[2] - v[1]**2 > 0:
                # Full conic coefficients [a, b, c, d, f, g]
                return np.concatenate((v, T @ v))
    except:
        # Any numerical issue → return NaNs
        return np.full(6, np.nan, dtype=np.float32)

    # No valid eigenvector found → return NaNs
    return np.full(6, np.nan, dtype=np.float32)


@njit
def _cart_to_pol_numba(coeffs):
    """
    Convert conic coefficients to ellipse geometric parameters.

    Parameters
    ----------
    coeffs : array-like, shape (6,)
        Conic coefficients [a, b, c, d, f, g] of the ellipse.

    Returns
    -------
    x0, y0 : float
        Center of the ellipse.
    ap, bp : float
        Semi-major and semi-minor axes (ap >= bp). Zero if invalid.
    e : float
        Eccentricity (in [0, 1)).
    phi : float
        Orientation angle of the major axis in [0, π).

    Notes
    -----
    Follows standard conic → geometric-ellipse conversion:
    - Center from partial derivatives.
    - Axes from quadratic form eigen-structure and scale.
    - Orientation from the cross-term b.
    """
    
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]
    
    
    # Denominator for center and axes (must be negative for ellipse)
    den = b**2 - a*c
    if den > 0:
        # Not an ellipse: return neutral values
        return np.nan, np.nan, 0.0, 0.0, 0.0, 0.0

    # Ellipse center (x0, y0)
    x0 = (c*d - b*f) / den
    y0 = (a*f - b*d) / den

    # Scale factor for axis lengths
    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)

    # Eigen-structure term
    fac = np.sqrt((a - c)**2 + 4*b**2)

    # Semi-axes before ordering
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Ensure ap >= bp (swap if needed)
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap
        
        
    # Eccentricity
    r = (bp / (ap + 1e-12))**2
    if r > 1:
        r = 1 / r
    e = np.sqrt(max(0.0, 1 - r))

    # Degenerate guard
    if ap == 0 or bp == 0:
        return x0, y0, 0.0, 0.0, 0.0, 0.0

    # Orientation (major-axis angle)
    if b == 0:
        phi = 0.0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        phi += np.pi/2

    # Normalize to [0, π)
    phi = phi % np.pi
    
    phi +=np.pi/2
    phi = phi % np.pi
    
    return x0, y0, ap, bp, e, phi

@njit
def _fit_all_ellipses(Df_results, sin_values, cos_values):
    """
    Fit an ellipse to the polar DF at every pixel.

    Parameters
    ----------
    Df_results : (H, W, A) float32
        Directional DF values per pixel; channel 0 is global (isotropic) and
        channels 1..A-1 are the angular samples used for the polar fit.
    sin_values, cos_values : (A-1,) float32
        Precomputed sin(theta_k) and cos(theta_k) for the A-1 angular samples.

    Returns
    -------
    major_axis : (H, W) float32
        Semi-major axis per pixel (ap).
    minor_axis : (H, W) float32
        Semi-minor axis per pixel (bp).
    phi : (H, W) float32
        Orientation of the major axis in [0, π).

    Notes
    -----
    For each pixel, we build the (x, y) polar samples as:
        x_k = DF_k * sin(theta_k),  y_k = DF_k * cos(theta_k)
    and fit a conic using `_fit_ellipse_numba`, then convert to geometric
    parameters with `_cart_to_pol_numba`.
    """
    
    H, W, A = Df_results.shape
    
    major_axis = np.zeros((H, W), dtype=np.float32)
    minor_axis = np.zeros((H, W), dtype=np.float32)
    phi = np.zeros((H, W), dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            # Use angular channels only (skip global index 0)
            df = Df_results[i, j, 1:]
            
            try:
                # Polar-to-Cartesian embedding of the DF profile
                x = df * sin_values
                y = df * cos_values
                
                # Algebraic conic fit
                coeffs = _fit_ellipse_numba(x, y)
                
                
                if not np.any(np.isnan(coeffs)):
                    # Convert algebraic conic to ellipse geometry
                    x0, y0, maj, min_, e, ph = _cart_to_pol_numba(coeffs)
                    major_axis[i, j] = maj
                    minor_axis[i, j] = min_
                    phi[i, j]        = ph
                else:
                    # Fallback for invalid/unstable fits
                    major_axis[i, j] = 0.0
                    minor_axis[i, j] = 0.0
                    phi[i, j]        = 0.0

            except:
                # Numerical guard on per-pixel fitting
                major_axis[i, j] = 0.0
                minor_axis[i, j] = 0.0
                phi[i, j]        = 0.0
                
    return major_axis, minor_axis, phi

def fit_ellipses(Df_results: np.ndarray, sin_angles: np.ndarray, cos_angles: np.ndarray):
    """
    Fit ellipses to directional dark-field (DF) profiles for all pixels.

    This is a wrapper around the Numba-accelerated `_fit_all_ellipses` function.
    It converts all inputs to `float32` for efficiency and numerical consistency,
    then performs per-pixel ellipse fitting in the (x, y) polar space of the
    directional DF signal.

    Parameters
    ----------
    Df_results : np.ndarray
        3D array of directional DF values with shape (H, W, A),
        where H and W are the spatial dimensions and A is the number of
        angular samples + 1 (the first channel is usually the isotropic term).
    sin_angles : np.ndarray
        Array of precomputed sine values of the sampled orientations (shape A−1).
    cos_angles : np.ndarray
        Array of precomputed cosine values of the sampled orientations (shape A−1).

    Returns
    -------
    major_axis : np.ndarray
        2D map (H, W) of fitted semi-major axis lengths per pixel.
    minor_axis : np.ndarray
        2D map (H, W) of fitted semi-minor axis lengths per pixel.
    phi : np.ndarray
        2D map (H, W) of ellipse orientation angles in radians, within [0, π).
    """
    return _fit_all_ellipses(Df_results.astype(np.float32), sin_angles.astype(np.float32), cos_angles.astype(np.float32))

def ellipse_params_to_maps(minor_axis, major_axis, phi):
    minor_axis = minor_axis.astype(np.float32)
    major_axis = major_axis.astype(np.float32)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ecc = 1-np.sqrt(1.0 - (minor_axis / (major_axis + 1e-12))**2)
        ecc = np.nan_to_num(ecc)
    
    inten = np.sqrt(major_axis**2 + minor_axis**2)
    inten = np.nan_to_num(inten)
    
    def _robust01(x):
        m, s = np.mean(x), np.std(x)
        lo, hi = 0.0, m + 3*s
        x = np.clip(x, lo, hi)
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)
    
    ecc_n = _robust01(ecc)
    inten_n = _robust01(inten)
    
    area = np.pi * major_axis * minor_axis
    area = np.nan_to_num(area)
    
    if np.max(area) - np.min(area) > 0:
        area_n = (area - np.min(area)) / (np.max(area) - np.min(area))
    else:
        area_n = np.zeros_like(area)
        
    return ecc_n, inten_n, area_n, ecc, inten, area

def _gaussian_kernel(sigma: float):
    """
    Create a 2D isotropic Gaussian kernel normalized to unit sum.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian, in pixels.
        The kernel size is automatically set to cover ±3σ.

    Returns
    -------
    g : np.ndarray
        2D array (float32) representing the normalized Gaussian kernel.
    """
    size = int(2 * round(sigma * 3))
    if size < 3:
        size = 3
    x = np.arange(-size//2, size//2 + 1, dtype=np.float32)
    g1 = np.exp(-x**2 / (2 * sigma**2))
    g = np.outer(g1, g1).astype(np.float32)
    g /= g.sum()
    return g

def corrected_orientation(theta: np.ndarray, sigma: float = 5.0):
    """
    Smooth an orientation field (π-periodic) using a Gaussian kernel
    in the doubled-angle space to account for angular periodicity.

    Parameters
    ----------
    theta : np.ndarray
        2D array of orientation angles in radians, within [0, π).
    sigma : float, optional
        Standard deviation (in pixels) of the Gaussian kernel
        used for spatial smoothing. Default is 5.0.

    Returns
    -------
    theta_s : np.ndarray
        Smoothed orientation map (in radians), same shape as input.
    norm : np.ndarray
        Magnitude (saturation) of the local orientation coherence.
        High values indicate locally consistent directions.

    Notes
    -----
    - The smoothing is performed in the complex plane using:
          cos(2θ) and sin(2θ)
      This accounts for the π-periodicity of orientations (i.e. θ and θ+π
      represent the same direction).
    - After convolution with a Gaussian, the mean direction is recovered as:
          θ_s = 0.5 * arctan2(sin(2θ), cos(2θ))
    - The norm provides a useful map of local angular coherence
      (analogous to orientation confidence).
    """
    
    gaussian = _gaussian_kernel(float(sigma))
    
    c = np.cos(2.0 * theta)
    s = np.sin(2.0 * theta)
    
    cc = convolve(c, gaussian, mode="constant")
    ss = convolve(s, gaussian, mode="constant")
    
    norm = np.hypot(cc, ss)
    norm[norm == 0] = 1e-6
    
    theta_s = 0.5 * np.arctan2(ss, cc)
    theta_s = np.where((cc == 0) & (ss == 0), 0.0, theta_s)
    theta_s %= np.pi
    
    return theta_s.astype(np.float32), norm.astype(np.float32)


def dfr_pipeline(
    img: np.ndarray,
    ref: np.ndarray,
    angles: Optional[np.ndarray] = None,
    n_angles: int = 19,
    window_size: int = 9,
    band_thickness: int = 1,
    step: float = 1.0,
    smooth_sigma: float = 4.0,
    return_intermediates: bool = True,
):
    """
    End-to-end wrapper:
    directional stats -> DF & transmission -> ellipse fit -> orientation smoothing -> derived maps.

    Parameters
    ----------
    img : np.ndarray
        Sample image (with scattering), shape (H, W).
    ref : np.ndarray
        Reference image (no sample), shape (H, W).
    angles : np.ndarray, optional
        Angles (radians) for directional sampling. If None, uses linspace on [0, 2π) with `n_angles`.
    n_angles : int, optional
        Number of angles if `angles` is None. Default: 19.
    window_size : int, optional
        Local window size (odd). Default: 9.
    band_thickness : int, optional
        Perpendicular thickness (in pixels) of the oriented band. Default: 1.
    step : float, optional
        Sampling step along the band (≤1 for oversampling). Default: 1.0.
    smooth_sigma : float, optional
        Spatial Gaussian sigma for smoothing the orientation field (π-periodic). Default: 4.0.
    return_intermediates : bool, optional
        If True, returns intermediate arrays (means/stds, raw phi, etc.). Default: True.

    Returns
    -------
    out : dict
        {
          "angles": angles,
          "Df": Df_res,                # (H, W, A)
          "T": T,                      # (H, W, A)
          "maj": maj,                  # (H, W)
          "min": min_,                 # (H, W)
          "phi": phi,                  # (H, W) raw
          "phi_s": phi_s,              # (H, W) smoothed
          "sat": sat,                  # (H, W) orientation coherence magnitude
          "ecc_n": ecc_n,              # (H, W) normalized eccentricity
          "inten_n": inten_n,          # (H, W) normalized intensity
          "area_n": area_n,            # (H, W) normalized area
          "ecc": ecc, "inten": inten, "area": area,  # unnormalized maps
          # optional intermediates:
          "mean_s": ..., "std_s": ..., "mean_r": ..., "std_r": ...
        }
    """
    print("Starting DF Retrieval pipeline...")
    # 1) Angles
    if angles is None:
        angles = np.linspace(0, 2*np.pi, int(n_angles), endpoint=False).astype(np.float32)
    else:
        angles = np.asarray(angles, dtype=np.float32)
        n_angles = len(angles)
    print(f"Using {n_angles} angles for directional statistics.")
    # 2) Directional statistics (sample & reference)
    mean_s, std_s = compute_directional_stats(
        img, angles, window_size=window_size, band_thickness=band_thickness, step=step
    )
    print("Computed directional statistics for sample image.")
    mean_r, std_r = compute_directional_stats(
        ref, angles, window_size=window_size, band_thickness=band_thickness, step=step
    )
    print("Computed directional statistics for reference image.")
    
    # 3) DF & transmission
    Df_res, T = compute_df(std_r, std_s, mean_r, mean_s)
    print("Computed directional, non-directional dark-field and transmission maps.")
    
    # 4) Ellipse fit on polar DF
    maj, min_, phi = fit_ellipses(Df_res, np.sin(angles), np.cos(angles))
    print("Fitted ellipses to directional dark-field profiles.")
    
    # 5) Orientation smoothing (π-periodic)
    phi_s, sat = corrected_orientation(phi, sigma=smooth_sigma)
    print("Corrected orientation map and computed saturation map.")
    
    # 6) Derived maps (eccentricity, intensity, area)
    ecc_n, inten_n, area_n, ecc, inten, area = ellipse_params_to_maps(min_, maj, phi)
    print("Computed derived maps: eccentricity, intensity, area.")
    out = {
        "angles": angles,
        "Non-oriented Df": Df_res[..., 0],
        "Oriented Df": Df_res[... , 1:],
        "T": T[... , 0],
        "major axis": maj,
        "minor axis": min_,
        "Orientation": phi,
        "Corrected Orientation": phi_s,
        "saturation": sat,
        "eccentricity": ecc,
        "intensity": inten,
        "area": area,
    }

    if return_intermediates:
        out.update({
            "mean_s": mean_s, "std_s": std_s,
            "mean_r": mean_r, "std_r": std_r,
        })

    return out


from typing import Optional, Tuple, Dict, Any
import numpy as np

def retrieval_Algorithm(
    img: np.ndarray,
    ref: np.ndarray,
    angles: Optional[np.ndarray] = None,
    n_angles: int = 19,
    window_size: int = 9,
    band_thickness: int = 1,
    step: float = 1.0,
    smooth_sigma: float = 4.0,
    return_intermediates: bool = True,
    strict_finite: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Perform a complete single-exposure retrieval of the transmission and directional dark-field
    signals from a pair of images (reference and sample).

    This function implements a full statistical analysis pipeline to extract the local
    transmission, isotropic dark-field, and directional dark-field properties from a single
    sample/reference image pair. The method relies on directional local statistics, ellipse
    fitting of the polar dark-field profile, and orientation smoothing to obtain stable
    anisotropy maps.

    The algorithm proceeds in the following steps:
    1. **Directional statistical analysis**:
    Compute local means and standard deviations within a sliding window along several
    discrete orientations. This captures angular variations of image visibility.
    2. **Dark-field and transmission computation**:
    Combine the directional statistics from the reference and sample images to estimate
    both the isotropic (non-directional) and directional dark-field contrasts, as well as
    the transmission map.
    3. **Ellipse fitting in polar dark-field space**:
    Each pixel’s directional DF profile is fitted by an ellipse, from which the
    major/minor axes and principal direction are extracted.
    4. **Orientation smoothing**:
    The retrieved orientation field (π-periodic) is spatially smoothed using a Gaussian
    kernel in the doubled-angle domain (cos 2θ, sin 2θ), producing a coherent and stable
    map of principal scattering directions.
    5. **Derived anisotropy maps**:
    From the ellipse parameters, normalized maps of eccentricity, intensity (scattering
    strength), and area are computed, summarizing local anisotropy and total DF amplitude.

    Parameters
    ----------
    img : np.ndarray
        2D sample image affected by scattering (e.g. image with the sample in place).
    ref : np.ndarray
        2D reference image acquired under identical conditions without the sample.
    angles : np.ndarray, optional
        Array of orientation angles (radians) used for directional analysis. If None,
        uniformly distributed `n_angles` are generated in [0, 2π).
    n_angles : int, optional
        Number of discrete orientations if `angles` is not provided. Default: 19.
    window_size : int, optional
        Size (odd integer ≥3) of the local square window used for computing directional
        statistics. Controls the trade-off between spatial resolution and noise robustness.
        Default: 9.
    band_thickness : int, optional
        Width (in pixels) of each oriented band within the local window. Defines how many
        parallel lines are averaged per direction. Default: 1.
    step : float, optional
        Sampling step along the band (≤1 for oversampling). Smaller steps yield smoother
        angular profiles at the cost of computation time. Default: 1.0.
    smooth_sigma : float, optional
        Standard deviation (in pixels) of the Gaussian kernel used to smooth the orientation
        map (π-periodic). Controls angular coherence and stability. Default: 4.0.
    return_intermediates : bool, optional
        If True, includes intermediate quantities (directional means, stds, raw DF values)
        in the output dictionary. Default: True.
    strict_finite : bool, optional
        If True, checks that `img` and `ref` contain only finite values and raises an error
        otherwise. Default: True.
    verbose : bool, optional
        If True, prints a concise summary of parameters and pipeline progress. Default: True.

    Returns
    -------
    out : dict
        Dictionary containing all computed contrasts and derived maps:
        - "angles" : array of sampled orientations (radians)
        - "Non-oriented Df" : isotropic dark-field (H, W)
        - "Oriented Df" : directional dark-field profiles (H, W, A−1)
        - "T" : transmission map (H, W)
        - "major axis" : fitted ellipse semi-major axis (H, W)
        - "minor axis" : fitted ellipse semi-minor axis (H, W)
        - "Orientation" : raw ellipse orientation (H, W)
        - "Corrected Orientation" : smoothed orientation (H, W)
        - "saturation" : local angular coherence (H, W)
        - "eccentricity" : local anisotropy (H, W)
        - "intensity" : overall scattering strength (H, W)
        - "area" : total ellipse area (H, W)
        - (optionally) "mean_s", "std_s", "mean_r", "std_r" if `return_intermediates=True`

    Notes
    -----
    - The method is based on local directional statistics and does not require multiple
    modulator positions or reference acquisitions.
    - The retrieved orientation corresponds to the principal direction of the scattering
    blur (i.e. the direction of the SAXS anisotropy).
    - The normalized maps (`ecc_n`, `inten_n`, `area_n`) can be directly visualized or
    combined in HSV color space for intuitive representation of directional dark-field
    magnitude and orientation.
    - All computations are performed in `float32` for numerical stability and performance.

    Example
    -------
    >>> out = dfr_pipeline_checked(img, ref, n_angles=19, window_size=9)
    >>> phi = out["Corrected Orientation"]
    >>> ecc = out["eccentricity"]
    >>> inten = out["intensity"]
    """
    
    # ------------ basic type/shape checks ------------
    if not isinstance(img, np.ndarray) or not isinstance(ref, np.ndarray):
        raise TypeError("`img` and `ref` must be numpy arrays.")
    if img.ndim != 2 or ref.ndim != 2:
        raise ValueError(f"`img` and `ref` must be 2D arrays; got {img.ndim}D and {ref.ndim}D.")
    if img.shape != ref.shape:
        raise ValueError(f"`img` and `ref` must have identical shapes; got {img.shape} vs {ref.shape}.")

    # dtype sanity (we can keep user dtype, but float is safest)
    if not np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32, copy=False)
    if not np.issubdtype(ref.dtype, np.floating):
        ref = ref.astype(np.float32, copy=False)

    # finite checks
    if strict_finite:
        if not np.all(np.isfinite(img)):
            raise ValueError("`img` contains NaN/Inf; clean or disable `strict_finite`.")
        if not np.all(np.isfinite(ref)):
            raise ValueError("`ref` contains NaN/Inf; clean or disable `strict_finite`.")

    # ------------ parameter assertions ------------
    if not isinstance(window_size, int) or window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"`window_size` must be an odd integer >= 3; got {window_size}.")
    if not isinstance(band_thickness, int) or band_thickness < 1:
        raise ValueError(f"`band_thickness` must be an integer >= 1; got {band_thickness}.")
    if not (isinstance(step, (int, float)) and step > 0):
        raise ValueError(f"`step` must be a positive number; got {step}.")
    if not (isinstance(smooth_sigma, (int, float)) and smooth_sigma >= 0):
        raise ValueError(f"`smooth_sigma` must be >= 0; got {smooth_sigma}.")
    if angles is None:
        if not (isinstance(n_angles, int) and n_angles >= 3):
            raise ValueError(f"`n_angles` must be an integer >= 3 when `angles` is None; got {n_angles}.")
    else:
        angles = np.asarray(angles, dtype=np.float32)
        if angles.ndim != 1 or angles.size < 3:
            raise ValueError(f"`angles` must be 1D with at least 3 values; got shape {angles.shape}.")
        if strict_finite and not np.all(np.isfinite(angles)):
            raise ValueError("`angles` contains NaN/Inf.")
        # Accept any real values; normalize later if needed by user pipeline
        # Optional strict range check (uncomment if desired):
        # if not (np.all(angles >= -4*np.pi) and np.all(angles <= 4*np.pi)):
        #     raise ValueError("`angles` must be within a reasonable range (±4π).")

    # ------------ friendly summary ------------
    if verbose:
        msg = (
            f"[dfr_pipeline_checked] HxW={img.shape},\n "
            f"angles={'given' if angles is not None else n_angles},\n "
            f"win={window_size}, thick={band_thickness}, step={step},\n "
            f"sigma={smooth_sigma}, strict_finite={strict_finite}\n"
        )
        print(msg)

    # ------------ delegate to original pipeline ------------
    # NOTE: assumes `dfr_pipeline` is defined in the same module.
    return dfr_pipeline(
        img=img,
        ref=ref,
        angles=angles,
        n_angles=n_angles,
        window_size=window_size,
        band_thickness=band_thickness,
        step=step,
        smooth_sigma=smooth_sigma,
        return_intermediates=return_intermediates,
    )