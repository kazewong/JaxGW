import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, Float

from jimgw.constants import MTSUN


def inner_product(
    h1: Float[Array, " n_sample"],
    h2: Float[Array, " n_sample"],
    frequency: Float[Array, " n_sample"],
    psd: Float[Array, " n_sample"],
) -> Float:
    """
        Evaluating the inner product of two waveforms h1 and h2 with the psd.

    Do psd interpolation outside the inner product loop to speed up the evaluation

        Parameters
        ----------
        h1 : Float[Array, "n_sample"]
                First waveform. Can be complex.
        h2 : Float[Array, "n_sample"]
                Second waveform. Can be complex.
        frequency : Float[Array, "n_sample"]
                Frequency array.
        psd : Float[Array, "n_sample"]
                Power spectral density.

        Returns
        -------
        Float
                Inner product of h1 and h2 with the psd.
    """
    # psd_interp = jnp.interp(frequency, psd_frequency, psd)
    df = frequency[1] - frequency[0]
    integrand = jnp.conj(h1) * h2 / psd
    return 4.0 * jnp.real(trapezoid(integrand, dx=df))


def m1_m2_to_M_q(m1: Float, m2: Float):
    """
    Transforming the primary mass m1 and secondary mass m2 to the Total mass M
    and mass ratio q.

    Parameters
        ----------
        m1 : Float
                Primary mass.
        m2 : Float
                Secondary mass.

        Returns
        -------
        M_tot : Float
                Total mass.
        q : Float
                Mass ratio.
    """
    M_tot = m1 + m2
    q = m2 / m1
    return M_tot, q


def M_q_to_m1_m2(M_tot: Float, q: Float):
    """
    Transforming the Total mass M and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Parameters
    ----------
    M_tot : Float
            Total mass.
    q : Float
            Mass ratio.

    Returns
    -------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.
    """
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


def m1_m2_to_Mc_q(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforming the primary mass m1 and secondary mass m2 to the chirp mass M_c
    and mass ratio q.

    Parameters
    ----------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.

    Returns
    -------
    M_c : Float
            Chirp mass.
    q : Float
            Mass ratio.
    """
    M_tot = m1 + m2
    eta = m1 * m2 / M_tot**2
    M_c = M_tot * eta ** (3.0 / 5)
    q = m2 / m1
    return M_c, q


def Mc_q_to_m1_m2(M_c: Float, q: Float) -> tuple[Float, Float]:
    """
    Transforming the chirp mass M_c and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Parameters
    ----------
    M_c : Float
            Chirp mass.
    q : Float
            Mass ratio.

    Returns
    -------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.
    """
    eta = q / (1 + q) ** 2
    M_tot = M_c / eta ** (3.0 / 5)
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


def m1_m2_to_M_eta(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforming the primary mass m1 and secondary mass m2 to the total mass M
    and symmetric mass ratio eta.

    Parameters
    ----------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.

    Returns
    -------
    M : Float
            Total mass.
    eta : Float
            Symmetric mass ratio.
    """
    M_tot = m1 + m2
    eta = m1 * m2 / M_tot**2
    return M_tot, eta


def M_eta_to_m1_m2(M_tot: Float, eta: Float) -> tuple[Float, Float]:
    """
    Transforming the total mass M and symmetric mass ratio eta to the primary mass m1
    and secondary mass m2.

    Parameters
    ----------
    M : Float
            Total mass.
    eta : Float
            Symmetric mass ratio.

    Returns
    -------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.
    """
    m1 = M_tot * (1 + jnp.sqrt(1 - 4 * eta)) / 2
    m2 = M_tot * (1 - jnp.sqrt(1 - 4 * eta)) / 2
    return m1, m2


def m1_m2_to_Mc_eta(m1: Float, m2: Float) -> tuple[Float, Float]:
    """
    Transforming the primary mass m1 and secondary mass m2 to the chirp mass M_c
    and symmetric mass ratio eta.

    Parameters
    ----------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.

    Returns
    -------
    M_c : Float
            Chirp mass.
    eta : Float
            Symmetric mass ratio.
    """
    M = m1 + m2
    eta = m1 * m2 / M**2
    M_c = M * eta ** (3.0 / 5)
    return M_c, eta


def Mc_eta_to_m1_m2(M_c: Float, eta: Float) -> tuple[Float, Float]:
    """
    Transforming the chirp mass M_c and symmetric mass ratio eta to the primary mass m1
    and secondary mass m2.

    Parameters
    ----------
    M_c : Float
            Chirp mass.
    eta : Float
            Symmetric mass ratio.

    Returns
    -------
    m1 : Float
            Primary mass.
    m2 : Float
            Secondary mass.
    """
    M = M_c / eta ** (3.0 / 5)
    m1 = M * (1 + jnp.sqrt(1 - 4 * eta)) / 2
    m2 = M * (1 - jnp.sqrt(1 - 4 * eta)) / 2
    return m1, m2


def q_to_eta(q: Float) -> Float:
    """
    Transforming the chirp mass M_c and mass ratio q to the symmetric mass ratio eta.

    Parameters
    ----------
    M_c : Float
            Chirp mass.
    q : Float
            Mass ratio.

    Returns
    -------
    eta : Float
            Symmetric mass ratio.
    """
    eta = q / (1 + q) ** 2
    return eta


def eta_to_q(eta: Float) -> Float:
    """
    Transforming the symmetric mass ratio eta to the mass ratio q.

    Copied and modified from bilby/gw/conversion.py

    Parameters
    ----------
    eta : Float
            Symmetric mass ratio.

    Returns
    -------
    q : Float
            Mass ratio.
    """
    temp = 1 / eta / 2 - 1
    return temp - (temp**2 - 1) ** 0.5


def euler_rotation(delta_x: Float[Array, " 3"]):
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angles, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.

    Copied and modified from bilby-cython/geometry.pyx
    """
    norm = jnp.linalg.vector_norm(delta_x)

    cos_beta = delta_x[2] / norm
    sin_beta = jnp.sqrt(1 - cos_beta**2)

    alpha = jnp.arctan2(-delta_x[1] * cos_beta, delta_x[0])
    gamma = jnp.arctan2(delta_x[1], delta_x[0])

    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    cos_gamma = jnp.cos(gamma)
    sin_gamma = jnp.sin(gamma)

    rotation = jnp.array(
        [
            [
                cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma,
                -sin_alpha * cos_beta * cos_gamma - cos_alpha * sin_gamma,
                sin_beta * cos_gamma,
            ],
            [
                cos_alpha * cos_beta * sin_gamma + sin_alpha * cos_gamma,
                -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma,
                sin_beta * sin_gamma,
            ],
            [-cos_alpha * sin_beta, sin_alpha * sin_beta, cos_beta],
        ]
    )

    return rotation


def angle_rotation(
    zenith: Float, azimuth: Float, rotation: Float[Array, " 3 3"]
) -> tuple[Float, Float]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame to the polar angle and azimuthal angle in sky frame.

    Copied and modified from bilby-cython/geometry.pyx

    Parameters
    ----------
    zenith : Float
            Zenith angle.
    azimuth : Float
            Azimuthal angle.
    rotation : Float[Array, " 3 3"]
            The rotation matrix.

    Returns
    -------
    theta : Float
            Polar angle.
    phi : Float
            Azimuthal angle.
    """
    sin_azimuth = jnp.sin(azimuth)
    cos_azimuth = jnp.cos(azimuth)
    sin_zenith = jnp.sin(zenith)
    cos_zenith = jnp.cos(zenith)

    theta = jnp.acos(
        rotation[2][0] * sin_zenith * cos_azimuth
        + rotation[2][1] * sin_zenith * sin_azimuth
        + rotation[2][2] * cos_zenith
    )
    phi = jnp.fmod(
        jnp.arctan2(
            rotation[1][0] * sin_zenith * cos_azimuth
            + rotation[1][1] * sin_zenith * sin_azimuth
            + rotation[1][2] * cos_zenith,
            rotation[0][0] * sin_zenith * cos_azimuth
            + rotation[0][1] * sin_zenith * sin_azimuth
            + rotation[0][2] * cos_zenith,
        )
        + 2 * jnp.pi,
        2 * jnp.pi,
    )
    return theta, phi


def theta_phi_to_ra_dec(theta: Float, phi: Float, gmst: Float) -> tuple[Float, Float]:
    """
    Transforming the polar angle and azimuthal angle to right ascension and declination.

    Parameters
    ----------
    theta : Float
            Polar angle.
    phi : Float
            Azimuthal angle.
    gmst : Float
            Greenwich mean sidereal time.

    Returns
    -------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    """
    ra = phi + gmst
    dec = jnp.pi / 2 - theta
    ra = ra % (2 * jnp.pi)
    return ra, dec


def rotate_y(angle, vec):
    """
    Rotate the vector (x, y, z) about y-axis
    """
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    rotation_matrix = jnp.array(
        [[cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]]
    )
    rotated_vec = jnp.dot(rotation_matrix, vec)
    return rotated_vec


def rotate_z(angle, vec):
    """
    Rotate the vector (x, y, z) about z-axis
    """
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    rotation_matrix = jnp.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
    )
    rotated_vec = jnp.dot(rotation_matrix, vec)
    return rotated_vec


def Lmag_2PN(m1, m2, v0):
    """
    Compute the magnitude of the orbital angular momentum
    to 2 post-Newtonian orders.
    """
    eta = m1 * m2 / (m1 + m2) ** 2
    ## Simplified from:
    ## (m1 + m2) * (m1 + m2) * eta = m1 * m2
    LN = (m1 + m2) * (m1 + m2) * eta / v0
    L_2PN = 1.5 + eta / 6.0
    return LN * (1.0 + v0 * v0 * L_2PN)


def spin_angles_to_cartesian_spin(
    theta_jn: Float,
    phi_jl: Float,
    tilt_1: Float,
    tilt_2: Float,
    phi_12: Float,
    chi_1: Float,
    chi_2: Float,
    M_c: Float,
    q: Float,
    fRef: Float,
    phiRef: Float,
) -> tuple[Float, Float, Float, Float, Float, Float, Float]:
    """
    Transforming the spin parameters

    The code is based on the approach used in LALsimulation:
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html

    Parameters:
    -------
    theta_jn: Float
        Zenith angle between the total angular momentum and the line of sight
    phi_jl: Float
        Difference between total and orbital angular momentum azimuthal angles
    tilt_1: Float
        Zenith angle between the spin and orbital angular momenta for the primary object
    tilt_2: Float
        Zenith angle between the spin and orbital angular momenta for the secondary object
    phi_12: Float
        Difference between the azimuthal angles of the individual spin vector projections
        onto the orbital plane
    chi_1: Float
        Primary object aligned spin:
    chi_2: Float
        Secondary object aligned spin:
    M_c: Float
        The chirp mass
    q: Float
        The mass ratio
    fRef: Float
        The reference frequency
    phiRef: Float
        Binary phase at a reference frequency

    Returns:
    -------
    iota: Float
        Zenith angle between the orbital angular momentum and the line of sight
    S1x: Float
        The x-component of the primary spin
    S1y: Float
        The y-component of the primary spin
    S1z: Float
        The z-component of the primary spin
    S2x: Float
        The x-component of the secondary spin
    S2y: Float
        The y-component of the secondary spin
    S2z: Float
        The z-component of the secondary spin
    """

    # Starting frame: LNh along the z-axis
    # S1hat on the x-z plane
    LNh = jnp.array([0.0, 0.0, 1.0])

    # Define the spin vectors in the LNh frame
    s1hat = jnp.array(
        [
            jnp.sin(tilt_1) * jnp.cos(phiRef),
            jnp.sin(tilt_1) * jnp.sin(phiRef),
            jnp.cos(tilt_1),
        ]
    )
    s2hat = jnp.array(
        [
            jnp.sin(tilt_2) * jnp.cos(phi_12 + phiRef),
            jnp.sin(tilt_2) * jnp.sin(phi_12 + phiRef),
            jnp.cos(tilt_2),
        ]
    )

    m1, m2 = Mc_q_to_m1_m2(M_c, q)
    v0 = jnp.cbrt((m1 + m2) * MTSUN * jnp.pi * fRef)

    # Define S1, S2, and J
    Lmag = Lmag_2PN(m1, m2, v0)
    s1 = m1 * m1 * chi_1 * s1hat
    s2 = m2 * m2 * chi_2 * s2hat
    J = s1 + s2 + jnp.array([0.0, 0.0, Lmag])

    # Normalize J, and find theta0 and phi0 (the angles in starting frame)
    Jhat = J / jnp.linalg.norm(J)
    theta0 = jnp.arccos(Jhat[2])
    phi0 = jnp.arctan2(Jhat[1], Jhat[0])

    # Rotation 1: Rotate about z-axis by -phi0
    s1hat = rotate_z(-phi0, s1hat)
    s2hat = rotate_z(-phi0, s2hat)

    # Rotation 2: Rotate about y-axis by -theta0
    LNh = rotate_y(-theta0, LNh)
    s1hat = rotate_y(-theta0, s1hat)
    s2hat = rotate_y(-theta0, s2hat)

    # Rotation 3: Rotate about z-axis by -phi_jl
    LNh = rotate_z(phi_jl - jnp.pi, LNh)
    s1hat = rotate_z(phi_jl - jnp.pi, s1hat)
    s2hat = rotate_z(phi_jl - jnp.pi, s2hat)

    # Compute iota
    N = jnp.array([0.0, jnp.sin(theta_jn), jnp.cos(theta_jn)])
    iota = jnp.arccos(jnp.dot(N, LNh))

    thetaLJ = jnp.arccos(LNh[2])
    phiL = jnp.arctan2(LNh[1], LNh[0])

    # Rotation 4: Rotate about z-axis by -phiL
    s1hat = rotate_z(-phiL, s1hat)
    s2hat = rotate_z(-phiL, s2hat)
    N = rotate_z(-phiL, N)

    # Rotation 5: Rotate about y-axis by -thetaLJ
    s1hat = rotate_y(-thetaLJ, s1hat)
    s2hat = rotate_y(-thetaLJ, s2hat)
    N = rotate_y(-thetaLJ, N)

    # Rotation 6:
    phiN = jnp.arctan2(N[1], N[0])
    s1hat = rotate_z(jnp.pi / 2.0 - phiN - phiRef, s1hat)
    s2hat = rotate_z(jnp.pi / 2.0 - phiN - phiRef, s2hat)

    S1 = s1hat * chi_1
    S2 = s2hat * chi_2
    return iota, S1[0], S1[1], S1[2], S2[0], S2[1], S2[2]


def cartesian_spin_to_spin_angles(
    iota: Float,
    S1x: Float,
    S1y: Float,
    S1z: Float,
    S2x: Float,
    S2y: Float,
    S2z: Float,
    M_c: Float,
    q: Float,
    fRef: Float,
    phiRef: Float,
) -> tuple[Float, Float, Float, Float, Float, Float, Float]:
    """
    Transforming the cartesian spin parameters to the spin angles

    The code is based on the approach used in LALsimulation:
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html

    Parameters:
    -------
    iota: Float
       Zenith angle between the orbital angular momentum and the line of sight
    S1x: Float
        The x-component of the primary spin
    S1y: Float
        The y-component of the primary spin
    S1z: Float
        The z-component of the primary spin
    S2x: Float
        The x-component of the secondary spin
    S2y: Float
        The y-component of the secondary spin
    S2z: Float
        The z-component of the secondary spin
    M_c: Float
        The chirp mass
    q: Float
        The mass ratio
    fRef: Float
        The reference frequency
    phiRef: Float
        The binary phase at the reference frequency

    Returns:
    -------
    theta_jn: Float
    Zenith angle between the total angular momentum and the line of sight
    phi_jl: Float
        Difference between total and orbital angular momentum azimuthal angles
    tilt_1: Float
        Zenith angle between the spin and orbital angular momenta for the primary object
    tilt_2: Float
        Zenith angle between the spin and orbital angular momenta for the secondary object
    phi_12: Float
        Difference between the azimuthal angles of the individual spin vector projections
        onto the orbital plane
    chi_1: Float
        Primary object aligned spin:
    chi_2: Float
        Secondary object aligned spin:
    """
    # Starting frame: LNh along the z-axis
    LNh = jnp.array([0.0, 0.0, 1.0])

    # Define the dimensionless component spin vectors and magnitudes
    s1_vec = jnp.array([S1x, S1y, S1z])
    s2_vec = jnp.array([S2x, S2y, S2z])
    chi_1 = jnp.linalg.norm(s1_vec)
    chi_2 = jnp.linalg.norm(s2_vec)

    # Define the spin unit vectors in the LNh frame
    if chi_1 > 0:
        s1hat = s1_vec / chi_1
    else:
        s1hat = jnp.array([0.0, 0.0, 0.0])
    if chi_2 > 0:
        s2hat = s2_vec / chi_2
    else:
        s2hat = jnp.array([0.0, 0.0, 0.0])

    # Azimuthal and polar angles of the spin vectors
    phi1 = jnp.arctan2(s1hat[1], s1hat[0])
    phi2 = jnp.arctan2(s2hat[1], s2hat[0])

    phi_12 = phi2 - phi1

    if phi_12 < 0:
        phi_12 += 2 * jnp.pi

    tilt_1 = jnp.arccos(s1hat[2])
    tilt_2 = jnp.arccos(s2hat[2])

    # Get angles in the J-N frame
    m1, m2 = Mc_q_to_m1_m2(M_c, q)
    v0 = jnp.cbrt((m1 + m2) * MTSUN * jnp.pi * fRef)

    # Define S1, S2, J
    S1 = m1 * m1 * s1_vec
    S2 = m2 * m2 * s2_vec

    Lmag = Lmag_2PN(m1, m2, v0)
    J = S1 + S2 + Lmag * LNh

    # Normalize J
    Jhat = J / jnp.linalg.norm(J)

    thetaJL = jnp.arccos(Jhat[2])
    phiJ = jnp.arctan2(Jhat[1], Jhat[0])

    # Azimuthal angle from phase angle
    phi0 = 0.5 * jnp.pi - phiRef
    # Line-of-sight vector in L-frame
    N = jnp.array(
        [jnp.sin(iota) * jnp.cos(phi0), jnp.sin(iota) * jnp.sin(phi0), jnp.cos(iota)]
    )

    # Inclination w.r.t. J
    theta_jn = jnp.arccos(jnp.dot(Jhat, N))

    # Rotate from L-frame to J-frame
    N = rotate_z(-phiJ, N)
    N = rotate_y(-thetaJL, N)

    LNh = rotate_z(-phiJ, LNh)
    LNh = rotate_y(-thetaJL, LNh)

    phiN = jnp.arctan2(N[1], N[0])
    LNh = rotate_z(0.5 * jnp.pi - phiN, LNh)

    phi_jl = jnp.arctan2(LNh[1], LNh[0])
    if phi_jl < 0:
        phi_jl += 2 * jnp.pi

    return theta_jn, phi_jl, tilt_1, tilt_2, phi_12, chi_1, chi_2


def zenith_azimuth_to_ra_dec(
    zenith: Float, azimuth: Float, gmst: Float, rotation: Float[Array, " 3 3"]
) -> tuple[Float, Float]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame to right ascension and declination.

    Parameters
    ----------
    zenith : Float
            Zenith angle.
    azimuth : Float
            Azimuthal angle.
    gmst : Float
            Greenwich mean sidereal time.
    rotation : Float[Array, " 3 3"]
            The rotation matrix.

    Copied and modified from bilby/gw/utils.py

    Returns
    -------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    """
    theta, phi = angle_rotation(zenith, azimuth, rotation)
    ra, dec = theta_phi_to_ra_dec(theta, phi, gmst)
    return ra, dec


def ra_dec_to_theta_phi(ra: Float, dec: Float, gmst: Float) -> tuple[Float, Float]:
    """
    Transforming the right ascension ra and declination dec to the polar angle
    theta and azimuthal angle phi.

    Parameters
    ----------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    gmst : Float
            Greenwich mean sidereal time.

    Returns
    -------
    theta : Float
            Polar angle.
    phi : Float
            Azimuthal angle.
    """
    phi = ra - gmst
    theta = jnp.pi / 2 - dec
    phi = (phi + 2 * jnp.pi) % (2 * jnp.pi)
    return theta, phi


def ra_dec_to_zenith_azimuth(
    ra: Float, dec: Float, gmst: Float, rotation: Float[Array, " 3 3"]
) -> tuple[Float, Float]:
    """
    Transforming the right ascension and declination to the zenith angle and azimuthal angle.

    Parameters
    ----------
    ra : Float
            Right ascension.
    dec : Float
            Declination.
    gmst : Float
            Greenwich mean sidereal time.
    rotation : Float[Array, " 3 3"]
            The rotation matrix.

    Returns
    -------
    zenith : Float
            Zenith angle.
    azimuth : Float
            Azimuthal angle.
    """
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    zenith, azimuth = angle_rotation(theta, phi, rotation)
    return zenith, azimuth
