from astropy.constants import pc  # type: ignore TODO: fix astropy stubs
import astropy.units as u  # type: ignore

C_SI = 299792458.0
""" Speed of light, m/s """

MSUN = 1.988409870698050731911960804878414216e30
""" Nominal solar mass, kg """

MTSUN = 4.925490947641266978197229498498379006e-6
""" Geometrised Nominal solar mass, s """

MRSUN = 1.476625061404649406193430731479084713e3
""" Geometrised Nominal solar mass, m """

year = (1 * u.yr).cgs.value  # type: ignore
Mpc = 1e6 * pc.value  # m
euler_gamma = 0.577215664901532860606512090082

EARTH_SEMI_MAJOR_AXIS = 6378137.0  # for ellipsoid model of Earth, in m
EARTH_SEMI_MINOR_AXIS = 6356752.314  # in m

DAYSID_SI = 86164.09053133354
DAYJUL_SI = 86400.0
