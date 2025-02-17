import jax
import jax.numpy as jnp

from jimgw.transforms import *
from jimgw.single_event.transforms import *

jax.config.update("jax_enable_x64", True)

forward_keys = ('theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2','M_c', 'q', 'phase')
backward_keys = ('iota', 's1_x', 's1_y', 's1_z', 's2_x', 's2_y', 's2_z', 'M_c', 'q', 'phase')


class TestSingleEventTransform:
    def test_spin_angles_transform(self):
        # Test transformation from spin angles to cartesian spins
        # Uncomment the following code to generate the input and output files for the test

        # import numpy as np
        # from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
        # from bilby.gw.conversion import (
        #     symmetric_mass_ratio_to_mass_ratio,
        #     chirp_mass_and_mass_ratio_to_component_masses,
        # )
        # from lal import MSUN_SI

        # inputs = []
        # for _ in range(100):
        #     thetaJN = np.array(np.random.uniform(0, np.pi))
        #     phiJL = np.array(np.random.uniform(0, 2*np.pi))
        #     theta1 = np.array(np.random.uniform(0, np.pi))
        #     theta2 = np.array(np.random.uniform(0, np.pi))
        #     phi12 = np.array(np.random.uniform(0, 2*np.pi))
        #     chi1 = np.array(np.random.uniform(0, 1))
        #     chi2 = np.array(np.random.uniform(0, 1))
        #     M_c = np.array(np.random.uniform(1, 100))
        #     eta = np.array(np.random.uniform(0.1, 0.25))
        #     fRef = np.array(np.random.uniform(10, 1000))
        #     phiRef = np.array(np.random.uniform(0, 2 * np.pi))

        #     q = symmetric_mass_ratio_to_mass_ratio(eta)
        #     m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(M_c, q)

        #     inputs.append(
        #         (
        #             thetaJN,
        #             phiJL,
        #             theta1,
        #             theta2,
        #             phi12,
        #             chi1,
        #             chi2,
        #             m1,
        #             m2,
        #             fRef,
        #             phiRef,
        #         )
        #     )
        # inputs = np.array(inputs)
        # np.savez(
        #     "test/unit/source_files/spin_angles_input.npz",
        #     thetaJN=inputs[:, 0],
        #     phiJL=inputs[:, 1],
        #     theta1=inputs[:, 2],
        #     theta2=inputs[:, 3],
        #     phi12=inputs[:, 4],
        #     chi1=inputs[:, 5],
        #     chi2=inputs[:, 6],
        #     m1=inputs[:, 7],
        #     m2=inputs[:, 8],
        #     fRef=inputs[:, 9],
        #     phiRef=inputs[:, 10],
        # )

        # bilby_outputs = []
        # for input in inputs:
        #     iota, S1x, S1y, S1z, S2x, S2y, S2z = (
        #         SimInspiralTransformPrecessingNewInitialConditions(
        #             input[0],
        #             input[1],
        #             input[2],
        #             input[3],
        #             input[4],
        #             input[5],
        #             input[6],
        #             input[7] * MSUN_SI,
        #             input[8] * MSUN_SI,
        #             input[9],
        #             input[10],
        #         )
        #     )
        #     bilby_outputs.append((iota, S1x, S1y, S1z, S2x, S2y, S2z))
        # bilby_outputs = np.array(bilby_outputs)
        # np.savez(
        #     "test/unit/source_files/cartesian_spins_output_for_bilby.npz",
        #     iota=bilby_outputs[:, 0],
        #     S1x=bilby_outputs[:, 1],
        #     S1y=bilby_outputs[:, 2],
        #     S1z=bilby_outputs[:, 3],
        #     S2x=bilby_outputs[:, 4],
        #     S2y=bilby_outputs[:, 5],
        #     S2z=bilby_outputs[:, 6],
        # )

        from jimgw.single_event.utils import m1_m2_to_Mc_q
        from jimgw.single_event.transforms import SpinAnglesToCartesianSpinTransform

        # read inputs from binary
        inputs = jnp.load("test/unit/source_files/spin_angles_input.npz")
        inputs = [inputs[key] for key in inputs.keys()]
        M_c, q = m1_m2_to_Mc_q(inputs[7], inputs[8])

        # compute jimgw spins
        for i in range(100):
            row = [
                inputs[0][i],
                inputs[1][i],
                inputs[2][i],
                inputs[3][i],
                inputs[4][i],
                inputs[5][i],
                inputs[6][i],
                M_c[i],
                q[i],
                inputs[9][i],
                inputs[10][i],
            ]
            
            freq_ref = row.pop(9)
            
            # print(dict(zip(forward_keys, row)))

            jimgw_spins, jacobian = SpinAnglesToCartesianSpinTransform(
                freq_ref=freq_ref).transform(dict(zip(forward_keys, row)))
            # print(type(jimgw_spins))
            bilby_spins = jnp.load(
                "test/unit/source_files/cartesian_spins_output_for_bilby.npz"
            )
            bilby_spins = jnp.array([bilby_spins[key] for key in bilby_spins.keys()]).T
            bilby_spins = bilby_spins[i]
            
            for key in ("M_c", "q", "phase"):
                jimgw_spins.pop(key)
            jim_spins = list(jimgw_spins.values())
            # jim_spins.extend([M_c[i], q[i], freq_ref, phase])
            # print(bilby_spins)

            assert jnp.allclose(jnp.array(jim_spins), bilby_spins)
            # default atol: 1e-8, rtol: 1e-5
            assert not jnp.isnan(jacobian).any()

        # Test transformation from cartesian spins to spin angles
        # Uncomment the following code to generate the input and output files for the test

        # import numpy as np
        # from lalsimulation import SimInspiralTransformPrecessingWvf2PE
        # from bilby.gw.conversion import (
        #     symmetric_mass_ratio_to_mass_ratio,
        #     chirp_mass_and_mass_ratio_to_component_masses,
        # )

        # inputs = []
        # for _ in range(100):
        #     iota = np.array(np.random.uniform(0, np.pi))
        #     S1x = np.array(np.random.uniform(-1, 1))
        #     S1y = np.array(np.random.uniform(-1, 1))
        #     S1z = np.array(np.random.uniform(-1, 1))
        #     S2x = np.array(np.random.uniform(-1, 1))
        #     S2y = np.array(np.random.uniform(-1, 1))
        #     S2z = np.array(np.random.uniform(-1, 1))
        #     M_c = np.array(np.random.uniform(1, 100))
        #     eta = np.array(np.random.uniform(0.1, 0.25))
        #     fRef = np.array(np.random.uniform(10, 100))
        #     phiRef = np.array(np.random.uniform(0, 2 * np.pi))

        #     q = symmetric_mass_ratio_to_mass_ratio(eta)
        #     m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(M_c, q)

        #     inputs.append((iota, S1x, S1y, S1z, S2x, S2y, S2z, m1, m2, fRef, phiRef))
        # inputs = np.array(inputs)
        # np.savez(
        #     "test/unit/source_files/cartesian_spins_input.npz",
        #     iota=inputs[:, 0],
        #     S1x=inputs[:, 1],
        #     S1y=inputs[:, 2],
        #     S1z=inputs[:, 3],
        #     S2x=inputs[:, 4],
        #     S2y=inputs[:, 5],
        #     S2z=inputs[:, 6],
        #     m1=inputs[:, 7],
        #     m2=inputs[:, 8],
        #     fRef=inputs[:, 9],
        #     phiRef=inputs[:, 10],
        # )

        # bilby_outputs = []
        # for input in inputs:
        #     thteaJN, phiJL, theta1, theta2, phi12, chi1, chi2 = (
        #         SimInspiralTransformPrecessingWvf2PE(*input)
        #     )
        #     bilby_outputs.append((thteaJN, phiJL, theta1, theta2, phi12, chi1, chi2))
        # bilby_outputs = np.array(bilby_outputs)
        # np.savez(
        #     "test/unit/source_files/spin_angles_output_for_bilby.npz",
        #     thetaJN=bilby_outputs[:, 0],
        #     phiJL=bilby_outputs[:, 1],
        #     theta1=bilby_outputs[:, 2],
        #     theta2=bilby_outputs[:, 3],
        #     phi12=bilby_outputs[:, 4],
        #     chi1=bilby_outputs[:, 5],
        #     chi2=bilby_outputs[:, 6],
        # )


        # read inputs from binary
        inputs = jnp.load("test/unit/source_files/cartesian_spins_input.npz")
        inputs = [inputs[key] for key in inputs.keys()]
        M_c, q = m1_m2_to_Mc_q(inputs[7], inputs[8])

        # compute jimgw spins
        for i in range(100):
            row = [
                inputs[0][i],
                inputs[1][i],
                inputs[2][i],
                inputs[3][i],
                inputs[4][i],
                inputs[5][i],
                inputs[6][i],
                M_c[i],
                q[i],
                inputs[9][i],
                inputs[10][i],
            ]
            
            freq_ref = row.pop(9)

            
            jimgw_spins, jacobian = SpinAnglesToCartesianSpinTransform(
                freq_ref=freq_ref).inverse(dict(zip(backward_keys, row)))
            # print(type(jimgw_spins))
            

            bilby_spins = jnp.load(
                "test/unit/source_files/spin_angles_output_for_bilby.npz"
            )
            bilby_spins = jnp.array([bilby_spins[key] for key in bilby_spins.keys()]).T
            bilby_spins = bilby_spins[i]
            
            for key in ("M_c", "q", "phase"):
                jimgw_spins.pop(key)
            jimgw_spins = list(jimgw_spins.values())
            
            assert jnp.allclose(jnp.array(jimgw_spins), bilby_spins)
            # default atol: 1e-8, rtol: 1e-5
            assert not jnp.isnan(jacobian).any()

        # Test if the transformation from cartesian spins to spin angles is the inverse of the transformation from spin angles to cartesian spins

        import jax
        from jimgw.single_event.utils import eta_to_q

        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, 11)
        iota = jax.random.uniform(subkeys[0], (100,), minval=0, maxval=jnp.pi)
        S1x = jax.random.uniform(subkeys[1], (100,), minval=-1, maxval=1)
        S1y = jax.random.uniform(subkeys[2], (100,), minval=-1, maxval=1)
        S1z = jax.random.uniform(subkeys[3], (100,), minval=-1, maxval=1)
        S2x = jax.random.uniform(subkeys[4], (100,), minval=-1, maxval=1)
        S2y = jax.random.uniform(subkeys[5], (100,), minval=-1, maxval=1)
        S2z = jax.random.uniform(subkeys[6], (100,), minval=-1, maxval=1)
        M_c = jax.random.uniform(subkeys[7], (100,), minval=1, maxval=100)
        eta = jax.random.uniform(subkeys[8], (100,), minval=0.1, maxval=0.25)
        fRef = jax.random.uniform(subkeys[9], (100,), minval=10, maxval=100)
        phiRef = jax.random.uniform(subkeys[10], (100,), minval=0, maxval=2 * jnp.pi)

        q = eta_to_q(eta)

        inputs = jnp.array([iota, S1x, S1y, S1z, S2x, S2y, S2z, M_c, q, phiRef]).T
        ## Jacobian will fail (with nans) if component spins are zeroes.

        for i in range(100):
            jimgw_spins, _ = SpinAnglesToCartesianSpinTransform(freq_ref=fRef[i]).inverse(dict(zip(backward_keys, inputs[i])))
            # jimgw_spins = list(jimgw_spins.values())
            # print([float(val) for val in jimgw_spins])
            jimgw_spins, _ = SpinAnglesToCartesianSpinTransform(freq_ref=fRef[i]).transform(jimgw_spins)
            jimgw_spins = list(jimgw_spins.values())
            assert jnp.allclose(jnp.array(jimgw_spins), jnp.array([*inputs[i][7:], *inputs[i][:7]]))
            # default atol: 1e-8, rtol: 1e-5

    # def test_sky_location_transform(self):
    #     from bilby.gw.utils import zenith_azimuth_to_ra_dec as bilby_earth_to_sky
    #     from bilby.gw.detector.networks import InterferometerList

    #     from jimgw.single_event.utils import (
    #         zenith_azimuth_to_ra_dec as jimgw_earth_to_sky,
    #     )
    #     from jimgw.single_event.detector import detector_preset
    #     from astropy.time import Time

    #     ifos = ["H1", "L1"]
    #     geocent_time = 1000000000

    #     for zenith in np.linspace(0, np.pi, 10):
    #         for azimuth in np.linspace(0, 2 * np.pi, 10):
    #             bilby_sky_location = np.array(
    #                 bilby_earth_to_sky(
    #                     zenith, azimuth, geocent_time, InterferometerList(ifos)
    #                 )
    #             )
    #             jimgw_sky_location = np.array(
    #                 jimgw_earth_to_sky(
    #                     zenith,
    #                     azimuth,
    #                     Time(geocent_time, format="gps")
    #                     .sidereal_time("apparent", "greenwich")
    #                     .rad,
    #                     detector_preset[ifos[0]].vertex
    #                     - detector_preset[ifos[1]].vertex,
    #                 )
    #             )
    #             assert np.allclose(bilby_sky_location, jimgw_sky_location
