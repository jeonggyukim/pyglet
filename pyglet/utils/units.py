import astropy.units as au
import astropy.constants as ac

# import numpy as np

class Units(object):
    """Simple class for simulation unit.

    Msun, Lsun, etc.: physical constants in code unit
    """
    def __init__(self, kind='tigris', muH=1.4):
        """
        Parameters
        ----------
           kind : str
              "tigris" for mass=1.4*muH*mH*(pc/cm)**3, 
                           length=pc
                           velocity=km/s
              with physical constants defined in tigris source code
           muH : float
              mean particle mass per H (for neutral gas).
        """
        
        if kind == 'tigris':
            # Physical constants defined in cgs units (src/units.hpp)
            self.mH = 1.6733e-24*au.g
            self.pc = 3.085678e+18*au.cm
            self.kms = 1.0e+5*au.cm/au.s
            self.kpc = 3.085678e+21*au.cm
            self.Myr = 3.155815e+13*au.s
            self.yr = 3.155815e+7*au.s
            self.c = 2.99792458e+10*au.cm/au.s
            self.k_B = 1.380658e-16*au.erg/au.K
            self.G = 6.67259e-8*au.cm**3/au.g/au.s**2
            self.M_sun = 1.9891e+33*au.g
            self.L_sun = 3.8268e+33*au.erg/au.s
            self.e = 4.80320427e-10*au.cm**1.5*au.g**0.5/au.s
            self.aR = 7.5646e-15*au.erg/au.cm**3/au.K**4
            
            self.muH = muH
            self.length = self.pc
            self.mass = ((self.mH*self.muH)*(self.length/au.cm)**3).to('Msun')
            self.density = (self.mH*self.muH)*au.cm**-3
            self.velocity = (self.kms).to('km s-1')
            self.time = (self.length/self.velocity).to('Myr')
            self.momentum = self.mass*self.velocity
            self.pressure = (self.density*self.velocity**2).to('erg cm-3')
            self.energy = self.mass*self.velocity**2

            # For yt
            self.units_override = dict(length_unit=(1.0, 'pc'),
                                       time_unit=(1.0, 'pc/km*s'),
                                       mass_unit=(muH*self.mH.value*self.length.value**3,
                                                  'g'))

        #     self.density = (self.mass/self.length**3).cgs
        # self.momentum = (self.mass*self.velocity).to('Msun km s-1')
        # self.energy = (self.mass*self.velocity**2).cgs
        # self.pressure = (self.density*self.velocity**2).cgs
        # self.energy_density = self.pressure.to('erg/cm**3')
        
        # self.mass_flux = (self.density*self.velocity).to('Msun pc-2 Myr-1')
        # self.momentum_flux = (self.density*self.velocity**2).to('Msun km s-1 pc-2 Myr-1')
        
        # Define (physical constants in code units)^-1
        #
        # Opposite to the convention chosen by set_units function in
        # athena/src/units.c This is because in post-processing we want to
        # convert from code units to more convenient ones by "multiplying" these
        # constants

        # self.pc = self.length.to('pc').value
        # self.kpc = self.length.to('kpc').value
        # self.Myr = self.time.to('Myr').value
        # self.kms = self.velocity.to('km/s').value
        # self.Msun = self.mass.to('Msun').value
        # self.Lsun = (self.energy/self.time).to('Lsun').value
        # self.erg = self.energy.to('erg').value
        # self.eV = self.energy.to('eV').value
        # self.s = self.time.to('s').value
        # self.pok = ((self.pressure/ac.k_B).to('cm**-3*K')).value
        # self.muG = np.sqrt(4*np.pi*self.energy_density.cgs.value)/1e-6
