import yt
import yt.units as yu
from yt import physical_constants as phyc
import numpy as np

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def add_fields(ds, u, x0=None):
    """Add derived fields to yt Dataset


    x0: reference position vector from which distance (in parsec) is measured
    """

    # Assumes that the density unit is muH*mH/cm**3
    def _nH(field, data):
        return data['rho'].value/yu.cm**3
    ds.add_field(("athena_pp", "nH"), function=_nH, units='cm**-3',
                 display_name=r'$n_{\rm H}$', take_log=True,
                 sampling_type='cell')

    def _pok(field, data):
        return data['press'].value*(u.pressure.value/u.k_B.value)/yu.cm**3*yu.K
    ds.add_field(("athena_pp", "pok"), function=_pok, units='cm**-3*K',
                 display_name=r'$P/k_{\rm B}$', take_log=True,
                 sampling_type='cell')

    
