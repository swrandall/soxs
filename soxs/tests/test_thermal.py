import numpy as np
import os
import shutil
import tempfile
from soxs.spectra import ApecGenerator
from soxs.spatial import PointSourceModel
from soxs.simput import write_photon_list
from soxs.instrument_registry import \
    get_instrument_from_registry
from soxs.instrument import instrument_simulator, \
    RedistributionMatrixFile, AuxiliaryResponseFile
from soxs.tests.utils import write_spectrum, get_wabs_absorb, \
    convert_rmf
from sherpa.astro.ui import load_user_model, add_user_pars, \
    load_pha, ignore, fit, set_model, set_stat, set_method, \
    covar, get_covar_results, set_covar_opt
from numpy.random import RandomState

prng = RandomState(70)

inst_name = "mucal"

rmf = RedistributionMatrixFile("xrs_%s.rmf" % inst_name)
agen = ApecGenerator(rmf.elo[0], rmf.ehi[-1], rmf.n_de, broadening=True)

def mymodel(pars, x, xhi=None):
    dx = x[1]-x[0]
    wabs = get_wabs_absorb(x+0.5*dx, pars[0])
    apec = agen.get_spectrum(pars[1], pars[2], pars[3], pars[4])
    eidxs = np.logical_and(rmf.elo >= x[0]-0.5*dx, rmf.elo-0.5*dx <= x[-1])
    return dx*wabs*apec.flux.value[eidxs]

def test_thermal():

    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    nH_sim = 0.02
    kT_sim = 6.0
    abund_sim = 0.4
    norm_sim = 1.0e-3
    redshift = 0.05

    exp_time = 5.0e4
    area = 40000.0

    spec = agen.get_spectrum(kT_sim, abund_sim, redshift, norm_sim)
    spec.apply_foreground_absorption(nH_sim)
    e = spec.generate_energies(exp_time, area, prng=prng)

    pt_src = PointSourceModel(30.0, 45.0, e.size)

    write_photon_list("thermal_model", "thermal_model", e.flux, pt_src.ra, pt_src.dec,
                      e, clobber=True)

    instrument_simulator("thermal_model_simput.fits", "thermal_model_evt.fits", exp_time, 
                         inst_name, [30.0, 45.0], astro_bkgnd=False,
                         instr_bkgnd=False, prng=prng)

    inst = get_instrument_from_registry(inst_name)
    arf = AuxiliaryResponseFile(inst["arf"])
    rmf = RedistributionMatrixFile(inst["rmf"])
    os.system("cp %s ." % arf.filename)
    convert_rmf(rmf.filename)

    write_spectrum("thermal_model_evt.fits", "thermal_model_evt.pha", clobber=True)

    load_user_model(mymodel, "wapec")
    add_user_pars("wapec", ["nH", "kT", "abund", "redshift", "norm"],
                  [0.01, 4.0, 0.2, redshift, norm_sim*0.8],
                  parmins=[0.0, 0.1, 0.0, -20.0, 0.0],
                  parmaxs=[10.0, 20.0, 10.0, 20.0, 1.0e9],
                  parfrozen=[False, False, False, True, False])

    load_pha("thermal_model_evt.pha")
    set_stat("cstat")
    set_method("simplex")
    set_model("wapec")
    ignore(":0.5, 8.0:")
    fit()
    set_covar_opt("sigma", 1.645)
    covar()
    res = get_covar_results()

    assert np.abs(res.parvals[0]-nH_sim) < res.parmaxes[0]
    assert np.abs(res.parvals[1]-kT_sim) < res.parmaxes[1]
    assert np.abs(res.parvals[2]-abund_sim) < res.parmaxes[2]
    assert np.abs(res.parvals[3]-norm_sim) < res.parmaxes[3]

    os.chdir(curdir)
    shutil.rmtree(tmpdir)

if __name__ == "__main__":
    test_thermal()
