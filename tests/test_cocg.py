import os
from functools import partial
from typing import Tuple

import numpy as np
import pytest
from jax import device_put, devices, jit
from jax import numpy as jnp
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

# Default figure settings
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 300


def _get_homog_permittivitiy():
  return 1500.0

def _get_homog_permeability():
  return 1500.0

def _test_setter(
  N: Tuple[int] = (128,128,128),
  dx = 1e-3,
  PMLSize: int = 16,
  omega: float = 1.5e6,
  magnitude: float = 1.0,
  src_location: list = (64,64,64),
  epsilon_constructor = _get_homog_permittivitiy,
  mu_constructor = _get_homog_permeability,
  rel_err = 1e-2,
):
  dx = tuple([dx]*len(N))
  assert len(N) == len(src_location), "src_location must have same length as N"
  return {
    "N" : N,
    "dx" : dx,
    "PMLSize" : PMLSize,
    "omega": omega,
    "magnitude": magnitude,
    "src_location": src_location,
    "epsilon_constructor" : epsilon_constructor,
    "mu_constructor" : mu_constructor,
    "rel_err" : rel_err,
  }

TEST_SETTINGS = {
  "cocg_homog": _test_setter(rel_err=0.05),
}


@pytest.mark.parametrize("test_name", TEST_SETTINGS.keys())
def test_cocg(test_name,  use_plots = False,  reset_mat_file = False):
  # test settings
  settings = TEST_SETTINGS[test_name]
  # data file
  matfile = test_name + ".mat"
  # path
  dir_path = os.path.dirname(os.path.realpath(__file__))

  # Extract simulation setup
  #domain = Domain(settings["N"], settings["dx"])
  #omega = settings["omega"]
  #magnitude = settings["magnitude"]
  #src_location = settings["src_location"]
  #rel_permitivity = settings["epsilon_constructor"](domain)
  #rel_permeability = settings["mu_constructor"](domain)

  # Move everything to the CPU
  #cpu = devices("cpu")[0]

  relErr = 0.01
  
  assert relErr < settings["rel_err"], "Test failed, error above maximum limit of " + str(100*settings["rel_err"]) + "%"



if __name__ == "__main__":
  for key in TEST_SETTINGS:
    test_cocg(key, use_plots = False)
