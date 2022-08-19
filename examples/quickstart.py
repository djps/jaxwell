from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jaxwell
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

# Check to make sure double-precision is enabled.
assert jnp.zeros((1,), jnp.float64).dtype == np.float64

def monitor_fn(x, errs):
  """
  Monitor progress by plotting error and `|Ez|`.

  Args:
    x: `(xx, yy, zz)` electric field
    errs: error
  """

  # as numpy array
  x = jnp.asarray(x)

  # number of spatial dimensions
  ndims = jnp.ndim( np.squeeze(x) ) - 1

  # index to plot at in x-y plane
  ind = int( np.shape(x[0])[0] // 2 )

  data = jnp.zeros( jnp.shape(x[0])[1:3], dtype=float )
  for i in jnp.arange(0,ndims):
    data += x[i][ind,:,:].imag * x[i][ind,:,:].imag + x[i][ind,:,:].real * x[i][ind,:,:].real

  

  fig0 = plt.figure(1)
  # # first plot
  # ax0 = fig0.add_subplot(1, 2, 1)
  # ax0.set_xlabel(r'step')
  # ax0.set_ylabel(r'error relative to $|b|$')
  # ax0.semilogy(errs)
  # second plot
  ax1 = fig0.add_subplot(1, 1, 1)
  ax1.set_xlabel(r'$x$')
  ax1.set_ylabel(r'$y$')
  ax1.set_title(r'$\left\Vert E \right\Vert$')
  ax1.imshow( data )

  fig1 = plt.figure(2)
  # first plot
  ax01 = fig1.add_subplot(2, 3, 1)
  ax01.set_xlabel(r'$x$')
  ax01.set_ylabel(r'$y$')
  ax01.set_title(r'$\Re \left[ E_z \right] $')
  ax01.imshow( x[0][ind,:,:].real  )
  # second plot
  ax11 = fig1.add_subplot(2, 3, 2)
  ax11.set_xlabel(r'$x$')
  ax11.set_ylabel(r'$y$')
  ax11.set_title(r'$\Re \left[ E_y \right] $')
  ax11.imshow( x[1][ind,:,:].real  )
  # first plot
  ax02 = fig1.add_subplot(2, 3, 3)
  ax02.set_xlabel(r'$x$')
  ax02.set_ylabel(r'$y$')
  ax02.set_title(r'$\Re \left[ E_x \right] $')
  ax02.imshow( x[2][ind,:,:].real  )
  # second plot
  ax12 = fig1.add_subplot(2, 3, 4)
  ax12.set_xlabel(r'$x$')
  ax12.set_ylabel(r'$y$')
  ax12.set_title(r'$\Im \left[ E_z \right] $')
  ax12.imshow( x[0][ind,:,:].imag  )
    # first plot
  ax03 = fig1.add_subplot(2, 3, 5)
  ax03.set_xlabel(r'$x$')
  ax03.set_ylabel(r'$y$')
  ax03.set_title(r'$\Im \left[ E_y \right] $')
  ax03.imshow( x[1][ind,:,:].imag  )
  # second plot
  ax13 = fig1.add_subplot(2, 3, 6)
  ax13.set_xlabel(r'$x$')
  ax13.set_ylabel(r'$y$')
  ax13.set_title(r'$\Im \left[ E_x \right] $')
  ax13.imshow( x[2][ind,:,:].imag  )

  # visualise
  plt.show()


def point_source_sim(shape, max_iters):
  """
  Solve a point source tissue, with characteristic length scale
  of a=1mm, which is close to the spatial resolutions (dx,dy,dz)
  from a dicom image.

  The governing equation is

  \nabla \times \nabla \times E - (\omega^2 / c^2 )\epsilon E = i\omega \mu_0 J

  Using c_0 = \dfrac{1}{ \sqrt{\epsilon_0 \mu_0} }

  \nabla \times \nabla \times E - \omega^2 \mu_0 \epsilon_0 \epsilon_r E = i\omega \mu_0 J

  - The electric field is nondimensionalised by $\epsilon_0$
  - The magnetic field is nondimensionalised by $\mu_0$.

  Args:
    shape: `(xx, yy, zz)` electric field
    max_iters: maximum number of iterations

  Returns:
    x: solution
    err: error

  Examples:
    A 2D simulation would be:
    >>> point_source_sim((1,100,100), 100000)

  """

  frequency = 0.1e12 # (optical) frequency [Hz]
  omega = 2.0 * np.pi * frequency # angular frequency

  mu_0 = 4.0E-7 * np.pi # permeability in free space

  epsilon_0 = 8.854E-12 # permittivity in free space

  e_r = 4.68E+1 # relative permittivity
  
  mu_r = 1.0 # relative permeability

  sigma = 8.61E-1 # electrical conductivity [S/m]

  mu = mu_r * mu_0 # permeability

  epsilon = e_r * epsilon_0	# real part of permittivity

  T = 1.0 / frequency # time for one cycle

  print("frequency = {}".format(frequency) + "\nomega = {}".format(omega) + "\nT = {} [s]".format(T) )

  # Spatial varying quantities: wavenumber, wavelength, velocity
  # Compute via $k = omega \sqrt{ \epsilon \mu}$

  # real and imaginary parts of wavenumber, i.e. the phase constant and the attenuation constant
  alpha = omega * np.sqrt( ( mu * epsilon ) * ( np.sqrt( 1.0 + ( sigma / (omega * epsilon) )**2 ) + 1.0 ) / 2.0 )
  beta  = omega * np.sqrt( ( mu * epsilon ) * ( np.sqrt( 1.0 + ( sigma / (omega * epsilon) )**2 ) - 1.0 ) / 2.0 )

  wavelength = (2.0 * np.pi) / alpha

  print("alpha = {}".format(alpha) + "\nbeta = {}".format(beta) + "\nwavelength = {} [m]".format(wavelength) + "\nomega = {} [Hz]".format(2.0 * np.pi / wavelength) )

  v_p = omega / alpha

  # Let the grid units be a=1mm, so that the wavelength is now given in millimetres, not metres. this is required as sigma has metres.

  a = 1000.0
  dx = 1.0  # Grid spacing in _grid units_
  wavelength = a * wavelength / dx   # Wavelength in _grid units_
  omega = 2.0 * np.pi / wavelength  # Angular frequency in _grid units_

  print("\nwavelength = {} []".format(wavelength) + "\nomega = {} []".format(omega) )

  # complex permitivity _in grid units_
  epsilon = e_r - 1j * (sigma / omega)

  # effective permitivity _in grid units_
  w_eff = omega * np.sqrt(epsilon)

  # scaled source term: -i * omega * J

  start = 1 + shape[2] // 2
  stop = 4 + shape[2] // 2
  b_strength = 10000.0
  b_zero = np.zeros(shape, np.complex128)
  b_source = np.zeros(shape, np.complex128)
  b_source[shape[0] // 2, shape[1] // 2, shape[2] // 2] = b_strength
  b_source[shape[0] // 2, shape[1] // 2, start:stop] = b_strength
  b = tuple( (b_source, b_zero, b_zero))

  print(b_source[shape[0] // 2, shape[1] // 2, start:stop])

  # domain properties
  z = w_eff**2 * jnp.ones(shape, jnp.complex128)
  z = tuple( z for _ in range(3) )

  # pml in 3d
  pml_z_plus = 10
  pml_z_minus = 10
  pml_y_plus = 10
  pml_y_minus = 10
  pml_x_plus = 10
  pml_x_minus = 10
  pml_thickness = ((pml_z_minus,pml_z_plus), (pml_y_minus, pml_y_plus), (pml_x_minus,pml_x_plus) )

  # solver tolerance
  solver_eps = 1e-6
  
  # maximum iterations
  max_iters = 1000000

  # parameters
  params = jaxwell.Params(
                      pml_ths=pml_thickness,
                      pml_omega=w_eff,
                      eps=solver_eps,
                      max_iters=max_iters)

  # solve linear system
  x, err = jaxwell.solve(params, z, b)

  return x, err


shape = (50, 50, 50)
max_iterations = 10000

x, errs = point_source_sim(shape, max_iterations)

monitor_fn(x, errs)
