"""
used `pip install . ` in jaxwell folder so that setup.py was run.
"""

from jax.config import config
config.update("jax_enable_x64", True)

# Imports

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

import jaxwell

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import numpy as np

# Check to make sure double-precision is enabled.
assert jnp.zeros((1,), np.float64).dtype == jnp.float64


# Helper functions for building the structure and source sub-models.
def split_int(a):
  """Split integer `a` as evenly as possible."""
  return (a // 2, a // 2 + a % 2)

def pad(x, shape):
  """Pad array `x` up to `shape`."""
  return jnp.pad(x, [split_int(a - b) for a, b in zip(shape, x.shape)])

def scaleto(x, lims):
  """Scale the values of `x` from `[0, 1]` to `lims`."""
  (a, b) = lims
  return (b - a) * x + a


# Build the structure, source, and loss sub-models.
def structure(theta, thickness, shape):
  """Builds the epsilon structure.

  The structure is a slab of material located in the center of the simulation.
  `theta` is extruded along the `z`-axis by `thickness` pixels, zero-padded to
  the full simulation size, and finally scaled from `[0, 1]` to `epsilon_range`.

  For simplicity, we do not take into account the offsets between the x-, y-,
  and z-components of epsilon in the Yee cell.

  Args:
    theta: `(xx, yy)` array with elements within `[0, 1]`.
    thickness: thickness of slab in pixels.
    shape: `(xx, yy, zz)` tuple defining the shape of the simulation.
  """
  z = jnp.reshape(pad(theta, shape[:2]), shape[:2] + (1,))
  z = jnp.repeat(z, thickness, axis=2)
  z = jnp.pad(z, [(0, 0)] * 2 + [split_int(shape[2] - thickness)])
  return (z, z, z)



def source(currents, z_location, shape):
  """Inserts `currents` into the simulation at `z_location`.

  Args:
    currents: `(xx, yy)` array accepting complex-valued elements.
    z_location: location of the current source along z in pixels.
    shape: `(xx, yy, zz)` defining the shape of the simulation.
  """

  #print(shape[:2], shape[:2] + (1,))
  b = jnp.reshape(pad(currents, shape[:2]), shape[:2] + (1,))
  #print(b.shape)
  b = jnp.pad(b, [(0, 0)] * 2 + [(z_location - 1, shape[2] - z_location)])
  #print(b.shape, [(0, 0)] * 2 + [(z_location - 1, shape[2] - z_location)], [(0, 0)] * 2 )
  b_zero = np.zeros(shape, np.complex128)
  return (b, b_zero, b_zero)



def loss(x):
  """Objective loss function of the simulation field `x`.

  Implements a "focusing" objective which simply attempts to maximize the
  intensity at a point below the structure.

  Args:
    x: Tuple of 3 `(xx, yy, zz)` arrays as returned by solving with Jaxwell.
  """
  s = x[0].shape
  return -jnp.linalg.norm(x[0][s[0] // 2, s[1] // 2, 3 * s[2] // 4])


def model_fns(shape, slab_thickness):
  """
  We have `f()` and three `visualize()` functions for our model.

  In this simulation the permitivity has been scaled by $\mu_0$, the permeability has been scaled by $\epsilon_0$ and the speed of light have been factored out.

  k^2 = \mu \omega \left( \epsilon \omega + i \sigma\right)

  Let the characteristic length scale be 1 mm, which is close to the voxel spacing.

  The frequency is 915 MHz.

  Args:
    shape: `(xx, yy, zz)` defining the simulation volume.
    slab_thickness: thickness of the slab in pixels.
  """

  frequency = 915e6 # [Hz]
  omega = 2.0 * np.pi * frequency

  epsilon = 4.68E+1
  sigma = 8.61E-1 # [S/m]

  mu = 4.0E-7 * np.pi

  alpha = omega * np.sqrt( ( mu * epsilon ) * ( np.sqrt( 1.0 * ( sigma / (omega * epsilon) )**2 ) + 1.0 ) /2.0 )
  beta = omega * np.sqrt( ( mu * epsilon ) * ( np.sqrt( 1.0 * ( sigma / (omega * epsilon) )**2 ) - 1.0 ) /2.0 )

  wavelength = (2.0 * np.pi) / alpha

  print(alpha, beta, wavelength)

  dx = 40  # Grid spacing.
  wavelength = 1550.0 / dx   # Wavelength in grid units.

  omega = 2.0 * np.pi / wavelength  # Angular frequency in grid units.
  epsilon_range = (2.25, 12.25)  # Epsilon from SiO2 to Si.

  # Set the simulation parameters.
  params = jaxwell.Params(
                       pml_ths=((0, 0), (10, 10), (10, 10)),
                       pml_omega=omega,
                       eps=1e-6,
                       max_iters=1000000)


  def _model(theta, currents):
    """
    Build a basic model
    """

    # Create the full vectorial arrays for the z and b.
    # First create b based on the current
    theta = jnp.clip(theta, 0, 1)  # Clip values outside of `[0, 1]`.
    theta = structure(theta, thickness=slab_thickness, shape=shape)

    currents = currents / jnp.linalg.norm(currents)  # Normalize to norm of 1.

    b = source(currents, z_location=15, shape=shape)
    #print(np.asarray(b).shape)

    # Scale by the angular frequency as is expected for Jaxwell.
    z = tuple(omega**2 * scaleto(t, epsilon_range) for t in theta)
    b = tuple(jnp.complex128(-1j * omega * b) for b in b)

    # Simulate.
    #print(np.shape(z), np.shape(b))
    x, err = jaxwell.solve(params, z, b)

    return x, err, theta


  def f(theta, currents):
    """The function `f` to optimize over."""
    x, _, _ = _model(theta, currents)
    return loss(x)


  def vis_field(theta, currents, fn=np.imag):
    """For eyeballs."""
    x, err, full_theta = _model(theta, currents)
    return x, err, full_theta
    #plt.imshow(fn(x[0][0].T), alpha=1 - 0.2 * full_theta[0][0].T)
    #plt.title(f'Objective: {loss(x):.3f}, Error: {err:1.1e}')


  def vis_structure(theta):
    """Also for eyeballs."""
    plt.plot(theta.flatten(), '.-')
    plt.fill_between(
        range(len(theta.flatten())),
        theta.flatten(),
        0,
        color='blue',
        alpha=0.2)
    plt.title('Theta values (unclipped)')
    plt.ylim(-1, 2)


  def vis_source(currents):
    """Eyeballs, again."""
    c = currents.flatten()
    c = c / np.linalg.norm(c)
    plt.plot(np.real(c), 'b.-')
    plt.plot(np.imag(c), 'g.-')
    plt.plot(np.abs(c), 'k.-')
    plt.title('Currents (normalized)')

  return f, vis_field, vis_structure, vis_source



def optimize(f, vis, params, num_steps, **opt_args):
    """
    This
    """
    opt_init, opt_update, get_params = optimizers.sgd(**opt_args)
    opt_state = opt_init(params)

    def step(step, opt_state):
        value, grads = jax.value_and_grad(f)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    values = []
    stacked_params = []
    for i in range(num_steps):
        value, opt_state = step(i, opt_state)
        params = get_params(opt_state)
        values.append(value)
        stacked_params.append(params)
        print( i, np.shape(params), np.shape(stacked_params) )

    return params, stacked_params



def optimize_theta(init_theta, num_steps, step_size):
    currents = np.ones((1, 20))
    opt_theta, stacked_params = optimize(
        f=lambda theta: f(theta, currents),
        vis=vis_structure,
        params=init_theta,
        num_steps=num_steps,
        step_size=step_size)

    # fig0 = plt.figure()
    #ax0 = fig0.add_subplot(1, 2, 1)
    x0, err0, full_theta0 = vis_field(init_theta, currents)
    # ax0.imshow(np.abs(x0[0][0].T), alpha=1 - 0.2 * full_theta0[0][0].T)
    # ax0.set_title(f'Objective: {loss(x0):.3f}, Error: {err0:1.1e}')

    # ax1 = fig0.add_subplot(1, 2, 2)
    x1, err1, full_theta1 = vis_field(opt_theta, currents)
    # ax1.imshow(np.abs(x1[0][0].T), alpha=1 - 0.2 * full_theta1[0][0].T)
    # ax1.set_title(f'Objective: {loss(x1):.3f}, Error: {err1:1.1e}')

    return jnp.abs(x1[0][0].T), stacked_params


# Optimizer for the current source.
def optimize_currents(init_currents, num_steps, step_size):
    """
    Another
    """
    theta = jnp.zeros((1, 70))
    opt_currents, stacked_params = optimize(
        f=lambda currents: f(theta, currents),
        vis=vis_source,
        params=init_currents,
        num_steps=num_steps,
        step_size=step_size)

    # fig0 = plt.figure()
    # ax0 = fig0.add_subplot(1, 2, 1)
    x0, err0, full_theta0 = vis_field(theta, init_currents)
    # ax0.imshow(np.abs(x0[0][0].T), alpha=1 - 0.2 * full_theta0[0][0].T)
    # ax0.set_title(f'Objective: {loss(x0):.3f}, Error: {err0:1.1e}')

    # ax1 = fig0.add_subplot(1, 2, 2)
    x1, err1, full_theta1 = vis_field(theta, opt_currents)
    # ax1.imshow(np.abs(x1[0][0].T), alpha=1 - 0.2 * full_theta1[0][0].T)
    # ax1.set_title(f'Objective: {loss(x1):.3f}, Error: {err1:1.1e}')

    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(1, 1, 1)

    return jnp.abs(x1[0][0].T), stacked_params


# Optimizers for the current and structure separately and together.
def optimize_both(init, num_steps, step_size):
    """
    """
    def vis_just_structure(params):
        vis_structure(params[0])

    opt = optimize(
        f=lambda init: f(init[0], init[1]),
        vis=vis_just_structure,
        params=init,
        num_steps=num_steps,
        step_size=step_size)


# Testing the model functions.
#-----------------------------
# shape of domain
shape = (1, 100, 60)
# functions
f, vis_field, vis_structure, vis_source = model_fns(shape=shape, slab_thickness=8)
# set values
theta = jnp.ones((1, 70))
currents = -1.0 * jnp.ones((1, 20), np.complex128)
# first pass
f0 = f(theta, currents)
# output
print(f'Objective: {f0:.3f}')

# plotting
fig0 = plt.figure()
ax0 = fig0.add_subplot(1, 3, 1)
x, err, full_theta = vis_field(theta, currents)
alpha = 1.0 - 0.2 * jnp.abs( full_theta[0][0].T )
ax0.imshow( jnp.imag(x[0][0].T) )#, alpha=1 - 0.2 * full_theta[0][0].T)
ax0.set_xlabel('x', fontsize=14)
ax0.set_ylabel('y', fontsize=14)
ax0.set_title('Field', fontsize=18)
ax0.grid(True)

ax1 = fig0.add_subplot(1, 3, 2)
ax1.plot(theta.flatten(), '.-')
ax1.fill_between(
    range(len(theta.flatten())),
    theta.flatten(),
    0,
    color='blue',
    alpha=0.2)
ax1.set_title('Theta values (unclipped)')
ax1.set_ylim(-1, 2)

ax2 = fig0.add_subplot(1, 3, 3)
c = currents.flatten()
c = c / np.linalg.norm(c)
ax2.plot(jnp.real(c), 'b.-')
ax2.plot(jnp.imag(c), 'g.-')
ax2.plot(jnp.abs(c), 'k.-')
ax2.set_title('Currents (normalized)')

plt.show()

# Run the optimization.
pos = jnp.linspace(-5, 5, num=70)
init_currents = jnp.exp(-jnp.square(pos))
init_currents = jnp.reshape(init_currents, (1, 70))
_, stacked_currents_params = optimize_currents(init_currents, num_steps=10,  step_size=1e3)

# Start with `theta=0` everywhere.
init_theta = 0.0 * jnp.ones((1, 70))
_, stacked_theta_params = optimize_theta(init_theta, num_steps=12, step_size=3e1 )

nx = np.shape(stacked_currents_params)[0]
ny = np.shape(stacked_currents_params)[2]
x = np.arange(nx)
y = np.arange(ny)

X, Y = np.meshgrid(x, y)
#Z = onp.squeeze( onp.array(stacked_currents_params), axis=1)
Z = jnp.asarray(stacked_currents_params)
Z = jnp.squeeze(Z)
print(jnp.shape(Z), Z.shape, X.shape, Y.shape)

figure1 = plt.figure()
axis1 = figure1.add_subplot(111, projection='3d')
axis1.plot_wireframe(X, Y, np.transpose(Z) )
axis1.set_xlabel('x-axis')
axis1.set_ylabel('y-axis')
axis1.set_zlabel('z-axis')
axis1.set_title('Current')


nx = np.shape(stacked_theta_params)[0]
ny = np.shape(stacked_theta_params)[2]
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)
Z = np.squeeze( np.asarray(stacked_theta_params) )
print(np.shape(Z), Z.shape, X.shape, Y.shape)
figure2 = plt.figure()
axis2 = figure2.add_subplot(111, projection='3d')
axis2.plot_wireframe(X, Y, np.transpose(Z) )
axis2.set_xlabel('x-axis')
axis2.set_ylabel('y-axis')
axis2.set_zlabel('z-axis')
axis2.set_title(r'Theta')

plt.show()
