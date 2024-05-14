import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy
fi = h5.File("gamma_data.h5")
xa = fi["xa"]
xb = fi["xb"]
ga = fi["ga"]
gb = fi["gb"]
ga = np.array(ga)
gb = np.array(gb)
def orthMatAdjustSign(uv:np.ndarray):
    """ Make sure that in each column of the matrix `uv` the entry with largest absolute value has positive sign. """
    for i in range(uv.shape[1]):
        uv[:, i] *= np.sign(uv[np.argmax(np.abs(uv[:, i])), i])
    return uv
orthMatAdjustSign(ga)
orthMatAdjustSign(gb)

ket = lambda a,b: np.kron(ga[:,a],gb[:,b]).reshape(len(xa),len(xb))

def plot(z):
    interpolate = True
    if interpolate: 
        N = 100
        interp = scipy.interpolate.RegularGridInterpolator((xa,xb), z, method="cubic")
        x = np.linspace(np.min(xa),np.max(xa),N)
        y = np.linspace(np.min(xb),np.max(xb),N)
        x,y = np.meshgrid(x,y)
        z = interp((x,y))
    else:
        x,y = np.meshgrid(xa,xb)
        z = z.T 

    cmap = cm.bwr
    nContours = 20
    ax = plt.gca()
    vMax = z.max()
    vMin = z.min()
    vRange = abs(vMax) + abs(vMin)
    l1 = int(nContours * abs(vMin) / vRange)
    l2 = int(nContours * abs(vMax) / vRange) + 1
    levels = np.hstack((np.linspace(vMin, 0, l1, endpoint=False), np.linspace(0, vMax, l2)[1:]))
    if vMax > 0:
        if vMin > 0:
            # or norm = DivergingNorm(vmin=z.min(), vcenter=0, vmax=z.max())
            norm = matplotlib.colors.TwoSlopeNorm(0.0, -vMax, vMax)
        else:
            norm = matplotlib.colors.TwoSlopeNorm(0.0, vMin, vMax)
    else:
        norm = matplotlib.colors.TwoSlopeNorm(0.0, vMin, -vMin) 
    cmap = plt.get_cmap(cmap, len(levels) - 1)

    cset1 = ax.contourf(x, y, z, levels, norm=norm, cmap=cmap)
    cset2 = ax.contour(x, y, z, cset1.levels, colors='k', alpha=0.8)
    for c in cset2.collections:
        c.set_linestyle('solid')
    if np.any(z > 0) and np.any(z < 0):
        cset3 = ax.contour(x, y, z, (0,), colors='darkred', linewidths=2, linestyles="dashed")
    out = cset1  # for colorbar
    plt.xlabel(r"$\gamma_a$")
    plt.ylabel(r"$\gamma_b$")
    plt.gcf().colorbar(out)
    plt.show()

    
def calculate_cart_z(z, xa, xb):
    """
    Calculate cartesian wavefunction for plotting side-by-side
    ARGS:
    z - wavefunction
    xa - value range for x-axis
    xb - value range for y-axis
    OUTS:
    x,y - meshgrid of x and y values for contour plot
    z - interpolated wavefunction using new x and y values
    """
    N = 100
    interp = scipy.interpolate.RegularGridInterpolator((xa,xb), z, method="cubic")
    x = np.linspace(np.min(xa),np.max(xa),N)
    y = np.linspace(np.min(xb),np.max(xb),N)
    x,y = np.meshgrid(x,y)
    z = interp((x,y))
    return x, y, z

#center the wavefunction to be at the origin
def center_wavefunction_axis(x):
    """
    center middle of wavefunction cut at (0,0)
    ARGS:
    x - value range for x or y -axis
    OUTS:
    x_pol - value range for polar x or y -axis
    """
    mid_x = np.mean([np.min(x), np.max(x)])
    x_pol = x - mid_x
    return x_pol
xa_pol = center_wavefunction_axis(xa)
xb_pol = center_wavefunction_axis(xb)

#scale xa and xb to between -1 and 1
def linear_rescale(a, new_min, new_max):
    """
    rescale array linearly to be in the range [new_min, new_max]
    ARGS:
    a - array to be scaled
    new_min - min of the desired range
    new_max - max of the desired range
    OUTS:
    ap - scaled array
    """
    old_max = np.max(a)
    old_min = np.min(a)
    m = (new_max-new_min)/(old_max-old_min)
    b = new_min - m * old_min
    ap = m * a + b
    return ap
xa_pol = linear_rescale(xa_pol, -1, 1)
xb_pol = linear_rescale(xb_pol, -1, 1)

def calculate_polar_z(z, xa, xb):
    """
    Calculate polar wavefunction for plotting side-by-side
    ARGS:
    z - wavefunction
    xa - value range for x-axis
    xb - value range for y-axis
    OUTS:
    R,Phi - meshgrid of R and Phi values for contour plot
    z - interpolated wavefunction using new x and y values, which are now R and Phi
    """
    N = 100
    #change fill_value to 0 to accommodate extrapolations
    #change bounds_error to False to force to work
    interp = scipy.interpolate.RegularGridInterpolator((xa,xb), z, method="cubic", fill_value=0, bounds_error=False)
    r = np.sqrt(xa**2+xb**2)
    r_min = np.min(r)
    r_max = np.max(r)
    r = np.linspace(r_min,r_max,N)
    #calculating the range on phi has rounding issues at 0, instead set range on phi to be [0,pi]
    phi = np.linspace(0,np.pi,N)
    #define new R and Phi for interpolating z
    R, Phi = np.meshgrid(r,phi)
    #convert to x-y values
    x = R*np.cos(Phi)
    y = R*np.sin(Phi)
    z = interp((x,y))
    return R, Phi, z

def plot_contour(fig, ax, x, y, z):
    """
    plot a contour plot into the given figure and axis
    ARGS:
    fig - figure returned by plt.subplots
    ax - a single axis object returned by plt.subplots
    x - x-axis values
    y - y-axis values
    z - interpolated wavefunction that matches with x and y
    OUTS:
    fig - the modified figure
    ax - the modified axis
    """
    cmap = cm.bwr
    nContours = 20
    vMax = z.max()
    vMin = z.min()
    vRange = abs(vMax) + abs(vMin)
    l1 = int(nContours * abs(vMin) / vRange)
    l2 = int(nContours * abs(vMax) / vRange) + 1
    levels = np.hstack((np.linspace(vMin, 0, l1, endpoint=False), np.linspace(0, vMax, l2)[1:]))
    if vMax > 0:
        if vMin > 0:
            # or norm = DivergingNorm(vmin=z.min(), vcenter=0, vmax=z.max())
            norm = matplotlib.colors.TwoSlopeNorm(0.0, -vMax, vMax)
        else:
            norm = matplotlib.colors.TwoSlopeNorm(0.0, vMin, vMax)
    else:
        norm = matplotlib.colors.TwoSlopeNorm(0.0, vMin, -vMin) 
    cmap = plt.get_cmap(cmap, len(levels) - 1)
    cset1 = ax.contourf(x, y, z, levels, norm=norm, cmap=cmap)
    cset2 = ax.contour(x, y, z, cset1.levels, colors='k', alpha=0.8)
    #set those edges to be solid
    for c in cset2.collections:
        c.set_linestyle('solid')
    if np.any(z > 0) and np.any(z < 0):
        cset3 = ax.contour(x, y, z, (0,), colors='darkred', linewidths=2, linestyles="dashed")
    out = cset1  # for colorbar
    fig.colorbar(out)#, cax=cax, orientation='vertical')
    return fig, ax

def plot_cart_and_polar(x,y,z1,R,Phi,z2):
    """
    plot the Cartesian and polar plots side-by-side
    ARGS:
    x - x-axis values
    y - y-axis values
    z1 - interpolated wavefunction that matches with x and y
    R - R-axis values
    Phi - Phi-axis values
    z2 - interpolated wavefunction that matches with R and Phi
    OUTS:
    matplotlib figure
    """
    #CARTESIAN
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(13,5))
    plot_contour(fig, ax1, x, y, z1)
    ax1.set_xlabel(r"$\gamma_a$")
    ax1.set_ylabel(r"$\gamma_b$")
    
    #POLAR
    plot_contour(fig, ax2, R, Phi, z2)
    ax2.set_xlabel(r"r", fontsize=24)
    ax2.set_ylabel(r"$\phi$", fontsize=24)
    ax2.set_xlim(0,1)
    plt.subplots_adjust(wspace=0.2)
    plt.show()
    
def calculate_and_plot_cart_and_polar(z):
    """
    interpolate z in Cartesian and polar and plot side-by-side
    ARGS:
    z - wavefunction from file
    GLOBAL:
    xa - gamma a values from file
    xb - gamma b values from file
    xa_pol - modified gamma a values shifted to origin and linearly scaled to be between -1 and 1
    xb_pol - modified gamma b values shifted to origin and linearly scaled to be between -1 and 1
    OUTS:
    matplotlib figure
    """
    x, y, z1 = calculate_cart_z(z, xa, xb)
    R, Phi, z2 = calculate_polar_z(z, xa_pol, xb_pol)
    plot_cart_and_polar(x, y, z1, R, Phi, z2)


plot(ket(2,0)+ket(0,2))
calculate_and_plot_cart_and_polar(ket(2,0)+ket(0,2))
