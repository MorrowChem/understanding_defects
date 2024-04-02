from collections.abc import Iterable
import numpy as np
from scipy.stats import gaussian_kde
from quippy.descriptors import Descriptor
from ase.optimize import BFGSLineSearch
from ase.constraints import ExpCellFilter
from ase.atoms import Atoms
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, \
    CoordinationAnalysisModifier, ComputePropertyModifier


def despine(axs):
    axs.xaxis.set_visible(False)
    axs.yaxis.set_visible(False)
    for i in axs.spines.values():
        i.set_visible(False)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    extras = (int(np.ceil((len(y)-len(y_smooth))/2)),
              int(np.floor((len(y)-len(y_smooth))/2)))
    yp = np.pad(y_smooth, extras, mode='edge')
    return yp

def flatten(o, recursive=False):
    l = []
    if recursive:
        for ct, el in enumerate(o):
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes, Atoms)):
                l += flatten(el, recursive=True)
            else:
                l += [el]

        return l

    else:
        return [item for sublist in o for item in sublist]

def density_estimation(m1, m2, xmin=None, xmax=None, ymin=None, ymax=None, bw=None, point_density=100j):
    """esimates gaussian density of two variables for countour plotting

    # Args:
        m1 (array): x data
        m2 (array): y data
        xmin (float): smallest x value for kernel evaluation. Works best if slightly smaller than actual min(x)
        xmax (float): largest x value for kernel evaluation. Works best if slightly larger than actual max(x)
        ymin (float): 
        ymax (float): 
        bw (float, optional): bin width for Gaussian. Defaults to scipy value of 0.1.
        point_density (complex, optional): density of grid for Gaussian evaluation. Defaults to 100j.

    # Returns:
        tuple: X, Y, Z data for contour plotting
    """
    if xmin is None:
        xmin = min(m1)
    if xmax is None:
        xmax = max(m1)
    if ymin is None:
        ymin = min(m2)
    if ymax is None:
        ymax = max(m2)

    X, Y = np.mgrid[xmin:xmax:point_density, ymin:ymax:point_density]                                                     
    positions = np.vstack([X.ravel(), Y.ravel()])                                                       
    values = np.vstack([m1, m2])                                                                        
    kernel = gaussian_kde(values, bw_method=bw)                                                                 
    Z = np.reshape(kernel(positions).T, X.shape)

    return X, Y, Z


def calculate_topology(file, cutoff1=2.85, cutoff2=5, energies=None):
    """
    Helper to calculate topology information from a structure file 
    for correlating with local energies.

    Parameters:
    - file (str): Path to the input structure.
    - cutoff1 (float, optional): First cutoff value to determine coordination number. Defaults to 2.85.
    - cutoff2 (float, optional): Outer cutoff value for analysing longer distances. Defaults to 6.
    - energies (numpy.ndarray, optional): Array of energies associated with atoms. Defaults to None.

    Returns:
    - dict: A dictionary containing the following keys:
    - 'bs' (numpy.ndarray): Array representing bond information, including topology, length, bond type, and optionally energies.
    - 'fig_data' (list): List containing sublists of bond data categorized by bond type and coordination.
    - 'data' (object): Topology data object.
    """
    print('calculating topology')
    pipeline = import_file(file)

    cb = CreateBondsModifier(cutoff = cutoff1)
    cb.bond_type.id = 1
    cb2 = CreateBondsModifier(cutoff = cutoff2)
    cb.bond_type.id = 2
    pipeline.modifiers.append(cb)
    pipeline.modifiers.append(cb2)
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff = cutoff1))

    pipeline.modifiers.append(ComputePropertyModifier(output_property='Length',
                                                    expressions='BondLength',
                                                    operate_on='bonds'))

    pipeline.modifiers.append(ComputePropertyModifier(output_property='bond_type',
                                                    expressions=['@1.Coordination',
                                                                '@2.Coordination',
                                                                'BondType'],
                                                    operate_on='bonds'))

    data = pipeline.compute()
    c_cn = data.particles['Coordination']
    print('Ovito finished')
    
    if energies is None:
        bs = np.vstack((np.hstack((data.particles.bonds['Topology'].array,
                data.particles.bonds['Length'].array[:, np.newaxis],
                data.particles.bonds['bond_type'].array[:, :2]
                )),
               np.hstack((data.particles.bonds['Topology'].array[:, 1::-1],
               data.particles.bonds['Length'].array[:, np.newaxis],
               data.particles.bonds['bond_type'].array[:, 1::-1]))))
    else:
        bs = np.vstack((np.hstack((data.particles.bonds['Topology'].array,
                data.particles.bonds['Length'].array[:, np.newaxis],
                data.particles.bonds['bond_type'].array[:, :2],
                energies[data.particles.bonds['Topology'].array.T[1]][:, np.newaxis],
                np.min((energies[data.particles.bonds['Topology'].array.T[1]][:, np.newaxis], 
                        energies[data.particles.bonds['Topology'].array.T[0]][:, np.newaxis]), axis=0),
                np.max((energies[data.particles.bonds['Topology'].array.T[1]][:, np.newaxis], 
                        energies[data.particles.bonds['Topology'].array.T[0]][:, np.newaxis]), axis=0)
                )),
               np.hstack((data.particles.bonds['Topology'].array[:, 1::-1],
               data.particles.bonds['Length'].array[:, np.newaxis],
               data.particles.bonds['bond_type'].array[:, 1::-1],
                energies[data.particles.bonds['Topology'].array[:, 1::-1].T[1]][:, np.newaxis],
                np.min((energies[data.particles.bonds['Topology'].array[:, 1::-1].T[1]][:, np.newaxis], 
                        energies[data.particles.bonds['Topology'].array[:, 1::-1].T[0]][:, np.newaxis]), axis=0),
                np.max((energies[data.particles.bonds['Topology'].array[:, 1::-1].T[1]][:, np.newaxis], 
                        energies[data.particles.bonds['Topology'].array[:, 1::-1].T[0]][:, np.newaxis]), axis=0)
                ))))


    bs = bs[np.lexsort((bs[:, 2], bs[:, 0]))]
    b_types = np.ones(len(bs))
    for i in range(1, len(bs)):
        if bs[i][0] == bs[i-1][0]:
            b_types[i] = b_types[i-1] + 1

    fig_b_data = [[bs[np.all(np.vstack((b_types==i, bs[:, 3] == j)), axis=0)] 
                for i in range(1,8)]
                for j in [3,4,5]]

    return {'bs': bs, 'fig_data':fig_b_data, 'data':data, 'c_cn': c_cn}


def opt(at, potential, cell=[True for i in range(6)], steps=100, P=0, fmax=5e-2, write_trajectory=False, traj='/dev/null',
                    silent=False, name='NA', check_convergence=True, return_opt=True, maxstep=0.2):
        '''single-structure optimisation routine'''

        at.calc = potential
        uf = ExpCellFilter(at, cell, scalar_pressure=P, hydrostatic_strain=True) # should ensure only the lattice can move, not atomic positions
        
        if silent:
            opt = BFGSLineSearch(atoms=uf, trajectory=traj, logfile='/dev/null', maxstep=maxstep)
        else:
            opt = BFGSLineSearch(atoms=uf, trajectory=None, maxstep=maxstep)

        opt.run(fmax, steps=steps) # converge to 0.1 GPa

        if check_convergence and not opt.converged():
            warnings.warn(('Warning: failed to converge on structure{}\n' +\
                          'in {} steps').format(name, steps))
        at.calc = None
        
        if return_opt:
            return at, opt
        else:
            nsteps = opt.get_number_of_steps()
            return at, nsteps
        
        
def label_descriptor(cfgs, desc=None,
                   average=False):
    '''calculates the descriptors for an atoms object
    Params: 
        cfgs, list of Atoms
        desc, (str)  descriptor string for quippy
        average, (bool) calculate configuration-averaged kernel or atomistic one
            (latter is memory-intensive). Ignored if
    Returns:
        len(cfgs) list of len(atoms) x len(descriptor) array
    '''
    if desc is None:
        desc_str = 'soap l_max=6 n_max=10 \
                   atom_sigma=0.5 cutoff=5.0 \
                   cutoff_transition_width=1.0 central_weight=1.0'
    else:
        desc_str = desc

    if ' average=' not in desc_str:
        if average:
            desc_str += ' average=T'
        else:
            desc_str += ' average=F'

    desc = Descriptor(desc_str)
    descs = desc.calc_descriptor(cfgs)

    return descs