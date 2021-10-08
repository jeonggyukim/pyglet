# from __future__ import print_function

import os
import os.path as osp

import logging
import glob
# import sys
# import re
# import getpass
# import warnings
# import functools
# import pandas as pd
# import xarray as xr
# import pickle

import numpy as np
import yt

from .io.read_athinput import read_athinput
from .fields.add_fields import add_fields
from .utils.units import Units

# from .io.read_athinput import read_athinput

class LoadSim(object):
    """Class to prepare athena-tigris simulation data analysis. Read input
    parameters, find simulation output files.

    Properties
    ----------
        basedir : str
            base directory of simulation output
        basename : str
            basename (tail) of basedir
        files : dict
            output file paths for hdf and hst
        problem_id : str
            prefix for output files
        par : dict
            input parameters and configure options read from a log file
        ds : yt DataSet
            yt dataset
        mesh : dict
            information about mesh
        load_method : str
            Default value is 'yt'
        nums : list of int
            hdf output numbers
        u : Units object
            simulation unit

    Methods
    -------
        load_hdf() :
            reads hdf file using yt and returns DataSet object
        print_all_properties() :
            prints all attributes and callable methods
    """

    def __init__(self, basedir, savdir=None, load_method='yt',
                 verbose=False):
        """Constructor for LoadSim class.

        Parameters
        ----------
        basedir : str
            Name of the directory where all data is stored
        savdir : str
            Name of the directory where pickled data and figures will be saved.
            Default value is basedir.
        load_method : str
            Load vtk using 'pyathena', 'pythena_classic', or 'yt'. 
            Default value is 'pyathena'.
            If None, savdir=basedir. Default value is None.
        verbose : bool or str or int
            Print verbose messages using logger. If True/False, set logger
            level to 'DEBUG'/'WARNING'. If string, it should be one of the string
            representation of python logging package:
            ('NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Numerical values from 0 ('NOTSET') to 50 ('CRITICAL') are also
            accepted.
        """

        self.basedir = basedir.rstrip('/')
        self.basename = osp.basename(self.basedir)

        self.load_method = load_method
        self.logger = self._get_logger(verbose=verbose)

        if savdir is None:
            self.savdir = self.basedir
        else:
            self.savdir = savdir
            
        self.logger.info('savdir : {:s}'.format(self.savdir))

        self._find_files()

        # Get mesh info
        try:
            self._get_mesh_from_par(self.par)
        except:
            self.logger.warning('Could not get mesh info from input parameters.')
            pass

        if self.par['cooling']['coolftn'] == 'tigress':
            self.u = Units(kind='tigris', muH=1.4271)
        else:
            self.u = Units(kind='tigris', muH=1.4)
        
    def load_hdf(self, num=None, load_method=None, verbose=False):
        """Function to read athena hdf file return DataSet object.
        
        Parameters
        ----------
        num : int
           Snapshot number
        verbose : bool
           Produce verbose message

        Returns
        -------
        ds : yt AthenaPPDataset
        """

        if num is None:
            raise ValueError('Specify snapshot number')
        
        # Override load_method
        if load_method is not None:
            self.load_method = load_method
        
        fname = self.files['hdf'][num]
        
        if self.load_method == 'yt':
            if hasattr(self, 'u'):
                units_override = self.u.units_override
            else:
                units_override = None
            
            # Suppress log and load data
            if not verbose:
                loglevel_ = int(yt.config.ytcfg.get('yt','loglevel'))
                yt.funcs.mylog.setLevel(50)
            self.ds = yt.load(fname, units_override=units_override)
            add_fields(self.ds, self.u)
            if not verbose:
                yt.funcs.mylog.setLevel(loglevel_)

        else:
            self.logger.error('load_method "{0:s}" not recognized.'.format(
                self.load_method) + ' Use "yt".')
        
        return self.ds
    
    def print_all_properties(self):
        """Print all attributes and callable methods
        """
        
        attr_list = list(self.__dict__.keys())
        print('Attributes:\n', attr_list)
        print('\nMethods:')
        method_list = []
        for func in sorted(dir(self)):
            if not func.startswith("__"):
                if callable(getattr(self, func)):
                    method_list.append(func)
                    print(func, end=': ')
                    print(getattr(self, func).__doc__)
                    print('-------------------------')

    def _get_mesh_from_par(self, par):
        """Get mesh info from par['mesh']. Time is set to None.
        """
        m = par['mesh']
        mesh = dict()
        
        mesh['Nx'] = np.array([m['nx1'], m['nx2'], m['nx3']])
        mesh['ndim'] = np.sum(mesh['Nx'] > 1)
        mesh['xmin'] = np.array([m['x1min'], m['x2min'], m['x3min']])
        mesh['xmax'] = np.array([m['x1max'], m['x2max'], m['x3max']])
        mesh['Lx'] = mesh['xmax'] - mesh['xmin']
        mesh['dx'] = mesh['Lx']/mesh['Nx']
        mesh['center'] = 0.5*(mesh['xmin'] + mesh['xmax'])
        mesh['time'] = None
        mesh['refinement'] = m['refinement']
        
        self.mesh = mesh

    def _find_match(self, patterns):
            glob_match = lambda p: sorted(glob.glob(osp.join(self.basedir, *p)))
            for p in patterns:
                f = glob_match(p)
                if f:
                    break
                
            return f

    def _find_files(self):
        """Function to find all output files under basedir and create "files" dictionary.

        """

        # self._out_fmt_def = ['hst', 'hdf']

        if not osp.isdir(self.basedir):
            raise IOError('basedir {0:s} does not exist.'.format(self.basedir))
        
        self.files = dict()

        athinput_patterns = [('athinput.*',),]
        
        hst_patterns = [('*.hst',),
                        ('hst', '*.hst'),]
       
        hdf_patterns = [('*.?????.athdf',),]

        self.logger.info('basedir: {0:s}'.format(self.basedir))

        # Read athinput files
        # Throw warning if not found
        fathinput = self._find_match(athinput_patterns)
        if fathinput:
            self.files['athinput'] = fathinput[0]
            self.par = read_athinput(self.files['athinput'])
            self.logger.info('athinput: {0:s}'.format(self.files['athinput']))
            self.out_fmt = []
            for k in self.par.keys():
                if 'output' in k:
                    self.out_fmt.append(self.par[k]['file_type'])
            self.problem_id = self.par['job']['problem_id']
            self.logger.info('problem_id: {0:s}'.format(self.problem_id))
        else:
            self.par = None
            self.logger.warning('Could not find athinput file in {0:s}'.\
                                format(self.basedir))
            # self.out_fmt = self._out_fmt_def

        # Find history dump and extract problem_id
        if 'hst' in self.out_fmt:
            fhst = self._find_match(hst_patterns)
            if fhst:
                self.files['hst'] = fhst[0]
                if not hasattr(self, 'problem_id'):
                    # Extract problem_id from /basedir/problem_id.hst
                    self.problem_id = osp.basename(self.files['hst'])[:-4]
                self.logger.info('hst: {0:s}'.format(self.files['hst']))
            else:
                self.logger.warning('Could not find hst file in {0:s}'.\
                                    format(self.basedir))

        # Find hdf files
        if 'hdf5' in self.out_fmt:
            self.files['hdf'] = self._find_match(hdf_patterns)
            self.nums = [int(f[-11:-6]) for f in self.files['hdf']]
            if self.nums:
                self.logger.info('hdf: {0:s} nums: {1:d}-{2:d}'.format(
                    osp.dirname(self.files['hdf'][0]),
                        self.nums[0], self.nums[-1]))

    def _get_logger(self, verbose=False):
        """Function to set logger and default verbosity.

        Parameters
        ----------
        verbose: bool or str or int
            Set logging level to "INFO"/"WARNING" if True/False.
        """

        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        if verbose is True:
            self.loglevel_def = 'INFO'
        elif verbose is False:
            self.loglevel_def = 'WARNING'
        elif verbose in levels + [l.lower() for l in levels]:
            self.loglevel_def = verbose.upper()
        elif isinstance(verbose, int):
            self.loglevel_def = verbose
        else:
            raise ValueError('Cannot recognize option {0:s}.'.format(verbose))
        
        l = logging.getLogger(self.__class__.__name__.split('.')[-1])
 
        try:
            if not l.hasHandlers():
                h = logging.StreamHandler()
                f = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
                # f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                h.setFormatter(f)
                l.addHandler(h)
                l.setLevel(self.loglevel_def)
            else:
                l.setLevel(self.loglevel_def)
        except AttributeError: # for python 2 compatibility
            if not len(l.handlers):
                h = logging.StreamHandler()
                f = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
                # f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                h.setFormatter(f)
                l.addHandler(h)
                l.setLevel(self.loglevel_def)
            else:
                l.setLevel(self.loglevel_def)

        return l
    
    class Decorators(object):
        """Class containing a collection of decorators for prompt reading of analysis
        output, (reprocessed) hst, and zprof. Used in child classes.

        """
        
        def check_pickle(read_func):
            @functools.wraps(read_func)
            def wrapper(cls, *args, **kwargs):

                # Convert positional args to keyword args
                from inspect import getcallargs
                call_args = getcallargs(read_func, cls, *args, **kwargs)
                call_args.pop('self')
                kwargs = call_args

                try:
                    prefix = kwargs['prefix']
                except KeyError:
                    prefix = '_'.join(read_func.__name__.split('_')[1:])

                if kwargs['savdir'] is not None:
                    savdir = kwargs['savdir']
                else:
                    savdir = osp.join(cls.savdir, prefix)

                force_override = kwargs['force_override']

                # Create savdir if it doesn't exist
                try:
                    if not osp.exists(savdir):
                        force_override = True
                        os.makedirs(savdir)
                except FileExistsError:
                    print('Directory exists: {0:s}'.format(savdir))
                except PermissionError as e:
                    print('Permission Error: ', e)

                if 'num' in kwargs:
                    fpkl = osp.join(savdir, '{0:s}_{1:04d}.p'.format(prefix, kwargs['num']))
                else:
                    fpkl = osp.join(savdir, '{0:s}.p'.format(prefix))

                if not force_override and osp.exists(fpkl):
                    cls.logger.info('Read from existing pickle: {0:s}'.format(fpkl))
                    res = pickle.load(open(fpkl, 'rb'))
                    return res
                else:
                    cls.logger.info('[check_pickle]: Read original dump.')
                    # If we are here, force_override is True or history file is updated.
                    res = read_func(cls, **kwargs)
                    try:
                        pickle.dump(res, open(fpkl, 'wb'))
                    except (IOError, PermissionError) as e:
                        cls.logger.warning('Could not pickle to {0:s}.'.format(fpkl))
                    return res
                
            return wrapper
                   
        def check_pickle_hst(read_hst):
            
            @functools.wraps(read_hst)
            def wrapper(cls, *args, **kwargs):
                if 'savdir' in kwargs:
                    savdir = kwargs['savdir']
                else:
                    savdir = osp.join(cls.savdir, 'hst')

                if 'force_override' in kwargs:
                    force_override = kwargs['force_override']
                else:
                    force_override = False

                # Create savdir if it doesn't exist
                if not osp.exists(savdir):
                    try:
                        os.makedirs(savdir)
                        force_override = True
                    except (IOError, PermissionError) as e:
                        cls.logger.warning('Could not make directory')

                fpkl = osp.join(savdir, osp.basename(cls.files['hst']) +
                                '.{0:s}.mod.p'.format(cls.basename))

                # Check if the original history file is updated
                if not force_override and osp.exists(fpkl) and \
                   osp.getmtime(fpkl) > osp.getmtime(cls.files['hst']):
                    cls.logger.info('[read_hst]: Reading pickle.')
                    #print('[read_hst]: Reading pickle.')
                    hst = pd.read_pickle(fpkl)
                    cls.hst = hst
                    return hst
                else:
                    cls.logger.info('[read_hst]: Reading original hst file.')
                    # If we are here, force_override is True or history file is updated.
                    # Call read_hst function
                    hst = read_hst(cls, *args, **kwargs)
                    try:
                        hst.to_pickle(fpkl)
                    except (IOError, PermissionError) as e:
                        cls.logger.warning('[read_hst]: Could not pickle hst to {0:s}.'.format(fpkl))
                    return hst

            return wrapper


# Would be useful to have something like this for each problem
class LoadSimAll(object):
    """Class to load multiple simulations

    """
    def __init__(self, models):

        self.models = list(models.keys())
        self.basedirs = dict()
        
        for mdl, basedir in models.items():
            self.basedirs[mdl] = basedir

    def set_model(self, model, savdir=None, load_method='yt',
                  verbose=False):
        self.model = model
        self.s = LoadSim(self.basedirs[model], savdir=savdir,
                         load_method=load_method, verbose=verbose)
        return self.s

