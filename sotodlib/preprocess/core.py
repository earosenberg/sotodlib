"""Base Class and PIPELINE register for the preprocessing pipeline scripts."""
import logging
import numpy as np
from .. import core
from so3g.proj import Ranges, RangesMatrix
from scipy.sparse import csr_array

class _Preprocess(object):
    """The base class for Preprocessing modules which defines the required
    functions and keys required in the configurations.

    Each preprocess module has four overwritable functions that are called by
    the processing scripts in site_pipeline. These four functions are each
    controlled by a specific key in a configuration dictionary passed to the
    module on creation.

    The configuration dictionary has 5 special keys: ``name``, ``process``,
    ``calc``, ``save``, and ``select``. ``name`` is the name used to register 
    the module with the PIPELINE registry. The other four keys are matched to 
    functions in the module, if the key is not present then that function will 
    be skipped when the preprocessing pipeline is run.

    There are two special AxisManagers expected to be part of the preprocessing
    pipeline. ``aman`` is the "standard" time ordered data AxisManager that is
    loaded via our default styles. ``proc_aman`` is the preprocess AxisManager,
    this is carry the data products that will be saved to whatever Metadata
    Archive is connected to the preprocessing pipeline. 
    """    

    def __init__(self, step_cfgs):
        self.process_cfgs = step_cfgs.get("process")
        self.calc_cfgs = step_cfgs.get("calc")
        self.save_cfgs = step_cfgs.get("save")
        self.select_cfgs = step_cfgs.get("select")

    def process(self, aman, proc_aman):
        """ This function makes changes to the time ordered data AxisManager.
        Ex: calibrating or detrending the timestreams. This function will use
        any configuration information under the ``process`` key of the
        configuration dictionary and is not expected to change or alter
        proc_aman.


        Arguments
        ---------
        aman : AxisManager 
            The time ordered data
        proc_aman : AxisManager 
            Any information generated by previous elements in the preprocessing 
            pipeline.
        """
        if self.process_cfgs is None:
            return
        raise NotImplementedError
        
    def calc_and_save(self, aman, proc_aman):
        """ This function calculates data products of some sort off of the time
        ordered data AxisManager.

        Ex: Calcuating the white noise of the timestream. This function will use
        any configuration information under the ``calc`` key of the
        configuration dictionary and can call the save function to make
        changes to proc_aman.

        Arguments
        ---------
        aman : AxisManager 
            The time ordered data
        proc_aman : AxisManager 
            Any information generated by previous elements in the preprocessing 
            pipeline.
        """
        if self.calc_cfgs is None:
            return
        raise NotImplementedError
    
    def save(self, proc_aman, *args):
        """ This function wraps new information into the proc_aman and will use
        any configuration information under the ``save`` key of the
        configuration dictionary.

        Arguments
        ---------
        proc_aman : AxisManager 
            Any information generated by previous elements in the preprocessing 
            pipeline.
        args : any
            Any additional information ``calc_and_save`` needs to send to the
            save function.
        """
        if self.save_cfgs is None:
            return
        raise NotImplementedError
        
    def select(self, meta, proc_aman=None):
        """ This function runs any desired data selection of the preprocessing
        pipeline results. Assumes the pipeline has already been run and that the
        resulting proc_aman is now saved under the ``preprocess`` key in the
        ``meta`` AxisManager loaded via context.

        Ex: removing detectors with white noise above some limit. This function will use
        any configuration information under the ``select`` key.


        Arguments
        ---------
        meta : AxisManager 
            Metadata related to the specific observation

        Returns
        -------
        meta : AxisManager 
            Metadata where non-selected detectors have been removed
        """

        if self.select_cfgs is None:
            return meta
        raise NotImplementedError

    @classmethod
    def gen_metric(cls, meta, proc_aman):
        """ Generate a QA metric from the output of this process.

        Arguments
        ---------
        meta : AxisManager
            Metadata related to the specific observation
        proc_aman : AxisManager
            The output of the preprocessing pipeline.

        Returns
        -------
        line : dict
            InfluxDB line entry elements to be fed to
            `site_pipeline.monitor.Monitor.record`
        """
        raise NotImplementedError

    @staticmethod
    def register(process_class):
        """Registers a new modules with the PIPELINE"""
        name = process_class.name

        if Pipeline.PIPELINE.get(name) is None:
            Pipeline.PIPELINE[name] = process_class
        else:
            raise ValueError(
                f"Preprocess Module of name {name} is already Registered"
            )

def _zeros_cls( item ):
    """return a callable zeros class that exactly matches type item for use 
    with wrap_new"""
    if isinstance( item, np.ndarray):
        return lambda shape: np.zeros( shape, dtype = item.dtype)
    elif isinstance( item, RangesMatrix):
        return RangesMatrix.zeros
    elif isinstance( item, Ranges ):
        def temp(shape):
            assert len(shape) == 1
            return Ranges( shape[0] )
        return temp
    elif isinstance( item, csr_array):
        return lambda shape: csr_array(tuple(shape), dtype=item.dtype)
    else:
        raise ValueError(f"Cannot find zero type for {type(item)}")

def _ranges_matrix_match( o, n, oidx, nidx):
    """align RangesMatrix n entries to RangesMatrix o"""
    assert len(oidx)==len(nidx)
    if len(oidx) > 2:
        raise NotImplemented
    for i, x in zip( oidx[0], nidx[0]):
        o.ranges[i] = _ranges_match( 
            o.ranges[i], n.ranges[x],
            [oidx[1]], [nidx[1]]
        )
    return o.copy()

def _ranges_match( o, n, oidx, nidx):
    """align Ranges n to Ranges o"""
    assert len(oidx)==len(nidx)
    assert len(oidx)==1
    omsk = o.mask()
    nmsk = n.mask()
    omsk[oidx[0]] = nmsk[nidx[0]]
    return Ranges.from_mask(omsk)

def _expand(new, full, wrap_valid=True):
    """new will become a top level axismanager in full once it is matched to
    size"""
    if 'dets' in new._axes:
        _, fs_dets, ns_dets = full.dets.intersection(
            new.dets, 
            return_slices=True
        )
    if 'samps' in new._axes:
        _, fs_samps, ns_samps = full.samps.intersection(
            new.samps, 
            return_slices=True
        )
    else:
        fs_samps = slice(None)

    out = core.AxisManager()
    for k, v in full._axes.items():
        if k in list(new._axes.keys())+['dets','samps']:
            out._axes[k] = v 

    for a in new._axes:
        if a not in out:
            out.add_axis( new[a] )
    for k, v in new._fields.items():
        if isinstance(v, core.AxisManager):
            out.wrap( k, _expand( v, full) )
        else:
            out.wrap_new( k, new._assignments[k], cls=_zeros_cls(v))
            oidx=[]; nidx=[]
            for a in new._assignments[k]:
                if a == 'dets':
                    oidx.append(fs_dets)
                    nidx.append(ns_dets)
                elif a == 'samps':
                    oidx.append(fs_samps)
                    nidx.append(ns_samps)
                else:
                    oidx.append(slice(None))
                    nidx.append(slice(None))
            oidx = tuple(oidx)
            nidx = tuple(nidx)
            if isinstance(out[k], RangesMatrix):
                assert new._assignments[k][-1] == 'samps'
                out[k] = _ranges_matrix_match( out[k], v, oidx, nidx)
            elif isinstance(out[k], Ranges):
                assert new._assignments[k][0] == 'samps'
                out[k] = _ranges_match( out[k], v, oidx, nidx)
            else:
                out[k][oidx] = v[nidx]
    if wrap_valid:
        x = Ranges( full.samps.count )
        m = x.mask()
        m[fs_samps] = True
        v = Ranges.from_mask(m)

        valid = RangesMatrix( 
            [v if i in fs_dets else x for i in range(full.dets.count)]
        )
        out.wrap('valid',valid,[(0,'dets'),(1,'samps')])
    return out

def update_full_aman(proc_aman, full, wrap_valid):
    """Copy new fields from proc_aman[dets,samps] over to 
    full[full-dets,full-samps] after correct re-sizing and indexing.

    Arguments
    ----------
    proc_aman: AxisManager
        A preprocess AxisManager from a pipeline run. The dets,samps axes in 
        proc_aman is assumed to be a subset of the dets,samps axes in full
    full: AxisManager
        A full shape AxisManager that begins the pipeline as the original shape
        of the TOD AxisManager
    """
    for fld in proc_aman._fields:
        if fld not in full._fields:
            assert isinstance(proc_aman[fld], core.AxisManager)
            full.wrap( 
                fld,
                _expand( proc_aman[fld], full, wrap_valid=wrap_valid)
            )

class Pipeline(list):
    """This class is designed to create and run pipelines out of a series of
    different preprocessing modules (classes that inherent from _Preprocess). It
    inherits list object. It also contains the registration of all possible
    preprocess modules in Pipeline.PIPELINE
    """

    PIPELINE = {}

    def __init__(self, modules, logger=None, wrap_valid=True):
        """
        Arguments
        ---------
        modules: iterable
            A list or other iterable that contains either instantiated
            _Preprocess instances or the configuration dictionary used to
            instantiate a module
        logger: optional
            logging.logger instance used by the pipeline to send updates
        """
        if logger is None:
            logger = logging.getLogger("pipeline")
        self.logger = logger
        self.wrap_valid = wrap_valid
        super().__init__( [self._check_item(item) for item in modules])
    
    def _check_item(self, item):
        if isinstance(item, _Preprocess):
            return item
        elif isinstance(item, dict):
            name = item.get("name")
            if name is None:
                raise ValueError(f"Processes made from dictionary must have a 'name' key")
            cls = self.PIPELINE.get(name)
            if cls is None:
                raise ValueError(f"'{name}' not registered as a pipeline element")
            return cls(item)
        else:
            raise ValueError(f"Unknown type created a pipeline element")
    
    # make pipeline have all the list pieces
    def append(self, item):
        super().append( self._check_item(item) )
    def insert(self, index, item):
        super().insert(index, self._check_item(item))
    def extend(self, index, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend( [self._check_item(item) for item in other])
    def __setitem__(self, index, item):
        super().__setitem__(index, self._check_item(item))
    
    def run(self, aman, proc_aman=None, select=True):
        """
        The main workhorse function for the pipeline class. This function takes
        an AxisManager TOD and successively runs the pipeline of preprocessing
        modules on the AxisManager. The order of operations called by run are::

            for process in pipeline:
                process.process()
                process.calc_and_save()
                    process.save() ## called by process.calc_and_save()
                process.select()

        Arguments
        ---------
        aman: AxisManager
            A TOD object. Generally expected to be raw, unprocessed data. This
            axismanager will be edited in place by the process and select
            functions of each preprocess module
        proc_aman: AxisManager (Optional)
            A preprocess axismanager. If this is provided it is assumed that the
            pipeline has previously been run on this specific TOD and has
            returned this preprocess axismanager. In this case, calls to
            ``process.calc_and_save()`` are skipped as the information is
            expected to be present in this AxisManager.
        select: boolean (Optional)
            if True, the aman detector axis is restricted as described in
            each preprocess module. Most pipelines are developed with 
            select=True. Running select=False may produce unstable behavior

        Returns
        -------
        proc_aman: AxisManager
            A preprocess axismanager that contains all data products calculated
            throughout the running of the pipeline
        
        """
        if proc_aman is None:
            proc_aman = core.AxisManager( aman.dets, aman.samps)
            full = core.AxisManager( aman.dets, aman.samps)
            run_calc = True
        else:
            if aman.dets.count != proc_aman.dets.count or not np.all(aman.dets.vals == proc_aman.dets.vals):
                self.logger.warning("proc_aman has different detectors than aman. Cutting aman to match")
                det_list = [det for det in proc_aman.dets.vals if det in aman.dets.vals]
                aman.restrict('dets', det_list)
                proc_aman.restrict('dets', det_list)
            full = proc_aman.copy()
            run_calc = False
        
        success = 'end'
        for process in self:
            self.logger.info(f"Running {process.name}")
            process.process(aman, proc_aman)
            if run_calc:
                process.calc_and_save(aman, proc_aman)
                update_full_aman( proc_aman, full, self.wrap_valid)
                
            if select:
                process.select(aman, proc_aman)
                proc_aman.restrict('dets', aman.dets.vals)
            self.logger.debug(f"{proc_aman.dets.count} detectors remaining")
            
            if aman.dets.count == 0:
                success = process.name
                break
        
        return full, success
        
