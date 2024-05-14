import numpy as np
from pixell import enmap, utils, tilemap, bunch
import so3g.proj

from .. import coords
from .utilities import *
from .pointing_matrix import *
from types import SimpleNamespace

hp = import_optional('healpy')
h5py = import_optional('h5py')

class DemodMapmaker:
    def __init__(self, signals=[], noise_model=None, dtype=np.float32, verbose=False, comps='TQU', singlestream=False):
        """Initialize a FilterBin Mapmaker for demodulated data
        Arguments:
        * signals: List of Signal-objects representing the models that will be solved
          jointly for. Typically this would be the sky map and the cut samples. NB!
          The way the cuts currently work, they *MUST* be the first signal specified.
          If not, the equation system will be inconsistent and won't converge.
        * noise_model: A noise model constructor which will be used to initialize the
          noise model for each observation. Can be overriden in add_obs.
        * dtype: The data type to use for the time-ordered data. Only tested with float32
        * verbose: Whether to print progress messages. Not implemented"""
        if noise_model is None:
            noise_model = NmatWhite()
        self.signals      = signals
        self.dtype        = dtype
        self.verbose      = verbose
        self.noise_model  = noise_model
        self.data         = []
        self.dof          = MultiZipper()
        self.ready        = False
        self.ncomp        = len(comps)
        self.singlestream = singlestream

    def add_obs(self, id, obs, noise_model=None, deslope=False, split_labels=None, det_weights='ivar', qp_kwargs={}):
        # Prepare our tod
        ctime  = obs.timestamps
        srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
        if self.singlestream == False:
            # now we have 3 signals, dsT / demodQ / demodU. We pack them into an array with shape (3,...)
            tod    = np.array([obs.dsT.astype(self.dtype, copy=False), obs.demodQ.astype(self.dtype, copy=False), obs.demodU.astype(self.dtype, copy=False)])
            if deslope:
                for i in range(self.ncomp):
                    utils.deslope(tod[i], w=5, inplace=True)
        else:
            tod = obs.signal.astype(self.dtype, copy=False)
            if deslope:
                utils.deslope(tod, w=5, inplace=True)
        # Allow the user to override the noise model on a per-obs level
        if noise_model is None: noise_model = self.noise_model
        # Build the noise model from the obs unless a fully
        # initialized noise model was passed
        if noise_model.ready:
            nmat = noise_model
        else:
            try:
                if self.singlestream==False:
                    # we build the noise model from demodQ. For now we will apply it to Q and U also, but this will change
                    nmat = noise_model.build(tod[1], srate=srate) # I have to define how the noise model will be build
                else:
                    nmat = noise_model.build(tod, srate=srate)
            except Exception as e:
                msg = f"FAILED to build a noise model for observation='{id}' : '{e}'"
                raise RuntimeError(msg)
        # And apply it to the tod
        '''
        if self.singlestream==False:
            for i in range(self.ncomp):
                tod[i]    = nmat.apply(tod[i])
        else:
            tod = nmat.apply(tod)
        '''
        # Add the observation to each of our signals
        for signal in self.signals:
            signal.add_obs(id, obs, nmat, tod, split_labels=split_labels, det_weights=det_weights, qp_kwargs=qp_kwargs)
        # Save what we need about this observation
        self.data.append(bunch.Bunch(id=id, ndet=obs.dets.count, nsamp=len(ctime), dets=obs.dets.vals, nmat=nmat))

class DemodSignal:
    """This class represents a thing we want to solve for, e.g. the sky, ground, cut samples, etc."""
    def __init__(self, name, ofmt, output, ext):
        """Initialize a Signal. It probably doesn't make sense to construct a generic signal
        directly, though. Use one of the subclasses.
        Arguments:
        * name: The name of this signal, e.g. "sky", "cut", etc.
        * ofmt: The format used when constructing output file prefix
        * output: Whether this signal should be part of the output or not.
        * ext: The extension used for the files.
        """
        self.name   = name
        self.ofmt   = ofmt
        self.output = output
        self.ext    = ext
        self.dof    = None
        self.ready  = False
    def add_obs(self, id, obs, nmat, Nd): pass
    def prepare(self): self.ready = True
    def to_work  (self, x): return x.copy()
    def from_work(self, x): return x
    def write   (self, prefix, tag, x): pass

class DemodSignalMap(DemodSignal):
    """Signal describing a non-distributed sky map."""
    def __init__(self, shape, wcs, comm, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits", dtype=np.float32, sys=None, recenter=None, tile_shape=(500,500), tiled=False, Nsplits=1, singlestream=False):
        """Signal describing a sky map in the coordinate system given by "sys", which defaults
        to equatorial coordinates. If tiled==True, then this will be a distributed map with
        the given tile_shape, otherwise it will be a plain enmap."""
        DemodSignal.__init__(self, name, ofmt, output, ext)
        self.comm  = comm
        self.comps = comps
        self.sys   = sys
        self.recenter = recenter
        self.dtype = dtype
        self.tiled = tiled
        self.data  = {}
        self.Nsplits = Nsplits
        self.singlestream = singlestream
        self.wrapper = lambda x : x
        ncomp      = len(comps)
        shape      = tuple(shape[-2:])
        if tiled:
            geo = tilemap.geometry(shape, wcs, tile_shape=tile_shape)
            self.rhs = tilemap.zeros(geo.copy(pre=(Nsplits,ncomp,)),      dtype=dtype)
            self.div = tilemap.zeros(geo.copy(pre=(Nsplits,ncomp,ncomp)), dtype=dtype)
            self.hits= tilemap.zeros(geo.copy(pre=(Nsplits,)),            dtype=dtype)
        else:
            self.rhs = enmap.zeros((Nsplits, ncomp)     +shape, wcs, dtype=dtype)
            self.div = enmap.zeros((Nsplits,ncomp,ncomp)+shape, wcs, dtype=dtype)
            self.hits= enmap.zeros((Nsplits,)+shape, wcs, dtype=dtype)

    def add_obs(self, id, obs, nmat, Nd, pmap=None, split_labels=None, det_weights='ivar', qp_kwargs={}):
        # Nd will have 3 components, corresponding to ds_T, demodQ, demodU with the noise model applied
        """Add and process an observation, building the pointing matrix
        and our part of the RHS. "obs" should be an Observation axis manager,
        nmat a noise model, representing the inverse noise covariance matrix,
        and Nd the result of applying the noise model to the detector time-ordered data.
        """
        ctime  = obs.timestamps
        for n_split in range(self.Nsplits):
            if pmap is None:
                # Build the local geometry and pointing matrix for this observation
                if self.recenter:
                    rot = recentering_to_quat_lonlat(*evaluate_recentering(self.recenter, ctime=ctime[len(ctime)//2], geom=(self.rhs.shape, self.rhs.wcs), site=unarr(obs.site)))
                else: rot = None
                if split_labels == None:
                    flagnames = ['glitch_flags'] # None
                    # this is the case with no splits
                else:
                    flagnames = ['glitch_flags', split_labels[n_split]]
                rangesmatrix = get_flags(obs, flagnames)
                threads='domdir'
                pmap_local = coords.pmat.P.for_tod(obs, comps=self.comps, geom=self.rhs.geometry, rot=rot, threads=threads, weather=unarr(obs.weather), site=unarr(obs.site), cuts=rangesmatrix, hwp=True, qp_kwargs=qp_kwargs)
            else:
                pmap_local = pmap

            det_weightsT, det_weightsQU = process_detweight_str(det_weights, nmat)
            if self.singlestream==False:
                obs_rhs, obs_div, obs_hits = project_all_demod(pmap_local, obs.dsT, obs.demodQ, obs.demodU, det_weightsT, det_weightsQU, self.ncomp, self.wrapper)
            else:
                obs_rhs, obs_div, obs_hits = project_all_single(pmap_local, Nd, None, self.ncomp, self.wrapper)
            # Update our full rhs and div. This works for both plain and distributed maps
            self.rhs[n_split] = self.rhs[n_split].insert(obs_rhs, op=np.ndarray.__iadd__)
            self.div[n_split] = self.div[n_split].insert(obs_div, op=np.ndarray.__iadd__)
            self.hits[n_split] = self.hits[n_split].insert(obs_hits[0],op=np.ndarray.__iadd__)
            # Save the per-obs things we need. Just the pointing matrix in our case.
            # Nmat and other non-Signal-specific things are handled in the mapmaker itself.
            self.data[(id,n_split)] = bunch.Bunch(pmap=pmap_local, obs_geo=obs_rhs.geometry)
        del Nd

    def prepare(self):
        """Called when we're done adding everything. Sets up the map distribution,
        degrees of freedom and preconditioner."""
        if self.ready: return
        if self.tiled:
            self.geo_work = self.rhs.geometry
            self.rhs  = tilemap.redistribute(self.rhs, self.comm)
            self.div  = tilemap.redistribute(self.div, self.comm)
            self.hits = tilemap.redistribute(self.hits,self.comm)
        else:
            if self.comm is not None:
                self.rhs  = utils.allreduce(self.rhs, self.comm)
                self.div  = utils.allreduce(self.div, self.comm)
                self.hits = utils.allreduce(self.hits,self.comm)
        # We will output the weighted_map, the weights map, the hits map. We don't need to invert any more.
#        self.idiv = []
#        for n_split in range(self.Nsplits):
#            self.idiv.append( safe_invert_div(self.div[n_split]) )
        self.ready = True

    @property
    def ncomp(self): return len(self.comps)

    def to_work(self, map):
        if self.tiled: return tilemap.redistribute(map, self.comm, self.geo_work.active)
        else: return map.copy()

    def from_work(self, map):
        if self.tiled: return tilemap.redistribute(map, self.comm, self.rhs.geometry.active)
        else: return utils.allreduce(map, self.comm)

    def write(self, prefix, tag, m):
        if not self.output: return
        oname = self.ofmt.format(name=self.name)
        oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
        if self.tiled:
            tilemap.write_map(oname, m, self.comm)
        else:
            if self.comm is None or self.comm.rank == 0:
                enmap.write_map(oname, m)
        return oname

class DemodSignalMapHealpix(DemodSignal):
    def __init__(self, nside, nside_tile=None, comm=None, comps="TQU", name="sky", ofmt="{name}", output=True,
            ext="fits", dtype=np.float32, Nsplits=1, singlestream=False):
        DemodSignal.__init__(self, name, ofmt, output, ext)
        self.comm  = comm
        self.comps = comps
        self.dtype = dtype
        self.tiled = (nside_tile is not None)
        self.data  = {}
        self.Nsplits = Nsplits
        self.singlestream = singlestream
        ncomp      = len(comps)
        ordering = 'NEST' # Only NEST allowed for tiled maps
        self.hp_geom = SimpleNamespace(nside=nside, nside_tile=nside_tile, ordering=ordering)
        npix = 12 * nside**2
        self.rhs = np.zeros((Nsplits, ncomp, npix), dtype=dtype)
        self.div = np.zeros((Nsplits, ncomp, ncomp, npix), dtype=dtype)
        self.hits = np.zeros((Nsplits, npix), dtype=dtype)
        if self.tiled:
            self.wrapper = untile_healpix
        else:
            self.wrapper = lambda x:x

    def add_obs(self, id, obs, nmat, Nd, pmap=None, split_labels=None, det_weights='ivar', qp_kwargs={}):
        for n_split in range(self.Nsplits):
            if pmap is None:
                # Build the local geometry and pointing matrix for this observation
                # we handle cuts here through obs.flags
                if split_labels is None:
                    # this is the case with no splits
                    flagnames = ['glitch_flags'] # None
                else:
                    flagnames = ['glitch_flags', split_labels[n_split]]
                rangesmatrix = get_flags(obs, flagnames)
                threads = ["tiles", "simple"][self.hp_geom.nside_tile is None] # 'simple' is likely to perform very poorly but no other method implemented for untiled healpix
                pmap_local = coords.pmat.P.for_tod(obs, comps=self.comps, geom=None, hp_geom=self.hp_geom, threads=threads, weather=unarr(obs.weather), site=unarr(obs.site), cuts=rangesmatrix, hwp=True, qp_kwargs=qp_kwargs)
            else:
                pmap_local = pmap

            det_weightsT, det_weightsQU = process_detweight_str(det_weights, nmat)
            # Build the RHS for this observation
            if self.singlestream==False:
                obs_rhs, obs_div, obs_hits = project_all_demod(pmap_local, obs.dsT, obs.demodQ, obs.demodU, det_weightsT, det_weightsQU, self.ncomp, self.wrapper)
            else:
                obs_rhs, obs_div, obs_hits = project_all_single(pmap_local, Nd, None, self.ncomp, self.wrapper) # Should there be det_weights here?

            # Update our full rhs and div. This works for both plain and distributed maps
            self.rhs[n_split] = obs_rhs
            self.div[n_split] = obs_div
            self.hits[n_split] = obs_hits[0]
            # Save the per-obs things we need. Just the pointing matrix in our case.
            # Nmat and other non-Signal-specific things are handled in the mapmaker itself.
            self.data[(id,n_split)] = bunch.Bunch(pmap=pmap_local)
            self.ready = True ## TODO  we'll see how tiling works
        del Nd ## TODO you can prob get rid of Nd entirely

    def prepare(self):
        ## For now need this method for compatibility
        ## In the future may support handling of partial maps through a scheme similar to pixell tilemap
        return

    @property
    def ncomp(self): return len(self.comps)

    def write(self, prefix, tag, m, write_partial=False):
        if not self.output: return
        assert (self.comm is None or self.comm.rank == 0) # Not really supporting comm but leave this here
        oname = self.ofmt.format(name=self.name)
        oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
        if write_partial and self.ext != 'fits':
            raise NotImplementedError("write_partial only supported for fits")

        if self.ext == "fits":
            if hp is None:
                raise ValueError("Cannot save healpix map as fits; healpy could not be imported. Install healpy or save as .npy or .h5")
            if m.ndim > 2:
                m = np.reshape(m, (np.product(m.shape[:-1]), m.shape[-1]), order='C') # Flatten wrapping axes; healpy.write_map can't handle >2d array
            hp.write_map(oname, m.view(self.dtype), nest=(self.hp_geom.ordering=='NEST'), partial=write_partial, overwrite=True) ## TODO Replace hard-coded nest
        elif self.ext == "npy":
            np.save(oname, m.view(self.dtype))
        elif self.ext in ['h5', 'hdf5']:
            if h5py is None:
                raise ValueError("Cannot save healpix map as hdf5; h5py could not be imported. Install h5py or save as .npy or .fits")
            with h5py.File(oname, 'w') as f:
                dset = f.create_dataset("data", m.shape, dtype=self.dtype, data=m)
        else:
            raise ValueError(f"Unknown extension {self.ext}")

def project_rhs_demod(pmap, signalT, signalQ, signalU, det_weightsT, det_weightsQU, wrapper=lambda x:x):
    zeros = lambda *args, **kwargs : wrapper(pmap.zeros(*args, **kwargs))
    to_map = lambda *args, **kwargs : wrapper(pmap.to_map(*args, **kwargs))

    rhs = zeros()
    ## Add support for different detweights
    rhs_T = to_map(signal=signalT, comps='T', det_weights=det_weightsT)
    rhs_demodQ = to_map(signal=signalQ, comps='QU', det_weights=det_weightsQU)
    rhs_demodU = to_map(signal=signalU, comps='QU', det_weights=det_weightsQU)
    rhs_demodQU = zeros(super_shape=(2), comps='QU',)

    rhs_demodQU[0][:] = rhs_demodQ[0] - rhs_demodU[1]
    rhs_demodQU[1][:] = rhs_demodQ[1] + rhs_demodU[0]
    del rhs_demodQ, rhs_demodU

    # we write into the rhs.
    rhs[0] = rhs_T[0]
    rhs[1] = rhs_demodQU[0]
    rhs[2] = rhs_demodQU[1]
    return rhs

def project_div_demod(pmap, det_weightsT, det_weightsQU, ncomp, wrapper=lambda x:x):
    zeros = lambda *args, **kwargs : wrapper(pmap.zeros(*args, **kwargs))
    to_weights = lambda *args, **kwargs : wrapper(pmap.to_weights(*args, **kwargs))

    div = zeros(super_shape=(ncomp, ncomp))
    # Build the per-pixel inverse covmat for this observation
    wT = to_weights(comps='T', det_weights=det_weightsT)
    wQU = to_weights(comps='T', det_weights=det_weightsQU)
    div[0,0] = wT
    div[1,1] = wQU
    div[2,2] = wQU
    return div

def project_all_demod(pmap, signalT, signalQ, signalU, det_weightsT, det_weightsQU, ncomp, wrapper=lambda x:x):
    rhs =  project_rhs_demod(pmap, signalT, signalQ, signalU, det_weightsT, det_weightsQU, wrapper)
    div = project_div_demod(pmap, det_weightsT, det_weightsQU, ncomp, wrapper)
    hits = wrapper(pmap.to_map(signal=np.ones_like(signalT))) ## Note hits is *not* weighted by det_weights
    return rhs, div, hits

def project_all_single(pmap, Nd, det_weights, ncomp, wrapper=lambda x:x):
    rhs = wrapper(pmap.to_map(signal=Nd, comps='TQU', det_weights=det_weights))
    div = pmap.zeros(super_shape=(ncomp, ncomp))
    pmap.to_weights(dest=div, comps='TQU', det_weights=det_weights)
    div = wrapper(div)
    hits = wrapper(pmap.to_map(np.ones_like(Nd)))
    return rhs, div, hits
