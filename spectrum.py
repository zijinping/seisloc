from numpy.fft import fft,fftshift,fftfreq
import numpy as np
import matplotlib.pyplot as plt


def _convolve(a, b, axis=-1):  
    return np.apply_along_axis(np.convolve, axis, a, b, mode='same')

def _boundaries(freqs): 
    freq_boundaries = np.zeros((len(freqs) + 1,), dtype=float)
    df = freqs[1] - freqs[0]
    freq_boundaries[1:] = freqs + 0.5 * df
    freq_boundaries[0] = freqs[0] - 0.5 * df
    return freq_boundaries

class Bunch(dict):
    """
    A dictionary that also provides access via attributes.

    Additional methods update_values and update_None provide
    control over whether new keys are added to the dictionary
    when updating, and whether an attempt to add a new key is
    ignored or raises a KeyError.

    The Bunch also prints differently than a normal
    dictionary, using str() instead of repr() for its
    keys and values, and in key-sorted order.  The printing
    format can be customized by subclassing with a different
    str_ftm class attribute.  Do not assign directly to this
    class attribute, because that would substitute an instance
    attribute which would then become part of the Bunch, and
    would be reported as such by the keys() method.

    To output a string representation with
    a particular format, without subclassing, use the
    formatted() method.
    """

    str_fmt = "{0!s:<{klen}} : {1!s:>{vlen}}\n"

    def __init__(self, *args, **kwargs):
        """
        *args* can be dictionaries, bunches, or sequences of
        key,value tuples.  *kwargs* can be used to initialize
        or add key, value pairs.
        """
        dict.__init__(self)
        for arg in args:
            self.update(arg)
        self.update(kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError("'Bunch' object has no attribute '%s'" % name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        return self.formatted()

    def formatted(self, fmt=None, types=False):
        """
        Return a string with keys and/or values or types.
        *fmt* is a format string as used in the str.format() method.
        The str.format() method is called with key, value as positional arguments, and klen, vlen as kwargs.  The latter are the maxima
        of the string lengths for the keys and values, respectively,
        up to respective maxima of 20 and 40.
        """
        if fmt is None:
            fmt = self.str_fmt

        items = list(self.items())
        items.sort()

        klens = []
        vlens = []
        for i, (k, v) in enumerate(items):
            lenk = len(str(k))
            if types:
                v = type(v).__name__
            lenv = len(str(v))
            items[i] = (k, v)
            klens.append(lenk)
            vlens.append(lenv)

        klen = min(20, max(klens))
        vlen = min(40, max(vlens))
        slist = [fmt.format(k, v, klen=klen, vlen=vlen) for k, v in items]
        return ''.join(slist)

    def _from_pylines(self, pylines):
        # We can't simply exec the code directly, because in
        # Python 3 the scoping for list comprehensions would
        # lead to a NameError.  Wrapping the code in a function
        # fixes this.
        d = dict()
        lines = ["def _temp_func():\n"]
        lines.extend([f"    {line.rstrip()}\n" for line in pylines])
        lines.extend(["\n    return(locals())\n",
                      "_temp_out = _temp_func()\n",
                      "del(_temp_func)\n"])
        codetext = "".join(lines)
        code = compile(codetext, '<string>', 'exec')
        exec(code, globals(), d)
        self.update(d["_temp_out"])
        return self

    # If I were starting again, I would probably make the following two
    # functions class methods instead of instance methods, so they would
    # follow the factory pattern.  Too late now.

    def from_pyfile(self, filename):
        """
        Read in variables from a python code file.
        """
        with open(filename) as f:
            pylines = f.readlines()
        return self._from_pylines(pylines)

    def from_pystring(self, pystr):
        """
        Read in variables from a python code string.
        """
        pylines = pystr.split('\n')
        return self._from_pylines(pylines)

    def update_values(self, *args, **kw):
        """
        arguments are dictionary-like; if present, they act as
        additional sources of kwargs, with the actual kwargs
        taking precedence.

        One reserved optional kwarg is "strict".  If present and
        True, then any attempt to update with keys that are not
        already in the Bunch instance will raise a KeyError.
        """
        strict = kw.pop("strict", False)
        newkw = dict()
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = dict([(k, v) for (k, v) in newkw.items() if k in self])
        self.update(dsub)

    def update_None(self, *args, **kw):
        """
        Similar to update_values, except that an existing value
        will be updated only if it is None.
        """
        strict = kw.pop("strict", False)
        newkw = dict()
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = dict([(k, v) for (k, v) in newkw.items()
                                if k in self and self[k] is None])
        self.update(dsub)

    def _check_strict(self, strict, kw):
        if strict:
            bad = set(kw.keys()) - set(self.keys())
            if bad:
                bk = list(bad)
                bk.sort()
                ek = list(self.keys())
                ek.sort()
                raise KeyError(
                    "Update keys %s don't match existing keys %s" % (bk, ek))

def detrend(x, method='linear', axis=-1):
    if method == 'none':
        return x
    if method == 'mean' or method == 'linear':
        try:
            xm = x.mean(axis=axis, keepdims=True)
        except TypeError:
            # older version of numpy
            sh = list(x.shape)
            sh[axis] = 1
            xm = x.mean(axis=axis).reshape(sh)
        y = x - xm
        if method == 'mean':
            return y
        t = np.linspace(-1, 1, x.shape[axis])
        t /= np.sqrt((t ** 2).sum())
        yy = y.swapaxes(axis, -1)
        a = np.dot(yy, t)
        if a.ndim > 0:
            a = a[..., np.newaxis]
        yydetrend = yy - a * t
        return np.ascontiguousarray(yydetrend.swapaxes(-1, axis))
    else:
        raise ValueError("method %s is not recognized" % method)
 
_detrend = detrend # alias to avoid being clobbered by kwarg name                

def window_vals(n, name):
    _name = name
    name = name.lower()
    x = np.arange(n, dtype=float)
    if name == 'boxcar' or name == 'none':
        weights = np.ones((n,), dtype=float)
    elif name == 'triangle':
        weights = 1 - np.abs(x - 0.5 * n) / (0.5 * n)
    elif name == 'welch' or name == 'quadratic':
        weights = 1 - ((x - 0.5 * n) / (0.5 * n)) ** 2
    elif name == 'blackman':
        phi = 2 * np.pi * (x - n / 2) / n
        weights = 0.42 + 0.5 * np.cos(phi) + 0.08 * np.cos(2 * phi)
    elif name == 'hanning':
        phi = 2 * np.pi * x / n
        weights = 0.5 * (1 - np.cos(phi))
    elif name == 'cosine10':
        weights = _Tukey(n, 0.1)
     
    else:
        raise ValueError("name %s is not recognized" % _name)
     
    # Correct for floating point error for all except the boxcar.
    if weights[0] < 1:
        weights[0] = 0
     
    return weights

def _slice_tuple(sl, axis, ndim):
    freqsel = [slice(None)] * ndim
    tup = freqsel[:]
    tup[axis] = sl
    return tuple(tup)

def _welch_params(*args, **kw):
    kw = Bunch(kw)
      
    args = [np.array(x, copy=False) for x in np.broadcast_arrays(*args)]
    axis = kw.get('axis', -1)
    npts = args[0].shape[axis]
    
    noverlap = int(kw.overlap * kw.nfft)
    
    weights = window_vals(kw.nfft, kw.window)
    
    step = kw.nfft - noverlap
    ind = np.arange(0, npts - kw.nfft + 1, step)
      
    args += [weights, ind]
    return args


def spectrum(x, y=None, nfft=256, dt=1, detrend='linear',
               window='hanning', overlap=0.5, axis=-1,
               smooth=None):
    """
    Scripts from soest team.
    https://currents.soest.hawaii.edu/ocn_data_analysis/_static/Spectrum.html

    Spectrum and optional cross-spectrum for N-D arrays.

    Rotary spectra are calculated if the inputs are complex.

    detrend can be 'linear' (default), 'mean', 'none', or a function.

    window can be a window function taking nfft as its sole argument,
    or the string name of a numpy window function (e.g. hanning)

    overlap is the fractional overlap, e.g. 0.5 for 50% (default)

    smooth is None or an odd integer. It can be used instead of,
    or in addition to, segment averaging.  To use it exclusively,
    set nfft=None.

    Returns a Bunch with spectrum, frequencies, etc.  The variables
    in the output depend on whether the input is real or complex,
    and on whether an autospectrum or a cross spectrum is being
    calculated.

    """

    if smooth is not None and smooth % 2 != 1:
        raise ValueError("smooth parameter must be None or an odd integer")

    if nfft is None:
        nfft = x.shape[axis]
        nfft -= nfft % 2  # If it was odd, chop off the last point.

    if nfft % 2:
        raise ValueError("nfft must be an even integer")

    # Raw frequencies, negative and positive:
    freqs = fftshift(fftfreq(nfft, dt))  # cycles per unit time used for dt

    if smooth is None:
        n_end = 0
    else:
        n_end = (smooth - 1) // 2  # to be chopped from each end

    kw = dict(nfft=nfft,
              detrend=detrend,
              window=window,
              overlap=overlap,
              axis=axis)
    if y is None:
        x, weights, seg_starts = _welch_params(x, **kw)
    else:
        x, y, weights, seg_starts = _welch_params(x, y, **kw)

    is_complex = (x.dtype.kind == 'c' or y is not None and y.dtype.kind == 'c')

    nsegs = len(seg_starts)

    segshape = list(x.shape)
    segshape[axis] = nfft
    ashape = tuple([nsegs] + segshape)
    fx_k = np.zeros(ashape, np.complex_)
    if y is not None:
        fy_k = fx_k.copy()

    # Make an indexing tuple for the weights.
    bcast = [np.newaxis] * x.ndim
    bcast[axis] = slice(None)
    bcast = tuple(bcast)

    segsel = [slice(None)] * x.ndim

    # Iterate over the segments.  (There might be only one.)
    for i, istart in enumerate(seg_starts):
        indslice = slice(istart, istart + nfft)
        segsel[axis] = indslice
        xseg = x[tuple(segsel)]
        xseg = weights[bcast] * _detrend(xseg, method=detrend, axis=axis)
        fx = fft(xseg, n=nfft, axis=axis)
        fx_k[i] = fx
        if y is not None:
            yseg = y[tuple(segsel)]
            yseg = weights[bcast] * _detrend(yseg, method=detrend, axis=axis)
            fy = fft(yseg, n=nfft, axis=axis)
            fy_k[i] = fy

    fxx = fftshift((np.abs(fx_k) ** 2).mean(axis=0), axes=[axis])
    if y is not None:
        fyy = fftshift((np.abs(fy_k) ** 2).mean(axis=0), axes=[axis])
        fxy = fftshift((np.conj(fx_k) * fy_k).mean(axis=0), axes=[axis])

    # Negative frequencies, excluding Nyquist:
    sl_cw = slice(1, nfft // 2)
    cwtup = _slice_tuple(sl_cw, axis=axis, ndim=x.ndim)

    # Positive frequencies, excluding 0:
    sl_ccw = slice(1 + nfft // 2, None)
    ccwtup = _slice_tuple(sl_ccw, axis=axis, ndim=x.ndim)

    if smooth is not None:
        # start with a boxcar; we can make it more flexible later
        smweights = np.ones((smooth,), dtype=float)
        smweights /= smweights.sum()
        fxx[cwtup] = _convolve(fxx[cwtup], smweights, axis=axis)
        fxx[ccwtup] = _convolve(fxx[ccwtup], smweights, axis=axis)
        if y is not None:
            fyy[cwtup] = _convolve(fyy[cwtup], smweights, axis=axis)
            fyy[ccwtup] = _convolve(fyy[ccwtup], smweights, axis=axis)
            fxy[cwtup] = _convolve(fxy[cwtup], smweights, axis=axis)
            fxy[ccwtup] = _convolve(fxy[ccwtup], smweights, axis=axis)
        # Note: end effects with this algorithm consist of a bias
        # towards zero.  We will adjust the
        # slices to select only the unbiased portions.

    psdnorm = (dt / (weights ** 2).sum())
    psd = fxx * psdnorm
    ps = fxx *  (1.0 / (weights.sum() ** 2))
    if y is not None:
        psd_x = psd
        psd_y = fyy * psdnorm
        psd_xy = fxy * psdnorm
        cohsq = np.abs(fxy) ** 2 / (fxx * fyy)
        cohsq = np.abs(fxy) ** 2 / (np.real(fxx) * np.real(fyy))
        phase = np.angle(fxy)

    if smooth is not None:
        ## Adjust the slices to avoid smoothing end effects:
        # Negative frequencies, excluding Nyquist:
        sl_cw = slice(1 + n_end, nfft // 2 - n_end)
        cwtup = _slice_tuple(sl_cw, axis=axis, ndim=x.ndim)

        # Positive frequencies, excluding 0:
        sl_ccw = slice(1 + nfft // 2 + n_end, -n_end)
        ccwtup = _slice_tuple(sl_ccw, axis=axis, ndim=x.ndim)

    out = Bunch(freqs=freqs[sl_ccw],
                freq_boundaries=_boundaries(freqs[sl_ccw]),
                seg_starts=seg_starts,
                smooth=smooth,
                nfft=nfft,
                detrend=detrend,
                window=window,
                overlap=overlap,
                axis=axis,
                dt = dt
                )

    if not is_complex:
        if y is None:
            out.psd = psd[ccwtup] * 2
            out.ps = ps[ccwtup] * 2
            out.fxx = fxx[ccwtup]
        else:
            out.psd_x = psd_x[ccwtup]
            out.psd_y = psd_y[ccwtup]
            out.psd_xy = psd_xy[ccwtup]
            out.fxx = fxx[ccwtup]
            out.fyy = fyy[ccwtup]
            out.fxy = fxy[ccwtup]
            out.cohsq = cohsq[ccwtup]
            out.phase = phase[ccwtup]

    else:
        out.cwfreqs = -freqs[sl_cw]
        out.cwfreq_boundaries = _boundaries(out.cwfreqs)
        out.ccwfreqs = freqs[sl_ccw]
        out.ccwfreq_boundaries = _boundaries(out.ccwfreqs)

        if y is None:
            out.cwpsd = psd[cwtup]
            out.cwps = ps[cwtup]
            out.ccwpsd = psd[ccwtup]
            out.ccwps = ps[ccwtup]
        else:
            out.cwpsd_x = psd_x[cwtup]
            out.cwpsd_y = psd_y[cwtup]
            out.cwpsd_xy = psd_xy[cwtup]
            out.cwcohsq = cohsq[cwtup]
            out.cwphase = phase[cwtup]

            out.ccwpsd_x = psd_x[ccwtup]
            out.ccwpsd_y = psd_y[ccwtup]
            out.ccwpsd_xy = psd_xy[ccwtup]
            out.ccwcohsq = cohsq[ccwtup]
            out.ccwphase = phase[ccwtup]

    return out

def phase_diff_time(spec,
                    plot=True,
                    x1=None,
                    x2=None,
                    coherence_level=0.95,
                    figsize=(8,8)):
    """
    Calculate phase_diff_time by spectrum result.
    Parameters:
        spec: result of spectrum function
        plot: plot corresponding result figure if True
       x1,x2: data sequences needed for plot
      coherence_level: plot a 95% coherence line in the plot
    """
    weight_cohsq = spec.cohsq**2/(1-spec.cohsq**2)
    W = np.zeros((len(spec.freqs),len(spec.freqs)))
    lenx = len(spec.freqs)
    G = spec.freqs.reshape(lenx,1)
    for i in range(lenx):
        W[i,i] = weight_cohsq[i]
    dobs = spec.phase.reshape(lenx,1)
    dobs = np.rad2deg(dobs)
    WG = np.matmul(W,G)
    Wdobs = np.matmul(W,dobs)
    mest,residuals,rank,sigulars = np.linalg.lstsq(WG,Wdobs,rcond=None)
    deltaT = mest[0][0]*100/360*10   #100Hz/360deg*10ms
    
    if plot:   
        fig,axs = plt.subplots(2,2,figsize=figsize)
        axs = axs.ravel()
        
        # axs[0]
        line1, = axs[0].plot(np.arange(0,len(x1))*spec.dt,x1)
        line2, = axs[0].plot(np.arange(0,len(x2))*spec.dt,x2)
        axs[0].set_xlabel("Time/s")
        axs[0].legend([line1,line2],["trace_1","trace_2"])
        axs[0].set_xlim([0,len(x1)*spec.dt])

        # axs[1]
        line1, = axs[1].plot(spec.freqs,spec.fxx,label="trace_1");
        line2, = axs[1].plot(spec.freqs,spec.fyy,label="trace_2");
        axs[1].legend([line1,line2],["trace_1","trace_2"])
        axs[1].set_xlabel("Frequency/Hz")
        axs[1].set_ylabel("|$S(f)$|")
        axs[1].set_xlim([spec.freqs[0],spec.freqs[-1]])

        # axs[2]
        axs[2].plot(spec.freqs,spec.cohsq)
        axs[2].set_xlabel("Frequency/Hz")
        axs[2].set_ylabel("Coherence")
        axs[2].set_xlim([spec.freqs[0],spec.freqs[-1]])

        # axs[3]
        deg_ms5_pos = spec.freqs*1*spec.dt/2*360
        axs[3].plot(spec.freqs,np.rad2deg(spec.phase))
        line3, = axs[3].plot(spec.freqs,mest[0][0]*spec.freqs,'--')

        line1, = axs[3].plot(spec.freqs,deg_ms5_pos)
        line2, = axs[3].plot(spec.freqs,-deg_ms5_pos)
        axs[3].legend([line1,line2,line3],
                      [f'Reference: +{format(1.0*spec.dt/2*1000,".2f")} ms',\
                       f'Reference: -{format(1.0*spec.dt/2*1000,".2f")} ms',\
                       f'Estimated: {format(deltaT,".2f")} ms'])
        axs[3].set_xlabel("Frequency/Hz")
        axs[3].set_ylabel("Phase/degree")
        axs[3].set_xlim([spec.freqs[0],spec.freqs[-1]])
        plt.tight_layout()
    
    return deltaT
