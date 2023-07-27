import numpy as np
from obspy import Stream, Trace

def lsfit(G,d,W=1,full=False):
    """
    Get the least square fit.
    | G: data kernel
    | d: observations
    | W: weighting matrix
    To be addded: return covariance matrix
    """
    assert len(G.shape)==2
    rows = G.shape[0]
    cols = G.shape[1]
    #-------- set up weight matrix --------
    if isinstance(W,int):
        _W = np.ones(rows)*W
        W = np.diag(_W)
    elif isinstance(W,list) or (isinstance(W,np.ndarray) and len(W.shape)==1):
        W = np.diag(W)
    elif len(W.shape)==2 and W.shape[0] == G.shape[0]:
        pass
    else:
        raise Expception("Wrong W set up!!!")
    WG = np.matmul(W,G)
    GTWG = np.matmul(G.T,WG)
    GTW = np.matmul(G.T,W)
    GTWd = np.matmul(GTW,d)
    GTWG_inv = np.linalg.inv(GTWG)
    m = np.matmul(GTWG_inv,GTWd)

    dpre = np.matmul(WG,m)
    e = dpre - d
    E = np.matmul(e.T,e)
    freedomDeg=len(d) - len(m)
    if freedomDeg == 0: freedomDeg = 1
    sigmad2 = E/freedomDeg

    GTWG_invGTW = np.matmul(GTWG_inv,GTW)

    u,s,vh = np.linalg.svd(GTWG_invGTW)

    s2 = np.matmul(np.diag(s),np.diag(s))
    us2 = np.matmul(u,s2)
    us2uT = np.matmul(us2,u.T)

    if full:
        return m, sigmad2,us2uT*sigmad2
    return m

def weighted_lsfit(G,dobs,W=1,criteria=0.01):
    """ 
    Residual variation criteria for output results: default 0.01(1%)
    """
    if isinstance(W,int):
        W = np.eye(G.shape[0],G.shape[0])
    if isinstance(W,np.ndarray) and len(W.shape)==1 and len(W)==len(G):
        W = np.diag(W)
    assert len(W.shape) == 2 and W.shape[0] == W.shape[1]
    
    mnew,sigmad2,covar = lsfit(G,dobs,W,full=True)
    dpre = np.matmul(G,mnew)
    dres = np.abs(dobs-dpre)
    dresReci = 1/dres       # Reciprocal
    midVal = np.median(dresReci)
    ks = np.where(dresReci>3*midVal)
    dresReci[ks] = 3*midVal        
    Wres = np.diag(dresReci)
 
    while True:
        mold = mnew
        Wnew = W*Wres
        dresWt = np.abs(Wnew@dres)
        mnew,sigmad2New,covarNew = lsfit(G,dobs,Wnew,full=True)
        dres = np.abs(dobs-np.matmul(G,mnew))
        dresReci = 1/dres       # Reciprocal
        midVal = np.median(dresReci)
        ks = np.where(dresReci>3*midVal)
        dresReci[ks] = 3*midVal        
        Wres = np.diag(dresReci)           
              
        mdiff = np.abs(mold-mnew)
        mdiffNorm = np.linalg.norm(mdiff)
        if mdiffNorm < 1E-6:
            break
    return mnew

def gaussian_filter(kernel_size, sigma=4, muu=0):
            
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
            
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
            
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)
            
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    return gauss
            
def avg_filter(kernel_size,dimension=2):
    if dimension==2:
        matrix = np.ones((kernel_size,kernel_size))
    elif dimension==1:
        matrix = np.ones(kernel_size)
    else:
        raise Exception(f"Unaccepted dimension value: {dimension}")
    matrix = matrix/np.sum(matrix)
    return matrix

def mean_period_taup(st,alpha=0.99):
    """
    Calculate mean period of event waveform
    """
    stTaup = st.copy()
    print(len(stTaup))
    for j,trvel in enumerate(st):
        delta = trvel.stats.delta
        tracc = trvel.copy()
        tracc.data[:-1] = (trvel.data[1:]-trvel.data[:-1])/delta
        tracc.data[-1] = 0
        Xi=0
        Di=0
        tauPs = []
        for i in range(len(trvel.data)):
            Xi=alpha*Xi+trvel.data[i]**2
            Di=alpha*Di+tracc.data[i]**2
            tauPi = 2*np.pi*np.sqrt(Xi/Di)
            tauPs.append(tauPi)
        stTaup[j].data = np.array(tauPs)
    
    return stTaup
