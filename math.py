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

def weighted_lsfit(G,dobs,W=1,wtMode='log10',wtReciStdMax=5,criteria=0.01,debug=False):
    """ 
    Conduct iterative lsfit until condition fulfilled.
    wtMode: mode for re-weight. 'log10': wt=log10(abs(reciprocal(res))/min(abs(reciprocal)))
    criteria: If ratio of change of m is smaller than the criteria, the iteration will stop
    """
    if isinstance(W,int):
        W = np.eye(G.shape[0],G.shape[0])
    if isinstance(W,np.ndarray) and len(W.shape)==1 and len(W)==len(G):
        W = np.diag(W)
    assert len(W.shape) == 2 and W.shape[0] == W.shape[1]
    
    mnew,sigmad2,covar = lsfit(G,dobs,W,full=True)
    dpre = np.matmul(G,mnew)     # prediction
    dres = np.abs(dobs-dpre)     # residual
    dresReci = 1/dres            # Reciprocal
    if wtMode == "log10":
        Wres = np.diag(W*np.log10(dresReci/np.min(dresReci)))
    count = 1
    if debug: print("Initial weighted residual: ",np.linalg.norm(Wres*dres))
    while True:
        mold = mnew
        mnew,sigmad2New,covarNew = lsfit(G,dobs,Wres,full=True)
        dres = np.abs(dobs-np.matmul(G,mnew))
        if debug:
            print("max/min/norm residual: ",np.max(dres),np.min(dres),np.linalg.norm(dres*Wres))
        dresReci = 1/dres       # Reciprocal
        #>>>>> weighting function >>>>>
        Wres = np.diag(W*np.log10(dresReci/np.min(dresReci)))
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              
        mdiff = np.abs(mold-mnew)
        moldNorm = np.linalg.norm(mold)
        mdiffNorm = np.linalg.norm(mdiff)
        count += 1
        if mdiffNorm/moldNorm < criteria:
            if debug:
                print(f"Iteration for {count} time timess")
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

def interception_two_lines(line1,line2,linspace=100):
    """
    Ouptut the cloest intercepted point on the stright line one w.r.t the line two.
    line1 format: [[x0,y0,...,z0],[x1,y1,...,z1]]
    """
    if not isinstance(line1,np.ndarray):
        line1 = np.array(line1)
    if not isinstance(line2,np.ndarray):
        line2 = np.array(line2)
    assert len(line1.shape) == 2
    assert len(line1.shape) == 2
    lines = np.linspace(line1[0,:],line1[1,:],linspace)

    norms = []
    for i in range(len(lines)):
        A = line2[0,:] - lines[i]
        B = line2[1,:] - lines[i]
        crossAB = np.cross(A,B)
        normAB = np.linalg.norm(crossAB)
        norms.append(normAB)
    k = np.argmin(norms)
    return lines[k]
