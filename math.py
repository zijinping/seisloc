import numpy as np

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
    if isinstance(W,list):
        W = np.diag(W)
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
