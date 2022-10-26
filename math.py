def lsFit(G,d,W=1):
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
        _W == np.ones(rows)*W
        W = np.diag(W)
    WG = np.matmul(W,G)
    GTWG = np.matmul(G.T,WG)
    GTW = np.matmul(G.T,W)
    GTWd = np.matmul(GTW,d)
    GTWG_inv = np.linalg.inv(GTWG)
    m = np.matmul(GTWG_inv,GTWd)
    
    return m
