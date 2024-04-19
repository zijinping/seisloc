from seisloc.math import lsfit
import numpy as np
from numba import jit,prange
from numpy.linalg import det
from obspy.geodetics import gps2dist_azimuth

@jit(nopython=True,fastmath=True)
def grid_search(G:np.ndarray,dts:np.ndarray,n2x:int,dx:float,n2y:int,dy:float,n2z:int,dz:float,W=1,secondRefine:bool=False)->tuple:
    """
    n2x: number of grid points in the positive x direction
    """
    nx = 2*n2x+1
    ny = 2*n2y+1
    nz = 2*n2z+1
    xs = np.arange(-n2x,n2x+1,1)*dx
    ys = np.arange(-n2y,n2y+1,1)*dy
    zs = np.arange(-n2z,n2z+1,1)*dz
    resMatrix = np.zeros((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for z in range(nz):
                m = np.array([[xs[i]],[ys[j]],[zs[z]]])
                ets = G@m
                resMatrix[i,j,z] = np.std((ets-dts)*W)
    minRes = np.min(np.abs(resMatrix))
    k = np.where(resMatrix==minRes)

    # Initial best fit
    x0 = xs[k[0][0]]
    y0 = ys[k[1][0]]
    z0 = zs[k[2][0]]
    # check whether local minium
    if x0==xs[0] or x0==xs[-1] or\
       y0==ys[0] or y0==ys[-1] or\
       z0==zs[0] or z0==zs[-1]:
       locMin = 0
    else:
       locMin = 1

    #-------- further refine -------------------
    if secondRefine:
        xs1 = xs/10
        ys1 = ys/10
        zs1 = zs/10
        
        resMatrix = np.zeros((nx,ny,nz))
        for i in range(nx):
            for j in range(ny):
                for z in range(nz):
                    m = np.array([[x0+xs1[i]],[y0+ys1[j]],[z0+zs1[z]]])
                    ets = G@m
                    resMatrix[i,j,z] = np.std((ets-dts)*W)
        
        minRes = np.min(np.abs(resMatrix))
        k = np.where(resMatrix == minRes)

        x0 = x0+xs1[k[0][0]]
        y0 = y0+ys1[k[1][0]]
        z0 = z0+zs1[k[2][0]]

    return x0,y0,z0,locMin

def locate_slave2(masLon,masLat,masDep,masVels,
                 phaseRecs,model,stas,stepDict,slaveId=-1,secondRefine=False):
    """
    Relative location of the template matching detected events with respect to 
    the template event (master event) using the grid-search method.

    Parameters
    | masLon,masLat,masDep: master event parameters
    |   masVels: P ans S velocties of the master event position
    | phaseRecs: Records (sta,dt,weight,phsType) of phase relationships between 
                 the slave event and the master event. dt = t(sla.) - t(mas.)
    |     model: Obspy taup model for the ray-tracing
    |  stepDict: Information regarding steps. E.g., {"x":[5,0.1];"y":[5,0.1],"z":{5,0.1}}
                 stands for search along x,y,z for -0.5 to 0.5 km with step length of 0.1 km.
    |      stas: seisloc.sta Sta() class that containing stations information
    |   slaveId: This parameter is designed for multiprocessing and will be 
                 returned as a marker of the multiprocessing result

    [[dtdx1,dtdy1,dtdz1],
     [dtdx2,dtdy2,dtdz2],
     [dtdx3,dtdy3,dtdz3],
     ...
     dtdxn,dtdyn,dtdzn]] @ [[ix],[iy],[iz]] = [[et1],[et2],[et3],...,[etn]] (et, estimated time)

    best location is the np.min(np.std([[et1],[et2],[et3],...,[etn]]-[[dt1],[dt2],[dt3],...,[dtn]]))

    """
    G = [] # data kernel 
    d = [] # observation 
    W = [] # weights
    # ----- set up kernel -----------
    for i,phaseRec in enumerate(phaseRecs):
        sta,dt,weight,phsType = phaseRec
        staLon = stas.dict["SC"][sta][0]
        staLat = stas.dict["SC"][sta][1]
        dist,az,baz = gps2dist_azimuth(masLat,masLon,staLat,staLon)
        distKm = dist/1000
        distDeg = distKm/111.19
        if phsType.lower() == "p":
            masVel = masVels[0]
            weight = 0.5
        if phsType.lower() == "s":
            masVel = masVels[1]
            weight = 1
        phase_list = [phsType.lower(),phsType.upper()]
        arrivals = model.get_ray_paths(source_depth_in_km=masDep,
                                       distance_in_degree=distDeg,
                                       phase_list = phase_list)
        

        if len(arrivals)>0:

            takeAngle = arrivals[0].takeoff_angle
            phkm = np.sin(np.deg2rad(180-takeAngle))/masVel
            dtdz = -phkm / np.tan(np.deg2rad(takeAngle))
            dtdx = -phkm * np.sin(np.deg2rad(az))
            dtdy = -phkm * np.cos(np.deg2rad(az))

            G.append([dtdx,dtdy,dtdz])
            d.append([dt])
            W.append(weight)
    G = np.array(G)
    d = np.array(d).reshape((len(d),1))
    W = np.array(W).reshape((len(W),1))
    
    xnode=stepDict["x"][0]
    xstep=stepDict["x"][1] # -xnode*xstep, ...,0,...,xnode*step
    ynode=stepDict["y"][0]
    ystep=stepDict["y"][1] # -xnode*xstep, ...,0,...,xnode*step
    znode=stepDict["z"][0]
    zstep=stepDict["z"][1] # -xnode*xstep, ...,0,...,xnode*step
    dxkm,dykm,dzkm,bdyStats = grid_search(G,d,xnode,xstep,ynode,ystep,znode,zstep,W=W,secondRefine=secondRefine)

    dlon = dxkm/(111.19*np.cos(np.deg2rad(masLat)))
    dlat = dykm/111.19

    return masLon+dlon,masLat+dlat,masDep+dzkm,bdyStats,slaveId

def locate_slave(masLon,masLat,masDep,masVels,
                 phaseRecs,model,stas,slaveId=-1):
    """
    Relative location of the template matching detected events with respect to 
    the template event (master event)

    Parameters
    | masLon,masLat,masDep: master event parameters
    |   masVels: P ans S velocties of the master event position
    | phaseRecs: Records (sta,dt,weight,phsType) of phase relationships between 
                 the slave event and the master event. dt = t(sla.) - t(mas.)
    |     model: Obspy taup model for the ray-tracing
    |      stas: seisloc.sta Sta() class that containing stations information
    |   slaveId: This parameter is designed for multiprocessing and will be 
                 returned as a marker of the multiprocessing result

    """
    G = [] # data kernel 
    d = [] # observation 
    W = [] # weights
    # ----- set up kernel -----------
    for i,phaseRec in enumerate(phaseRecs):
        sta,dt,weight,phsType = phaseRec
        staLon = stas.dict["SC"][sta][0]
        staLat = stas.dict["SC"][sta][1]
        dist,az,baz = gps2dist_azimuth(masLat,masLon,staLat,staLon)
        distKm = dist/1000
        distDeg = distKm/111.19
        if phsType.lower() == "p":
            masVel = masVels[0]
        if phsType.lower() == "s":
            masVel = masVels[1]
        phase_list = [phsType.lower(),phsType.upper()]
        arrivals = model.get_ray_paths(source_depth_in_km=masDep,
                                       distance_in_degree=distDeg,
                                       phase_list = phase_list)
        if len(arrivals)>0:
            takeAngle = arrivals[0].takeoff_angle
            phkm = np.sin(np.deg2rad(180-takeAngle))/masVel
            dtdz = -phkm / np.tan(np.deg2rad(takeAngle))
            dtdx = -phkm * np.sin(np.deg2rad(az))
            dtdy = -phkm * np.cos(np.deg2rad(az))
            G.append([dtdx,dtdy,dtdz])
            d.append([dt])
            W.append(weight)
    G = np.array(G)
    # ----- Normalize G columns to enhance the stability of inversion -----
    norm0 = np.linalg.norm(G[:,0])
    norm1 = np.linalg.norm(G[:,1])
    norm2 = np.linalg.norm(G[:,2])
    G[:,0] = G[:,0]/norm0
    G[:,1] = G[:,1]/norm1
    G[:,2] = G[:,2]/norm2
    d = np.array(d)
    W = np.diag(W)
    WG = np.matmul(W,G)
    GTWG = np.matmul(G.T,WG)
    if len(G)<3 or det(GTWG) == 0: # No enough observation or singular matrix
        m = np.zeros((3,1))
    else:
        m = lsfit(G,d,W)
    # ----- G@m=d --> G/norm @ norm*m = d -----
    dxkm = m.ravel()[0]/norm0
    dykm = m.ravel()[1]/norm1
    dzkm = m.ravel()[2]/norm2
    dlon = dxkm/(111.19*np.cos(np.deg2rad(masLat)))
    dlat = dykm/111.19
                            
    return masLon+dlon,masLat+dlat,masDep+dzkm,slaveId
