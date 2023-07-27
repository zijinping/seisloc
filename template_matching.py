from seisloc.math import lsfit
import numpy as np
from obspy.geodetics import gps2dist_azimuth

def locate_slave(masLon,masLat,masDep,masVels,phaseRecs,model,stas):
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
            dtdz = -phkm * np.tan(np.deg2rad(takeAngle))
            dtdx = -phkm * np.sin(np.deg2rad(az))
            dtdy = -phkm * np.cos(np.deg2rad(az))
            G.append([dtdx,dtdy,dtdz])
            d.append([dt])
            W.append(weight)
    G = np.array(G)
    norm0 = np.linalg.norm(G[:,0])
    norm1 = np.linalg.norm(G[:,1])
    norm2 = np.linalg.norm(G[:,2])
    G[:,0] = G[:,0]/norm0
    G[:,1] = G[:,1]/norm1
    G[:,2] = G[:,2]/norm2

    d = np.array(d)  
    if len(G)<3:       # No enough observation     
        m = np.zeros((3,1))
    else:                   
        m = lsfit(G,d,W)                        
    dxkm = m.ravel()[0]/norm0
    dykm = m.ravel()[1]/norm1
    dzkm = m.ravel()[2]/norm2
    dlon = dxkm/(111.19*np.cos(np.deg2rad(masLat)))
    dlat = dykm/111.19
                            
    return masLon+dlon,masLat+dlat,masDep+dzkm
