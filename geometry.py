import os
import numpy as np
from scipy.interpolate import griddata
from math import sin,cos,asin,acos,pi,radians
from numba import jit
import logging
from obspy.geodetics import gps2dist_azimuth
import subprocess

def spherical_dist(lon_1,lat_1,lon_2,lat_2):
    """
    Calculate the distance of two postions and return distance in degree
    """
    lon_1 = radians(lon_1)
    lat_1 = radians(lat_1)
    lon_2 = radians(lon_2)
    lat_2 = radians(lat_2)
    a=acos(sin(lat_1)*sin(lat_2)+cos(lat_1)*cos(lat_2)*cos(lon_2-lon_1))
    return a*180/pi

@jit(nopython=True)
def in_rectangle(locs,alon,alat,blon,blat,width,mode='normal'):
    """
    Judge a list of locations in within a certain distance of line a-b or not
    mode: "normal" for Cartesian coordinate; "geo" for geographic coordinate
    """
    #----------quality control---------------------
    assert len(locs.shape) == 2
    assert mode in ["normal","geo"]

    results = np.zeros(locs.shape)
    dlon1 = blon - alon                 # start point and end point
    dlat1 = blat - alat
    mlon1 = (blon+alon)/2
    mlat1 = (blat+alat)/2
    rad_alon = radians(alon)            # convert to radians
    rad_alat = radians(alat)
    norm1 = (dlon1**2+dlat1**2)**0.5
#    if mode == "geo":
#        norm1 = ((dlon1*np.cos(np.deg2rad((mlat1)))**2+dlat1**2)**0.5
    for i in range(locs.shape[0]):
        ilon = locs[i,0]
        ilat = locs[i,1]
        dlon2 = ilon - alon
        dlat2 = ilat - alat
        mlon2 = (ilon+alon)/2
        mlat2 = (ilat+alat)/2
        norm2 = (dlon2**2+dlat2**2)**0.5
#    if mode == "geo":
#        norm2 = ((dlon2*np.cos(np.deg2rad((mlat2)))**2+dlat2**2)**0.5
        if norm2 == 0:
            results[i,0]=1
            continue
        proj_ratio = (dlon1*dlon2+dlat1*dlat2)/norm1**2

        rad_jlon = radians(alon+proj_ratio*dlon1)
        rad_jlat = radians(alat+proj_ratio*dlat1)
        rad_a=acos(sin(rad_alat)*sin(rad_jlat)+cos(rad_alat)*cos(rad_jlat)*cos(rad_alon-rad_jlon))
        proj_length = rad_a*180/pi*111.1
        sin_value = np.abs(dlon1*dlat2-dlon2*dlat1)/(norm1*norm2)
        vdist = sin_value*norm2
        if proj_ratio>=0 and proj_ratio<=1 and vdist<=width:
            results[i,0]=1
            results[i,1]=proj_length
    return results

def in_ellipse(xy_list,width,height,angle=0,xy=[0,0]):
    """
    Find data points inside an ellipse and return index list

    Parameters:
        xy_list: Points needs to be deteced.
        width: Width of the ellipse
        height: Height of the ellipse
        angle: anti-clockwise rotation angle in degrees
        xy: the origin of the ellipse
    """
    if isinstance(xy_list,list):
        xy_list = np.array(xy_list)
    if not isinstance(xy_list,np.ndarray):
        raise Exception(f"Unrecoginzed data type: {type(xy_list)}, \
                          should be list or np.ndarray")
    new_xy_list = xy_list.copy()
    new_xy_list = new_xy_list - xy

    #------------ define coordinate conversion matrix----------
    theta = angle/180*np.pi         # degree to radians
    con_mat = np.zeros((2,2))
    con_mat[:,0] = [np.cos(theta),np.sin(theta)]
    con_mat[:,1] = [np.sin(theta),-np.cos(theta)]

    tmp = np.matmul(con_mat,new_xy_list.T)
    con_xy_list = tmp.T

    #------------ check one by one ----------------------------
    idxs = []
    for i,[x,y] in enumerate(con_xy_list):
        if ((x/(width/2))**2+(y/(height/2))**2) < 1:
            idxs.append(i)
        
    return idxs

def loc_by_width(lon1,lat1,lon2,lat2,width,direction='right'):
    """
    Calculate the points of a rectangle with width and two tips provided.

    Parameters:
      lon1,lat1: longitude and latitude of tip 1
      lon2,lat2: longitude and latitude of tip 2
      direction: The side of new points from tip 1 to tip2 direction
    """
    dlon = (lon2 - lon1) * np.cos(np.deg2rad(lat1))
    dlat = lat2 - lat1
    dist=(dlon**2+dlat**2)**0.5
    
    if direction == "right":  # extend width to the right 
    	delta_lat = -width*(dlon/dist)    # cos_theta
    	delta_lon =  width*(dlat/dist)    # sin_theta

    if direction == "left":
    	delta_lat =  width*(dlon/dist)    # cos_theta
    	delta_lon = -width*(dlat/dist)    # sin_theta

    new_lon1 = lon1 + delta_lon
    new_lat1 = lat1 + delta_lat
    new_lon2 = lon2 + delta_lon
    new_lat2 = lat2 + delta_lat

    return new_lon1,new_lat1,new_lon2,new_lat2

def seismic_path_calculation(e_lon,e_lat,e_dep,in_angle,vel_set):
    """
    Parameters:
        e_lon,e_lat,e_dep: the earthquake longitude, latitude and depth
        sta_lon,sta_lat: the station location
        vel_set: array format [[dep1,vel1],[dep2,vel2]], where vel1 indicates the velocity between dep1 and dep2
    """
    # initiation
    trace_points = []  
    trace_points.append([0,e_dep]) # store the location of trace points
    # First need to know which layer the event depth belongs to
    if not isinstance(vel_set,np.ndarray):
        vel_set = np.array(vel_set)
        
    # Find the corresponding layer source belongs to
    idx = -1
    for i in range(vel_set.shape[0]-1):
        if vel_set[i,0] <= e_dep and vel_set[i+1,0] > e_dep:
            idx = i
    if idx == -1: # not assigned in above loop, then it is below the last depth
            idx = vel_set.shape[0]-1
    
    # calculate p value
    v_start = vel_set[idx,1]
    p = np.sin(np.pi*in_angle/180)/v_start
    
    # calculate the total T(time in seconds) and X(horizontal distance in meters)
    T_sum = 0
    X_sum = 0
    tmp_gap = e_dep - vel_set[idx,0]
    tmp_T = tmp_gap/np.cos(np.pi*in_angle/180)/vel_set[idx,1]
    tmp_X = tmp_gap*np.tan(np.pi*in_angle/180)
    T_sum += tmp_T
    X_sum += tmp_X
    trace_points.append([trace_points[-1][0]+tmp_X,trace_points[-1][1]-tmp_gap])
    for i in range(idx):
        tmp_angle = np.arcsin(vel_set[idx-1-i,1]*p)*180/np.pi
        print(tmp_angle)
        tmp_gap = vel_set[idx-i,0]-vel_set[idx-1-i,0]
        tmp_T = tmp_gap/np.cos(np.pi*tmp_angle/180)/vel_set[idx-1-i,1]
        tmp_X = tmp_gap*np.tan(np.pi*tmp_angle/180)
        T_sum += tmp_T
        X_sum += tmp_X
        trace_points.append([trace_points[-1][0]+tmp_X,trace_points[-1][1]-tmp_gap])
    print("T_sum, X_sum: ",T_sum,X_sum)
    print(trace_points)
    return np.array(trace_points)

def loc_by_width_sphe(alon,alat,blon,blat,width,direction='left'):
    """
    Calculate the points of a rectangle with width and two tips provided.
    a,b,N,aa,bb is the start point,end point, north point, calculated aa and bb
    Parameters:
      alon,alat: longitude and latitude of tip a
      blon,blat: longitude and latitude of tip b
          width: width in degree
      direction: The side of new points from tip a to tip b direction
    """
    sphe_dist = spherical_dist(alon,alat,blon,blat)
    alon,alat,blon,blat,sphe_dist,width=list(map(radians,[alon,alat,blon,blat,sphe_dist,width]))
    dlon = blon - alon
    dlat = blat - alat
    if dlon>0 and dlat<0: # get angle loc1-loc2-N pole
        if direction=="left":
            abN = asin(sin(pi/2-alat)*sin(dlon)/sin(sphe_dist))
            Nbbb = pi/2-abN
            bN = pi/2-blat
            Nbb = acos(cos(bN)*cos(width)+sin(bN)*sin(width)*cos(Nbbb))
            bblat = pi/2-Nbb
            bNbb=asin(sin(width)*sin(Nbbb)/sin(Nbb))
            bblon=(blon+bNbb)
        elif direction=="right":
            abN = asin(sin(pi/2-alat)*sin(dlon)/sin(sphe_dist))
            Nbbb = pi/2+abN
            bN = pi/2-blat
            Nbb = acos(cos(bN)*cos(width)+sin(bN)*sin(width)*cos(Nbbb))
            bblat = pi/2-Nbb
            bNbb=asin(sin(width)*sin(Nbbb)/sin(Nbb))
            bblon=(blon-bNbb)
            
    elif dlon>0 and dlat>0:
        if direction=="left":
            abN = asin(sin(pi/2-alat)*sin(dlon)/sin(sphe_dist))
            Nbbb = abN
            bN = pi/2-blat
            Nbb = acos(cos(bN)*cos(width)+sin(bN)*sin(width)*cos(Nbbb))
            bblat = pi/2-Nbb
            bNbb=asin(sin(width)*sin(Nbbb)/sin(Nbb))
            bblon=(blon-bNbb)
        elif direction=="right":
            abN = asin(sin(pi/2-alat)*sin(dlon)/sin(sphe_dist))
            Nbbb = pi-abN
            bN = pi/2-blat
            Nbb = acos(cos(bN)*cos(width)+sin(bN)*sin(width)*cos(Nbbb))
            bblat = pi/2-Nbb
            bNbb=asin(sin(width)*sin(Nbbb)/sin(Nbb))
            bblon=(blon+bNbb)
    elif dlon<0 and dlat>0:
        if direction=="left":
            abN = asin(sin(pi/2-alat)*sin(dlon)/sin(sphe_dist))
            Nbbb = abN
            bN = pi/2-blat
            Nbb = acos(cos(bN)*cos(width)+sin(bN)*sin(width)*cos(Nbbb))
            bblat = pi/2-Nbb
            bNbb=asin(sin(width)*sin(Nbbb)/sin(Nbb))
            bblon=(blon-bNbb)
        elif direction=="right":
            abN = asin(sin(pi/2-alat)*sin(dlon)/sin(sphe_dist))
            Nbbb = pi-abN
            bN = pi/2-blat
            Nbb = acos(cos(bN)*cos(width)+sin(bN)*sin(width)*cos(Nbbb))
            bblat = pi/2-Nbb
            bNbb=asin(sin(width)*sin(Nbbb)/sin(Nbb))
            bblon=(blon+bNbb)
    elif dlon<0 and dlat>0:
        if direction=="right":
            abN = asin(sin(pi/2-alat)*sin(dlon)/sin(sphe_dist))
            Nbbb = abN
            bN = pi/2-blat
            Nbb = acos(cos(bN)*cos(width)+sin(bN)*sin(width)*cos(Nbbb))
            bblat = pi/2-Nbb
            bNbb=asin(sin(width)*sin(Nbbb)/sin(Nbb))
            bblon=(blon-bNbb)
        elif direction=="left":
            abN = asin(sin(pi/2-alat)*sin(dlon)/sin(sphe_dist))
            Nbbb = abN
            bN = pi/2-blat
            Nbb = acos(cos(bN)*cos(width)+sin(bN)*sin(width)*cos(Nbbb))
            bblat = pi/2-Nbb
            bNbb=asin(sin(width)*sin(Nbbb)/sin(Nbb))
            bblon=(blon+bNbb)
    elif dlon == 0:
        if direction=='right':
            bblon = blon+dlat/np.abs(dlat)*width
        elif direction=="left":
            bblon = blon-dlat/np.abs(dlat)*width
        bblat = blat  
    elif dlat == 0:
        if direction=='right':
            bblat = blat-dlon/np.abs(dlon)*width
        elif direction=="left":
            bblat = blat+dlon/np.abs(dlon)*width
        bblon = blon   
    elif dlat==0 and dlon==0:
        raise Error("Point a and b shouldn't have the same location")
    return bblon*180/pi,bblat*180/pi

def cartesian_rotate(xy,center=[0,0],degree=0):
    """
    Degree is positive for anticlockwise
    """
    if isinstance(xy,list):
        xy = np.array(xy)
    if len(xy.shape)==1:
        raise Exception("xy should be 2 dimensional matrix")
    if isinstance(center,list):
        center = np.array(center)
    logging.info("Now in function cartesian_rotate")
    logging.info(f"Parameters: center({center[0]},{center[1]}) rotate({degree} degree)")

    xy_ref = xy - center

    rotate_matrix = [[np.cos(degree/180*pi),-np.sin(degree/180*pi)],[np.sin(degree/180*pi),np.cos(degree/180*pi)]]
    rotate_matrix = np.array(rotate_matrix)

    xy_rotate = np.matmul(rotate_matrix,xy_ref.T).T + center
    
    return xy_rotate

def event_rotate(inFile,center,deg):
    """
    Rotate event input file for tomoDD
    """
    outFile = inFile+".rot"
    cont = []
    xys = []
    with open(inFile,'r') as f:
        for line in f:
            line = line.rstrip()
            cont.append(line)
            _lat,_lon = line[20:39].split()
            lat = float(_lat)
            lon = float(_lon)
            xys.append([lon,lat])
    xys_rot = spherical_rotate(xys,center=center,degree=deg)

    f = open(outFile,'w')
    for i in range(len(cont)):
        line = cont[i]
        f.write(line[:20])
        f.write(format(xys_rot[i,1],">8.4f"))
        f.write("  ")
        f.write(format(xys_rot[i,0],">9.4f"))
        f.write(line[39:])
        f.write("\n")

def sta_rotate(inFile,center,deg):
    """
    Rotate station file for tomoDD
    """
    outFile = inFile+".rot"
    xys = []
    stas = []
    lonlats = []
    _eles = []
    with open(inFile,'r') as f:
        for line in f:
            line = line.rstrip()
            sta,_lat,_lon,_ele = line.split()
            stas.append(sta)
            lonlats.append([float(_lon),float(_lat)])
            _eles.append(_ele)
            
    lonlats_rot = spherical_rotate(lonlats,center=center,degree=deg)
    f = open(outFile,'w')
    for i in range(len(stas)):
        f.write(format(stas[i],"<7s"))
        f.write(" ")
        f.write(format(lonlats_rot[i,1],'>10.6f'))
        f.write(" ")
        f.write(format(lonlats_rot[i,0],'>11.6f'))
        f.write(" ")
        f.write(_eles[i])
        f.write("\n")

def spherical_rotate(lonlats,center,degree):
    """
    rotate in degree, postive value for anti-clockwise direction
    """
    lonlatBs = lonlats.copy()
    if isinstance(lonlatBs,list):
        lonlatBs = np.array(lonlatBs)
    lonlatCs = []
    if len(lonlatBs.shape) == 1:
        lonC,latC = _spherical_rotate(lonlatBs,center,degree)
        lonlatCs.append([lonC,latC])
    else:
        for i in range(lonlatBs.shape[0]):
            lonC,latC = _spherical_rotate(lonlatBs[i,:],center,degree)
            lonlatCs.append([lonC,latC])
    return np.array(lonlatCs)
            
def _spherical_rotate(lonlatB,center,degree):

    # A the rotation center, 
    # B the point need to be rotated
    # C the rotated point
    # P the north pole
    # O the sphere center
    
    epsilon = 0.000001

    lonA = np.deg2rad(center[0])
    latA = np.deg2rad(center[1])
    lonB = np.deg2rad(lonlatB[0])
    latB = np.deg2rad(lonlatB[1])
    
    rotate = np.deg2rad(degree)
    
    if rotate % (2*pi) == 0:      # no change
        return lonlatB[0],lonlatB[1]

    dist_deg = spherical_dist(center[0],center[1],lonlatB[0],lonlatB[1])
    AOB = np.deg2rad(dist_deg)
    if AOB<epsilon:
        return lonlatB[0],lonlatB[1]

    dlon = lonB - lonA
    dlat = latB - latA
        
    # spherical sines law
    # sin(PAB) = sin(POB)*sin(APB)/sin(AOB)
    
    if dlon>=0:
        tmp = sin(pi/2-latB)*sin(dlon)/sin(AOB)
        if tmp >=1 and tmp <= 1.0+epsilon:
            tmp = 1
        if dlat >=0:
            PAB = asin(tmp)
        else:
            PAB = pi - asin(tmp)
        PAC = PAB - rotate

        # cos(a) = cos(b)cos(c)+sin(b)sin(c)cos(A)
        AOC = AOB # spherical dist is the same before and after rotation
        POC = acos(cos(pi/2-latA)*cos(AOC)+sin(pi/2-latA)*sin(AOC)*cos(PAC))
        latC = pi/2 - POC

        # sines law
        tmp = sin(AOC)*sin(PAC)/sin(POC)
        if tmp >=1 and tmp <= 1.0+epsilon:
            tmp = 1
        APC = asin(tmp)
        lonC = lonA + APC
    else:
        tmp = -sin(pi/2-latB)*sin(dlon)/sin(AOB)
        if tmp >=1 and tmp <= 1.0+epsilon:
            tmp = 1
        if dlat >=0:
            PAB = asin(tmp)
        else:
            PAB = pi - asin(tmp)
        PAC = PAB + rotate      

        # cos(a) = cos(b)cos(c)+sin(b)sin(c)cos(A)
        AOC = AOB # spherical dist is the same before and after rotation
        POC = acos(cos(pi/2-latA)*cos(AOC)+sin(pi/2-latA)*sin(AOC)*cos(PAC))
        latC = pi/2 - POC

        # sines law
        tmp = sin(AOC)*sin(PAC)/sin(POC)
        if tmp >=1 and tmp <= 1.0+epsilon:
            tmp = 1
        APC = asin(tmp)
        lonC = lonA - APC
    # quality control
    BOC_deg = spherical_dist(np.rad2deg(lonB),np.rad2deg(latB),np.rad2deg(lonC),np.rad2deg(latC))
    BOC = np.deg2rad(BOC_deg)
    tmp = (cos(BOC)-cos(AOC)*cos(AOB))/(sin(AOC)*sin(AOB))
    if tmp > 1 and tmp < 1.0+epsilon:
        tmp = 1
    if tmp <-1 and tmp > -1.0-epsilon:
        tmp = -1
    inverse_rotate = acos(tmp)
    
    assert (np.abs(inverse_rotate) - np.abs(rotate)) <=0.01
    distAB = spherical_dist(np.rad2deg(lonA),np.rad2deg(latA),np.rad2deg(lonB),np.rad2deg(latB))
    distAC = spherical_dist(np.rad2deg(lonA),np.rad2deg(latA),np.rad2deg(lonC),np.rad2deg(latC))        
    assert (distAB-distAC)<=epsilon

    return np.rad2deg(lonC),np.rad2deg(latC)

def mesh_rotate(x1,y1,center,degree,method="Cartesian"):
    ori_shape = x1.shape
    length = ori_shape[0] * ori_shape[1]
    tmp_x1 = np.zeros((length,1))
    tmp_y1 = np.zeros((length,1))
    tmp_x1[:,0] = x1.ravel()
    tmp_y1[:,0] = y1.ravel()
    tmp_x1y1 = np.concatenate((tmp_x1,tmp_y1),axis=1)
    if method == "Cartesian":
        rotated_tmp_x1y1 = cartesian_rotate(tmp_x1y1,center=center,degree=degree)
    elif method == "Sphere":
        rotated_tmp_x1y1 = spherical_rotate(tmp_x1y1,center=center,degree=degree)
    else:
        raise Exception("Method provided not in ['Cartesian','Sphere']")
    rotated_x1 = rotated_tmp_x1y1[:,0].reshape(ori_shape[0],ori_shape[1])
    rotated_y1 = rotated_tmp_x1y1[:,1].reshape(ori_shape[0],ori_shape[1])
    return rotated_x1,rotated_y1

def fault_vectors(strike,dip,rake,unit='degree'):
    """
    rake: slip angle
    return
    | n: the unit normal vector of the fault plane
    | d: the unit vector of the slip direction
    | b: intermediate vector by n x d
    | t: sigma3
    | p: sigma1
    """
    if unit == 'degree':
        strike = np.deg2rad(strike)
        dip = np.deg2rad(dip)
        rake = np.deg2rad(rake)
    elif unit == 'radian':
        pass
    else:
        raise Exception("Wrong unit, should be 'degree' or 'radian'")
    
    n = np.zeros(3)
    n[0] = -1*np.sin(dip)*np.sin(strike)
    n[1] = -1*np.sin(dip)*np.cos(strike)
    n[2]=np.cos(dip)
    
    d = np.zeros(3)
    d[0] = np.cos(rake)*np.cos(strike) + np.sin(rake)*np.cos(dip)*np.sin(strike)
    d[1] = -1*np.cos(rake)*np.sin(strike)+np.sin(rake)*np.cos(dip)*np.cos(strike)
    d[2] = np.sin(rake)*np.sin(dip)
    
    b = np.cross(n,d)
    t = n+d
    p = n-d
    
    return n,d,b,t,p

def lonlat_by_dist(orglon,orglat,delta_x,delta_y,R=6378.1):
    """
    Assuming sphere earth, calculate new longitude and latitude 
    base on delta x and y in kilometers.
    
    Parameters
    |  orglon: Original longitude
    |  Orglat: Oiginal latitude
    | delta_x: distance along x direction in km
    | delta_y: distance along y direction in km
    """
    # quality control
    if orglat <-90 or orglat >90:
        raise Exception("Latitude should be in [-90,90]")
    if orglon<-180 or orglon>180:
        raise Exception("Longitude should be in [-180,180]")
    
    delta_lat = delta_y/R/np.pi*180
    newlat = orglat + delta_lat
    if newlat < -90:
        newlat =180 + newlat
    if newlat >90:
        newlat = 180-newlat
        
    R_ = R*np.cos(np.deg2rad(orglat))
    delta_lon = delta_x/R_/np.pi*180
    
    newlon = orglon + delta_lon
    
    if newlon>180:
        newlon -=180
    if newlon <=-180:
        newlon += 180

    return newlon,newlat

def ellipse(center=[0,0],xamp=1,yamp=1,inters=101,rotate=0):
    """
    return x, y for ellipse drawing.
    xamp,yamp: amplification of x values and y values
    inters: the interpolation nodes along the x direction
    rotate: rotation angle in degree, positive for anticlockwise
    """
    x = np.linspace(1,-1,inters)
    y = np.sqrt(1-x**2)
    centerx = center[0]
    centery = center[1]
    xs = np.concatenate((x,-x))
    ys = np.concatenate((y,-y))
    xs = xs*xamp
    ys = ys*yamp
    if rotate!=0:
        logging.info(f"ellipse function rotation applied...")
        xys = np.zeros((len(xs),2))
        xys[:,0] = xs
        xys[:,1] = ys
        xys_rotate = cartesian_rotate(xys,degree=degree)
        xs = xys_rotate[:,0]
        ys = xys_rotate[:,1]

    return centerx+xs,centery+ys

def data3Dinterp(nodes,data, xs,ys,zs,method='linear'):
    # interpolate data by xs, lat, zs provided
    zzzs, yyys, xxxs = np.meshgrid(zs, ys, xs, indexing='ij')
    vals =  griddata(nodes, data, (zzzs, yyys, xxxs), method=method)

    return vals
    
def data_extraction(xs,ys,dataXs,dataYs,dataVvs,mode='geo'):
    """
    extract data from 2D array for provided points (x,y). Cloest node values returned
    parameters
    |      xs: 1D x values list to be extracted
    |      ys: 1D y values list to be extracted
    |  dataXs: 1D x values list for 2D array
    |  dataYs: 1D y values list for 2D array
    | dataVvs: 2D data values to be extracted
    |    mode: "geo" or "km"
    return dists and values. If mode is 'geo', dists in unit of 'km'

    """
    assert mode in ['geo','km']
    
    vs = []
    dists = []
    
    for i in range(len(xs)):
        tmpx = xs[i]
        tmpy = ys[i]
        if mode == 'geo':
            dist,_,_ = gps2dist_azimuth(ys[0],xs[0],tmpy,tmpx)
            distKm = dist/1000
        elif mode == "km":
            distKm = np.sqrt((tmpx-xs[0])**2+(tmpy-ys[0])**2)

        tmpdx = np.abs(dataXs-tmpx)
        tmpidx = np.argmin(tmpdx)
        tmpdy = np.abs(dataYs-tmpy)
        tmpidy = np.argmin(tmpdy)
        vs.append(dataVvs[tmpidy,tmpidx])
        dists.append(distKm)
    
    return dists,vs
    
def cata_projection_GMT(cataPth,blon,blat,elon,elat,_widths='-3/3'):
    """
    return projected catalog using GMT
    Parameters
      cataPth: Path for catalog file, it should be format: 
                 | evid | lon | lat | dep | mag | relative_time(int,float) |
    blon,blat: The projection start point longitude and latitude
    elon,elat: The projection end point longitude and latitude
       widths: Projection width, "-3/3" means left 3 km and right 3 km
    """
    print("Check GMT version: ",end = ' ')
    if os.system("gmt --version") != 0:   # gmt not installed
        raise Exception("GMT not installed!")
    rdName = np.random.randint(100)
    cmd=f'cata={cataPth}\n'
    cmd+=f'blon={blon}\n'
    cmd+=f'blat={blat}\n'
    cmd+=f'elon={elon}\n'
    cmd+=f'elat={elat}\n'
    cmd+=f"widths={_widths}\n"
    cmd+="awk '{print $2,$3,$4,$6,$5}' $cata|gmt project -C$blon/$blat -E$elon/$elat -Fxyzp -Lw -W$widths -Q >"+\
        f"{rdName}.project"
    os.system(cmd)
    eqs = np.loadtxt(f"{rdName}.project")
    os.system(f'rm {rdName}.project')
    
    return eqs
    
def cata_projection_GMT(cata,blon,blat,elon,elat,_widths='-3/3'):
    """
    return projected catalog using GMT
    Parameters
         cata: Python Catalog object from seisloc.cata
    blon,blat: The projection start point longitude and latitude
    elon,elat: The projection end point longitude and latitude
       widths: Projection width, "-3/3" means left 3 km and right 3 km
    """
    print("Check GMT version: ",end = ' ')
    if os.system("gmt --version") != 0:   # gmt not installed
        raise Exception("GMT not installed!")
    cata.write_info(fileName="tmp_cata.txt")
    rdName = np.random.randint(100)
    cmd=f'blon={blon}\n'
    cmd+=f'blat={blat}\n'
    cmd+=f'elon={elon}\n'
    cmd+=f'elat={elat}\n'
    cmd+=f"widths={_widths}\n"
    cmd+="awk '{print $2,$3,$4,$6,$5}' tmp_cata.txt|gmt project -C$blon/$blat -E$elon/$elat -Fxyzp -Lw -W$widths -Q >"+\
        f"{rdName}.project"
    os.system(cmd)
    eqs = np.loadtxt(f"{rdName}.project")
    os.system(f'rm {rdName}.project')
    subprocess.run(["rm","tmp_cata.txt"])
    
    return eqs
    
def xy_projection_GMT(xys,blon,blat,elon,elat,_widths='-3/3'):
    """
    return projected xy points using GMT
    Parameters
         xys: 2D list of xy points, first row is lon, second row is lat
    blon,blat: The projection start point longitude and latitude
    elon,elat: The projection end point longitude and latitude
       widths: Projection width, "-3/3" means left 3 km and right 3 km
    """
    print("Check GMT version: ",end = ' ')
    if os.system("gmt --version") != 0:   # gmt not installed
        raise Exception("GMT not installed!")
    if isinstance(xys,list):
        xys = np.array(xys)
    if len(xys.shape)!=2:
        raise Exception("Expected xys to be 2-D array")
    np.savetxt("tmp_xys.tmp",xys)
    rdName = np.random.randint(100)
    cmd=f'blon={blon}\n'
    cmd+=f'blat={blat}\n'
    cmd+=f'elon={elon}\n'
    cmd+=f'elat={elat}\n'
    cmd+=f"widths={_widths}\n"
    cmd+="awk '{print $1,$2}' tmp_xys.tmp|gmt project -C$blon/$blat -E$elon/$elat -Fxyp -Lw -W$widths -Q >"+\
        f"{rdName}.project"
    os.system(cmd)
    results = np.loadtxt(f"{rdName}.project")
    os.system(f'rm {rdName}.project')
    subprocess.run(["rm","tmp_xys.tmp"])
    
    return results  
