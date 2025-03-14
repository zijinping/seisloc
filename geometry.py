import os
import math
import logging
import subprocess
import numpy as np
from numba import jit
from matplotlib import path
from scipy.interpolate import griddata
from math import sin,cos,asin,acos,pi,radians
from obspy.geodetics import gps2dist_azimuth
from seisloc.math.math import lsfit,weighted_lsfit

def move_by_dist_az_sphere(laAdeg, loAdeg, distKm, az, R=6371.0):
    """
    Calculate the destination latitude/longitude on a sphere after traveling a 
    certain distance along a specific azimuth.
    
    Args:
      lat_A_deg (float): Starting latitude in degrees
      lon_A_deg (float): Starting longitude in degrees
    bearing_deg (float): Bearing in degrees (0° is north, increases clockwise)
    distance_km (float): Distance to travel in kilometers
              R (float): Earth radius in kilometers (default 6371.0)
        
    Returns:
        (lat_B_deg, lon_B_deg): Destination latitude and longitude in degrees
    """
    laA = math.radians(laAdeg)
    loA = math.radians(loAdeg)
    alpha = math.radians(az)
    theta = distKm / R

    sinPhi1 = math.sin(laA)
    cosPhi1 = math.cos(laA)
    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    sinAlpha = math.sin(alpha)
    cosAlpha = math.cos(alpha)

    sinPhi2 = sinPhi1 * cosTheta + cosPhi1 * sinTheta * cosAlpha
    sinPhi2 = max(min(sinPhi2, 1.0), -1.0)
    phi2 = math.asin(sinPhi2)

    y = sinAlpha * sinTheta
    x = cosTheta * cosPhi1 - sinPhi1 * sinTheta * cosAlpha
    delta_lambda = math.atan2(y, x)

    loB = loA + delta_lambda
    loB = (loB + math.pi) % (2 * math.pi) - math.pi

    laBdeg = math.degrees(phi2)
    loBdeg = math.degrees(loB)
    
    return round(laBdeg, 6), round(loBdeg, 6)


def dist_az_sphere(lat_A_deg, lon_A_deg, lat_B_deg, lon_B_deg, R=6371.0):
    """
    Calculate the great-circle distance and bearing between two points on a sphere.
    
    Args:
        lat_A_deg, lon_A_deg (float): Latitude and longitude of point A in degrees
        lat_B_deg, lon_B_deg (float): Latitude and longitude of point B in degrees
        R (float): Earth radius in kilometers (default 6371.0)
        
    Returns:
        distance_km (float): Great-circle distance in kilometers
        bearing_deg (float): Bearing in degrees, 0° is north, increases clockwise
    """
    lat_A = math.radians(lat_A_deg)
    lon_A = math.radians(lon_A_deg)
    lat_B = math.radians(lat_B_deg)
    lon_B = math.radians(lon_B_deg)
    d_lon = lon_B - lon_A

    cos_term = (math.sin(lat_A) * math.sin(lat_B) +
                math.cos(lat_A) * math.cos(lat_B) * math.cos(d_lon))
    distance_km = R * math.acos(max(min(cos_term, 1.0), -1.0))

    y = math.sin(d_lon) * math.cos(lat_B)
    x = (math.cos(lat_A) * math.sin(lat_B) -
         math.sin(lat_A) * math.cos(lat_B) * math.cos(d_lon))
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad) % 360

    return distance_km, bearing_deg

def loc_by_width(la1:float,lo1:float,la2:float,lo2:float,width:float,direction='right',mode="geo")->tuple:
    """
    Given points A(la1,lo1) and B(la2,lo2), extend A and B laterally to new positions 
    according to width and direction provided under the sphere earth model.
    If mode is "normal", then the calculation will be in the Cartesian model.

    Parameters:
      la1,lo1: latitude and longitude of point A
      la2,lo2: latitude and longitude of point B
        width: (kilometer)
    direction: the side of new points wrt. A->B
         mode: default "geo", if "normal", calculation will be in Cartesian
    """
    if mode == "geo":
        dist, az = dist_az_sphere(la1,lo1,la2,lo2)
        if direction == "right":  # extend width to the right
            la1New,lo1New = move_by_dist_az_sphere(la1,lo1,width,az+90)
            la2New,lo2New = move_by_dist_az_sphere(la2,lo2,width,az+90)
        elif direction == "left":
            la1New,lo1New = move_by_dist_az_sphere(la1,lo1,width,az-90)
            la2New,lo2New = move_by_dist_az_sphere(la2,lo2,width,az-90)
        else:
            raise Exception("direction should be 'right' or 'left'")
        
        return la1New,lo1New,la2New,lo2New
    elif mode == "normal":
        # use the Cartesian model
        x1 = lo1; y1 = la1
        dx = x2 - x1
        x2 = lo2; y2 = la2
        dy = y2 - y1
        dist = ((x2-x1)**2+(y2-y1)**2)**0.5
        if direction == "right":
            cross = np.cross([dx,dy,0],[0,0,1])
            v = cross/np.linalg.norm(cross)
        elif direction == "left":
            cross = np.cross([0,0,1],[dx,dy,0])
            v = cross/np.linalg.norm(cross)
        else:
            raise Exception("direction should be 'right' or 'left'")
        deltaY = width*v[1]    # cos_theta
        deltaX = width*v[0]    # sin_theta
        x1New = x1 + deltaX
        y1New = y1 + deltaY
        x2New = x2 + deltaX
        y2New = y2 + deltaY

        return y1New,x1New,y2New,x2New
    else:
        raise Exception("mode should be 'geo' or 'normal'")

def latlon_to_xyz(lat, lon, R=6371e3):
    """
    Convert latitude/longitude to 3D Cartesian coordinates (in meters).
    
    Args:
        lat (float or ndarray): Latitude in degrees
        lon (float or ndarray): Longitude in degrees
        R (float): Earth radius in meters (default 6371e3)
        
    Returns:
        ndarray: 3D coordinates [x, y, z]
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return np.column_stack([x, y, z]) if isinstance(lat, np.ndarray) else np.array([x, y, z])


def signed_projections_sphere(A_lat, A_lon, B_lat, B_lon, C_lats, C_lons, R=6371e3):
    """
    Compute the signed parallel and perpendicular distances from point(s) C to the great-circle arc from A to B (strict spherical model).
    
    Args:
        A_lat, A_lon (float): Latitude and longitude of point A in degrees
        B_lat, B_lon (float): Latitude and longitude of point B in degrees
        C_lats, C_lons (array-like): Latitudes and longitudes of point(s) C in degrees
        R (float): Earth radius in meters (default 6371e3)
        
    Returns:
        d_parallel (ndarray): Signed arc length along AB (meters). Positive along AB's direction.
        d_perpendicular (ndarray): Signed perpendicular arc length (meters). Sign determined by the normal of OA×OB.
    """
    OA = latlon_to_xyz(A_lat, A_lon, R)
    OB = latlon_to_xyz(B_lat, B_lon, R)
    OC = latlon_to_xyz(C_lats, C_lons, R)

    n = np.cross(OA, OB)
    n_norm = np.linalg.norm(n)
    AB = OB - OA
    AB_unit = AB / np.linalg.norm(AB)

    t = np.dot(OC, n) / n_norm**2
    OP_plane = OC - t.reshape(-1, 1) * n
    OP_norm = OP_plane / np.linalg.norm(OP_plane, axis=1, keepdims=True) * R

    OA_unit = OA / R
    OP_unit = OP_norm / R
    cos_theta_AP = np.sum(OA_unit * OP_unit, axis=1)
    theta_AP = np.arccos(np.clip(cos_theta_AP, -1, 1))
    d_parallel = R * theta_AP

    AP_vectors = OP_norm - OA.reshape(1, 3)
    sign_parallel = np.sign(np.einsum('ij,j->i', AP_vectors, AB_unit))
    d_parallel *= sign_parallel

    OC_dot_n = np.dot(OC, n)
    sign_perp = np.sign(OC_dot_n)
    sin_theta = np.abs(OC_dot_n) / (R * n_norm)
    theta_perp = np.arcsin(np.clip(sin_theta, 0, 1))
    d_perpendicular = sign_perp * R * theta_perp

    return d_parallel/1000, d_perpendicular/1000

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

def bdy2pts(xmin,xmax,ymin,ymax):
    """
    Convert boundary to points for plot
    """
    points = [[xmin,ymin],
              [xmin,ymax],
              [xmax,ymax],
              [xmax,ymin],
              [xmin,ymin]]

    return np.array(points)

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

def in_polygon(xys,polygon,mode="normal"):
    """
    Check if points are inside the polygon
    mode: "normal" or "geo"

    xys: list of [lon,lat]
    polygon: list of [lon,lat]
    """
    if mode == "geo":
        results = []
        vertices = [s2.S2LatLng.FromDegrees(lat, lon).ToPoint() for (lat, lon) in polygon]
        p = s2.S2Loop(vertices)
        for xy in xys:
            point = s2.S2LatLng.FromDegrees(xy[1],xy[0]).ToPoint()
            results.append(p.Contains(point))
        return results
    if mode == "normal":
        p = path.Path(polygon)
        return p.contains_points(xys)

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



def seismic_path_calculation(e_dep,in_angle,vel_set):
    """
    Parameters:
        e_lon,e_lat,e_dep: the earthquake longitude, latitude and depth
        sta_lon,sta_lat: the station location
        vel_set: array format [[dep1,vel1],[dep2,vel2]], where vel1 indicates 
        the velocity between dep1 and dep2
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



def rotate_matrix(theta):
    """
    Return 2D antickwise rotation matrix. Input unit in degree
    """
    thetaRad = np.radians(theta)
    return np.array([[np.cos(thetaRad),-np.sin(thetaRad)],
                     [np.sin(thetaRad),np.cos(thetaRad)]])

def cartesian_rotate(xy,center=[0,0],angle=0):
    """
    angle is positive for anticlockwise
    """
    if isinstance(xy,list):
        xy = np.array(xy)
    if len(xy.shape)==1:
        raise Exception("xy should be 2 dimensional matrix")
    if isinstance(center,list):
        center = np.array(center)

    tmpxy = xy - center
    rM = rotate_matrix(angle)
    tmpxyRot = np.matmul(tmpxy,rM)
    xyRot = tmpxyRot + center
    
    return xyRot

def event_dat_rotate(inFile,center,angle):
    """
    Rotate the longitude and latitude of events in the event.dat (tomoDD/hypoDD) 
    file. 
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
    xys_rot = spherical_rotate(xys,center=center,degree=angle)

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
    The coordinate system: x(east), y(north), and z(upward)
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
    #==================================
    # In the conventional coordinate: x(north);y(west);z(upward)
    # I prefer x(east);y(north);z(upward)
    strike=strike-np.pi/2
    #==================================

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
        xys_rotate = cartesian_rotate(xys,degree=rotate)
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

def geo_az_from_xy(x,y):
    if x>=0:
        strikeDeg = np.rad2deg(np.arccos(y/(x**2+y**2)**0.5))
    elif x<0:
        strikeDeg = 360 - np.rad2deg(np.arccos(y/(x**2+y**2)**0.5))
    
    return strikeDeg
    
def strike_dip_from_locs(lons,lats,deps,refLon=None,refLat=None,W=1):
    '''
    only applicable for dimension that earthquake spherical effect is weak
    lons,lats: 1-D longitude and latitude numpy array
    W: Paramter for weighting, can be in the format of a value, 1-D array or 2-D array
    '''
    #---------- format control ------------------
    if isinstance(lons,list):
        lons = np.array(lons)
    if isinstance(lats,list):
        lats = np.array(lats)
    if isinstance(deps,list):
        deps = np.array(deps)
    assert len(lons.shape) == 1 and len(lats.shape)==1 and len(deps.shape)==1
    assert len(lons) == len(lats) and len(lons) == len(deps)
    #--------------- preprocessing ---------------
    if refLon == None:
        refLon = np.mean(lons)
    if refLat == None:
        refLat = np.mean(lats)
    xs = (lons-refLon)*111.1*np.cos(np.deg2rad(refLat))
    ys = (lats-refLat)*111.1
    rs = np.sqrt(xs**2+ys**2)
    if np.max(rs) >100:
        print("Warning: the raidus exceeds 100 km!")

    G = np.ones((len(lons),3))
    G[:,0] = xs
    G[:,1] = ys
    dobs = deps
    m = weighted_lsfit(G,dobs,W)
    refZ = m[2]
    dipAzDeg = geo_az_from_xy(m[0],m[1])
    strikeAzDeg=dipAzDeg-90
    if strikeAzDeg <0:
        strikeAzDeg += 360
    
    dipDeg = 90-np.rad2deg(np.arctan(1/np.linalg.norm(m[:2])))
    
    return strikeAzDeg, dipDeg, refZ

def vector_decompose1(v1,v2):
    """
    Decompose vector(v1) into the component along v2 and the component normal to v2
    """
    v2n = v2/np.linalg.norm(v2)
    v1AlongV2 = np.dot(v1,v2n)*v2n
    v1NormalV2 = v1 - v1AlongV2

    return v1AlongV2, v1NormalV2

def vector_decompose2(v1,v2,v3=np.array([0,0,1])):
    """
    Get the component of v1 in the plane defined by v2 and v3
    """
    vn = np.cross(v2,v3)
    vn = vn/np.linalg.norm(vn)
    if np.linalg.norm(vn) == 0:
        raise Exception(f"The two vectors {v2} and {v3} are parallel!")
    v1NormalV2v3 = np.dot(v1,vn)*vn
    v1InV2v3 = v1 - v1NormalV2v3

    return v1InV2v3, v1NormalV2v3

def angle_vxvy(vx,vy,vz=[0,0,1]):
    """
    The angle between two vectors, the result is in degree. vz works as a reference 
    for the determination of "mkr". The "mkr" is 1 if vx,vy, and vz follows a 
    right-hand rule, otherwise -1.

    Return
    | angle: (unit: degree) the angle between the two vectors in degree
    |   mkr: a marker for the degree sign. 1 means v1 in the 
    """
    dot_product = np.dot(vx,vy)
    normVx = np.linalg.norm(vx)
    normVy = np.linalg.norm(vy)
    cosPhi = dot_product/(normVx*normVy)
    angleRad = np.arccos(cosPhi)
    angle = angleRad/np.pi*180

    mkr = 1
    crossProduct = np.cross(vx,vy)
    if np.dot(crossProduct,vz) < 0:
        mkr = -1 

    return angle,mkr

def dlodla2xy(dlo,dla,refLa=0):
    """
    Convert the difference in longitude and latitude to the x-y coordinate system.
    Unit in degree, x: eastward (km), y: northward (km)
    """
    x = dlo*111.19*np.cos(refLa/180*np.pi)
    y = dla*111.19
    return x,y

def sta2eve_xyz(evlo,evla,evdp,stlo,stla,stel):
    """
    Calculate the xyz location of the station w.r.t the earthquake hypocenter.
    x: estaward; y: northward; z: upward
    stel: (unit: km) station elevation, positive for above zero
    """
    xsta,ysta = dlodla2xy(stlo-evlo,stla-evla,refLa=evla)
    zsta = evdp + stel
    return [xsta,ysta,zsta]

def in_plane_normal_vector(v1,vref):
    """
    In the v1_vref plane, get a vecor that is normal to v1 and point far away from v1
    """
    v3 = np.cross(v1,vref)  # vector normal to v1_vref plane
    v4 = np.cross(v1,v3)    # 1) normal to v1; 2) point away from v1 and vref. 

    return v4/np.linalg.norm(v4)
