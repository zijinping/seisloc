#-----------------------------------------------------------------------------
import numpy as np
from math import sin,cos,asin,acos,pi,radians
from numba import jit

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
def in_rectangle(locs,alon,alat,blon,blat,width):
    results = np.zeros(locs.shape)
    dlon1 = blon - alon
    dlat1 = blat - alat
    rad_alon = radians(alon)
    rad_alat = radians(alat)
    norm1 = (dlon1**2+dlat1**2)**0.5
    for i in range(locs.shape[0]):
        ilon = locs[i,0]
        ilat = locs[i,1]
        dlon2 = ilon - alon
        dlat2 = ilat - alat
        norm2 = (dlon2**2+dlat2**2)**0.5
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
    dlon = lon2 - lon1
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

@jit(nopython=True)
def densityMap(lonlist,latlist,locs,longap,latgap,near=5):
    denSums = np.zeros((len(latlist),len(lonlist)))
    denCounts = np.zeros((len(latlist),len(lonlist)))
    
    for i,lat in enumerate(latlist):
        for j,lon in enumerate(lonlist):
            for ii in range(-1*near+1,near):
                for jj in range(-1*near+1,near):
                    if (i+ii)>=0 and (i+ii)<len(latlist) and (j+jj)>=0 and (j+jj)<len(lonlist):
                        denCounts[i+ii,j+jj]+=1
            for loc in locs:
                loclon = loc[0]
                loclat = loc[1]
                if loclon>=lon and loclon<lon+longap and loclat>=lat and loclat<lat+latgap:
                    for kk in range(-1*near+1,near):
                        for ll in range(-1*near+1,near):
                            if (i+kk)>=0 and (i+kk)<len(latlist) and (j+ll)>=0 and (j+ll)<len(lonlist):
                                denSums[i+kk,j+ll]+=1
    return denCounts,denSums
