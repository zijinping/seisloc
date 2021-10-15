#-----------------------------------------------------------------------------
import numpy as np
from math import acos,sin,cos,pi,radians

def spherical_dist(lon_1,lat_1,lon_2,lat_2):
    """
    Calculate the distance of two postions and return distance in degree
    """

    lon_1,lat_1,lon_2,lat_2 = map(radians,[lon_1,lat_1,lon_2,lat_2])
    a=acos(sin(lat_1)*sin(lat_2)+cos(lat_1)*cos(lat_2)*cos(lon_2-lon_1))
    return a*180/pi

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
    sphe_dist = spherical_dist(lon1,lat1,lon2,lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    if direction == "right":  # extend width to the right 
    	delta_lat = -width*(dlon/sphe_dist)    # cos_theta
    	delta_lon =  width*(dlat/sphe_dist)    # sin_theta

    if direction == "left":
    	delta_lat =  width*(dlon/sphe_dist)    # cos_theta
    	delta_lon = -width*(dlat/sphe_dist)    # sin_theta

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
