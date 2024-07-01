import os
from obspy.geodetics import gps2dist_azimuth

def geo_spatial_dist(lo1,la1,dp1,lo2,la2,dp2):
    """
    Input depth in kilometers
    Return distance in kilometers
    """
    distHm,_,_ = gps2dist_azimuth(la1,lo1,la2,lo2)
    distHkm = distHm/1000
    ddep = dp1 - dp2

    return (distHkm**2+ddep**2)**0.5
