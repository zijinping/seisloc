import numpy as np

def az_dip_to_beach_xy(stk,dip,phi,size):
    """
    Relationship learned from obspy.imaging.beachball.beach function
    
    Parameters
    | az, dip: (in degrees) azimuth and dip angle
    |     stk: (in degrees) fault strike of the corresponding beach ball
    |    size: (no unit) same with the size parameter in the "beach" function
    """

    stkRad = np.radians(stk)
    phiRad = np.radians(phi)
    
    l1 = np.sqrt(                                           
        np.power(dip, 2) / (                                    
        np.power(np.sin(phiRad), 2) + 
        np.power(np.cos(phiRad), 2) * 
        np.power(dip, 2) / np.power(90, 2)))
    x1,y1 = l1*np.cos(phiRad+stkRad),l1*np.sin(phiRad+stkRad)
    
    x1 = x1*size/2/90
    y1 = y1*size/2/90

    return x1, y1
