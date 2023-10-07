import matplotlib.pyplot as plt
import numpy as np

def draw_vel(dep_list,vel_list,ax=None,color='k',linestyle='-',label=""):
    """
    Draw velocity line on the ax based on the depth list and velocity list
    """
    points_list = []
    points_list.append([dep_list[0],vel_list[0]])
    for i in range(1,len(dep_list)):
        points_list.append([dep_list[i],vel_list[i-1]])
        points_list.append([dep_list[i],vel_list[i]])
        
    points_list = np.array(points_list)
    if ax==None:
        ax = plt.gca()
    line, = ax.plot(points_list[:,1],points_list[:,0],color=color,linestyle=linestyle,label=label)
    return line
