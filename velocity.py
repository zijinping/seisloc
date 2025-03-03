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

def layer2linear_vel(inp_depths,inp_vels,linear_nodes):
    """
    Convert from layered 1-D model to linear velocity model, designed to conver from 
    the HYPOINVERSE model to tomoDD input model. It balances the average slowness.
    Parameters:
    | inp_depths: input depths, 1-D array/list
    | inp_vels: input velocities, 1-D array/list
    | linear_nodes: the depth nodes of output velocities
    
    Return:
    | average vlocity list correponding to the input velocity nodes

    Example:
    >>> inp_depths = [ -1,0.5,1.,1.5,2.13,4.13,6.9,10.07,15.93,17.,18.,27.,31.89]
    >>> inp_vels = [3.67,4.7,5.38,5.47,5.5,5.61,5.73,6.19,6.23,6.31,6.4,6.45,6.5]
    >>> tomo_deps = [0,1.5,3.0,4.5,6.0,7.5,9.0,10.5,12,18,24]
    >>> layer2linear_vel(inp_depths,inp_vels,tomo_deps)
    """
    
    #----------- quality control --------------------------------
    if linear_nodes[-1]>inp_depths[-1]:
        raise Exception("linear_nodes[-1] should less than inp_depths[-1]")
    if inp_depths.shape[0] != inp_vels.shape[0]:
        raise Eception("Different length of input depths quantity and velocity values")
    
    #----------- load in data ------------------------------------
    pxnodes = []
    pynodes = []
    for i in range(inp_depths.shape[0]):
        if i == 0:
            pxnodes.append(inp_vels[i])
            pynodes.append(inp_depths[i])
        else:
            pxnodes.append(inp_vels[i-1])
            pxnodes.append(inp_vels[i])
            pynodes.append(inp_depths[i])
            pynodes.append(inp_depths[i])

    #----------- calculate the mean velocity ----------------------
    avg_vels = []
    for i in range(len(linear_nodes)):
        dep = linear_nodes[i]
        #---------------get left and right range--------------
        if i == 0:                                   # first node
            dep_il = linear_nodes[i]
            dep_ir = (linear_nodes[i+1]+linear_nodes[i])/2
        elif i == len(linear_nodes) - 1:             # last node
            dep_il = (linear_nodes[i-1]+linear_nodes[i])/2
            dep_ir = linear_nodes[i]+(linear_nodes[i]-linear_nodes[i-1])/2
        else:
            dep_il = (linear_nodes[i-1]+linear_nodes[i])/2
            dep_ir = (linear_nodes[i+1]+linear_nodes[i])/2

        slw_list = []                                # slow list
        embrace_first = False
        for j in range(len(pynodes)):
            pynode = pynodes[j]
            velocity = pxnodes[j]                    # velocity

            if pynode >=dep_il: # consider velocity node within the range
                if embrace_first == False:
                    slw_list.append([dep_il,1/velocity]) # embrace the left node
                    embrace_first = True

                end_velocity = velocity
                if pynode>dep_ir:
                    break
                slw_list.append([pynode,1/velocity])
        slw_list.append([dep_ir,1/end_velocity])      # embrace the right node

        # calculate average slowness
        trav = 0          # travel time        
        for k in range(len(slw_list)-1):
            dep_a = slw_list[k][0]
            slw_a = slw_list[k][1]
            dep_b = slw_list[k+1][0]
            slw_b = slw_list[k+1][1]
            if (dep_b - dep_a)==0 or slw_a==slw_b:
                trav += (dep_b -dep_a)*slw_a
            else:
                raise Exception("(dep_b - dep_a)==0 or vel_a==vel_b")
        avg_vel = (dep_ir-dep_il)/trav
        avg_vels.append(avg_vel)
        
    return avg_vels
