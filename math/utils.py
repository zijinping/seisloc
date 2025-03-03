from numba import cuda
import math
import matplotlib.pyplot as plt

@cuda.jit()
def _matmul_gpu(A,B,C):
    row,col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row,k] * B[k,col]
        C[row,col] = tmp

def matmul_gpu(A,B,C,TPX=16,TPY=16):
    if len(cuda.gpus) == 0:
        raise Exception("Error, no gpu available")
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array(A.shape[0],B.shape[1])
    threads_per_block = (TPX,TPY)
    blocks_per_grid_x = int(math.ceil(A.shape[0]/threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(B.shape[1]/threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x,blocks_per_grid_y)

    _matmul_gpu[blocks_per_grid,threads_per_block](A_global_mem,
                                                  B_global_mem,
                                                  C_global_mem)
    cuda.syschronize()
    C_global_gpu = C_global_mem.copy_to_host()


def matrix_show(*args,**kwargs):
    """
    Show matrix values in grids shape
    Parameters:cmap="cool",gridsize=0.6,fmt='.2f',label_data=True
    """
    ws = []
    H = 0
    str_count = 0
    ndarr_count = 0
    new_args = []
    for arg in args:
        if isinstance(arg,str):
            new_args.append(arg)
            continue
        if isinstance(arg,list):
            arg = np.array(arg)
        if len(arg.shape)>2:
            raise Exception("Only accept 2D array")
        if len(arg.shape) == 1:
            n = arg.shape[0]
            tmp = np.zeros((n,1))
            tmp[:,0] = arg.ravel()
            arg = tmp
        h,w = arg.shape
        if h>H:
            H=h
        ws.append(w)
        new_args.append(arg)
        ndarr_count += 1
    W = np.sum(ws)+len(ws)    # text+matrix+text+...+matrix+text
    if W<0:
        raise Exception("No matrix provided!")
        
    fmt = '.2f'
    grid_size = 0.6
    cmap = 'cool'
    label_data = True
    pltShow = True
    for arg in kwargs:
        if arg == "fmt":
            fmt = kwargs[arg]
        if arg == 'grid_size':
            grid_size = kwargs[arg]
        if arg == 'cmap':
            cmap = kwargs[arg]
        if arg == 'label_data':
            label_data = kwargs[arg]
    fig = plt.figure(figsize=(W*grid_size,H*grid_size))
    gs = fig.add_gridspec(nrows=H,ncols=W)
    
    wloop = 0
    matrix_id = 0
    for arg in new_args:
        if isinstance(arg,str):
            ax = fig.add_subplot(gs[0:H,wloop-1:wloop])
            ax.axis("off")
            ax.set_xlim(0,1)
            ax.set_ylim(0,H)
            ax.text(0.5,H/2,arg,horizontalalignment='center',verticalalignment='center')
        if isinstance(arg,np.ndarray):
            h,w = arg.shape
            hlow = int(np.round((H-h+0.01)/2))        # Find the height grid range
            hhigh = hlow+h
            wlow = wloop
            whigh = wlow+w
#            print("H: ",H,hlow,hhigh,"; W ",W,wlow,whigh)
            ax = fig.add_subplot(gs[hlow:hhigh,wlow:whigh])
            
            plt.pcolormesh(arg,cmap=cmap)
            for i in range(1,w):
                plt.axvline(i,color='k',linewidth=0.5)
            for j in range(1,h):
                plt.axhline(j,color='k',linewidth=0.5)
            if label_data:
                for i in range(h):
                    for j in range(w):
                        plt.text(j+0.5,i+0.5,format(arg[i,j],fmt),
                                 horizontalalignment='center',
                                 verticalalignment='center')
            plt.xlim(0,w)
            plt.ylim([h,0])
            plt.xticks([])
            plt.yticks([])
            wloop+=w+1
            matrix_id+=1
    if pltShow == True:
        plt.show()