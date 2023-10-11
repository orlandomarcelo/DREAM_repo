
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

from copy import copy
import imageio

import scipy

from scipy.spatial.distance import cdist
from matplotlib.ticker import NullFormatter
import mvgavg

CBB_PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def dtw(X, Y, metric='euclidean'):
    """
    
    Parameters
    ----------
    X : array_like
        M x D matrix
    Y : array_like
        N x D matrix    
    metric : string
             The distance metric to use. 
             Can be :
             'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
             'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
             'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
             'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
             'wminkowski', 'yule'.
             See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    Returns
    -------
    total_cost : float
                 Total (minimum) cost of warping
    pointwise_cost :  array_like
                      M x N matrix with cost at each (i, j)
    accumulated_cost : array_like
                       M x N matrix with (minimum) cost accumulated till (i,j)
                       having started from (0, 0)
        
    """
    
    X = np.array(X)
    Y = np.array(Y)
    if len(X.shape) == 1:
        # Reshape to N x 1 form
        X = X[:, np.newaxis]
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    # m = X length
    # n = Y length
    m, n = X.shape[0], Y.shape[0]
    D = np.zeros((m+1, n+1))
    D[1:, 0] = np.inf
    D[0, 1:] = np.inf
    D[1:,1:] = cdist(X, Y, metric)
    pointwise_cost = D[1:,1:].copy()
    for i in range(0, m):
        for j in range(0, n):
            cost = D[i+1, j+1]
            D[i+1, j+1] = cost + min (D[i, j+1], D[i+1, j], D[i, j])           
    accumulated_cost = D[1:, 1:]
    total_cost = D[m, n]/sum(D.shape)
    return total_cost, pointwise_cost, accumulated_cost


def get_path(D):
    """Traceback path of minimum cost 
    
    Given accumulated cost matrix D,
    trace back the minimum cost path
    
    Parameters
    -----------
    
    D : array_like
        M x N matrix as obtained from `accumulated_cost` using:
        total_cost, pointwise_cost, accumulated_cost = dtw(X, Y, metric='euclidean')
    
    Returns
    -------
    traceback_x, traceback_x : array_like
                               M x 1 and N x 1 array containing  indices of movement
                               starting from (0, 0) going to (M-1, N-1)
    """
    m , n = D.shape
    m = m - 1
    n = n - 1
    # Starting point is the end point
    traceback_x, traceback_y = [m], [n]
    while (m > 0 and n > 0):
        min_idx = np.argmin([D[m-1, n-1], D[m, n-1], D[m-1, n]])
        if min_idx == 0:
            # move diagonally
            m = m - 1
            n = n - 1
        elif min_idx == 1:
            # move vertically
            n = n - 1
        else:
            # move horizontally
            m = m - 1
        traceback_x.insert(0, m)
        traceback_y.insert(0, n)
    # End point is the starting point
    traceback_x.insert(0, 0)
    traceback_y.insert(0, 0)
    return np.array(traceback_x), np.array(traceback_y)

def plot_warped_timeseries(x, y, pointwise_cost, 
                           accumulated_cost, path, 
                           colormap=plt.cm.Blues,
                           linecolor=CBB_PALETTE[-2]):
    nullfmt = NullFormatter()    
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_heatmap = [left, bottom, width, height]
    rect_x = [left, bottom_h, width, 0.2]
    rect_y = [left_h, bottom, 0.2, height]

    fig = plt.figure(1, figsize=(8, 12))

    axHeatmap = plt.axes(rect_heatmap)
    axX = plt.axes(rect_x, sharex=axHeatmap)
    axY = plt.axes(rect_y, sharey=axHeatmap)

    # no labels
    axX.xaxis.set_major_formatter(nullfmt)
    axY.yaxis.set_major_formatter(nullfmt)
    
    axY.plot(y, range(0, len(y)), color=CBB_PALETTE[2])
    axX.plot(x, color=CBB_PALETTE[1])
    axHeatmap.imshow(accumulated_cost.T, 
                     origin='lower', 
                     cmap=colormap, 
                     interpolation='nearest')
    axHeatmap.plot(path[0], path[1], '-x', color=linecolor)
    #axHeatmap.xlim((-0.5, accumulated_cost.shape[0]-0.5))
    #axHeatmap.ylim((-0.5, accumulated_cost.shape[1]-0.5))
    return fig





def spectrum_a(time, fluo): #exponential decay
    beta = 6.5
    import numpy as np
    time = time + time[1]/2
    tau = time/beta
    logtau = np.log10(tau)
    corr = np.zeros(len(time))
    for j, tps in enumerate(time):
        pass
        funcwf = np.exp(-time[0:j+1]/tau[j])
        meanwf = np.sum(funcwf[0:j + 1]) / (j + 1)
        stdwf = np.sum((funcwf[0:j + 1] - meanwf) ** 2)
        meanfluo = np.sum(fluo[0:j + 1]) / (j + 1)
        stdfluo = np.sum((fluo[0:j + 1] - meanfluo)**2)
        num = np.sum((fluo[0:j+1])*(funcwf-meanwf))
        deno = np.sqrt(stdfluo * stdwf)
        corr[j] = num / deno /1.
    return logtau, corr

def spectrum_d(time, fluo):
    beta = 3.247
    import numpy as np
    time = time + time[1]/2 #démarre à zero + décalé d'un demi pas de temps correspond au milieu du temps d'intégration du capteur -> décaler d'un demi temps d'intégration de la cqamera
    tau = time/beta
    logtau = np.log10(tau)
    corr = np.zeros(len(time))
    def wf(x):
        conds = [x <= beta*tau[j]/2, (beta*tau[j]/2 < x) & (x <= beta*tau[j])] #+1 sur moitié, -1 sur l'autre
        funcs = [1, -1]
        return np.piecewise(x, conds, funcs)
    for j, tps in enumerate(time):
        pass
        funcwf = wf(time[0:j+1])
        meanwf = np.sum(funcwf[0:j + 1]) / (j + 1)
        stdwf = np.sum((funcwf[0:j + 1] - meanwf) ** 2)
        meanfluo = np.sum(fluo[0:j + 1]) / (j + 1)
        stdfluo = np.sum((fluo[0:j + 1] - meanfluo)**2)
        num = np.sum((fluo[0:j+1])*(funcwf-meanwf))
        deno = np.sqrt(stdfluo * stdwf)
        corr[j] = num / deno /0.771779
    return logtau, corr



def exp_decay(parameters, xdata):
    '''
    Calculate an exponential decay of the form:
    S = a * exp(-xdata/b)
    '''
    A = parameters[0]
    tau = parameters[1]
    y0 = parameters[2]
    return A * np.exp(-xdata/tau) + y0

def exp_decay_max(parameters, xdata):
    '''
    Calculate an exponential decay of the form:
    S = a * exp(-xdata/b)
    '''
    A = -xdata[0] + parameters[2]
    tau = parameters[1]
    y0 = -xdata[0] + parameters[0]
    return A * np.exp(-xdata/tau) + y0

def biexponential(parameters, xdata):
    A0, tau0, A1, tau1, y0, delay = parameters
    return A0*np.exp(-xdata/tau0) + A1*np.exp(-(xdata-delay)/tau1) + y0

def sigmoid(parameters, xdata):

    A = parameters[0]
    tau = parameters[1]
    y0 = parameters[2]
    dt = parameters[3]
    return A / (1 + np.exp(-(xdata-dt)/tau)) + y0


def residuals(parameters,x_data,y_observed,func):
    '''
    Compute residuals of y_predicted - y_observed
    where:
    y_predicted = func(parameters,x_data)
    '''
    return func(parameters,x_data) - y_observed

#                OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (-1e9,1e9),
#                                                    args = (x[0:(stop - start)//2], data_sample, function_to_fit))



def get_fit(time, y, x0, function_to_fit):
    time = time - time[0]
    lower_bounds = [-1e9]*len(x0)
    lower_bounds[1] = 0
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
                                    args = (time,y , function_to_fit))
    parameters_estimated = OptimizeResult.x
    ypred = function_to_fit(parameters_estimated, time)
    tau = parameters_estimated[1]
    duration = time[-1]-time[0]
    convert = duration/len(time)
    if tau >  duration//2: #if too high
            tau =  duration//2
    if tau < 5: #if too low, increase it
            tau = 5
    x0 = parameters_estimated #initial guess: parameters from previous fit
    #second fit
    
    bound = int(tau*5/convert)
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
                                    args = (time[0:bound],y[0:bound] , function_to_fit))
    return ypred, parameters_estimated







###################################################ORIENTATION

def gradient_magnitude(im):
    [H, L] = im.shape
    im0 = copy(im)
    col = np.zeros((H, 1))
    lin = np.zeros((1, L))
    imcolp = np.concatenate((im0, col), 1)
    imlinp = np.concatenate((im0, lin), 0)
    imcoln = np.concatenate((col, im0), 1)
    imlinn = np.concatenate((lin, im0), 0)
    diff = np.sqrt((imcolp[0:H, 1:L +1] - imcoln[0:H, 0:L])**2 + 
                   (imlinp[1: H+1, 0:L]-imlinn[0:H,0:L])**2)
    return diff

def gradient_orientation(im):
    [H, L] = im.shape
    im0 = copy(im)
    col = np.zeros((H, 1))
    lin = np.zeros((1, L))
    imcolp = np.concatenate((im0, col), 1)
    imlinp = np.concatenate((im0, lin), 0)
    imcoln = np.concatenate((col, im0), 1)
    imlinn = np.concatenate((lin, im0), 0)
    ori = np.arctan2((imcolp[0:H, 1:L +1] - imcoln[0:H, 0:L]), 
                   (imlinp[1: H+1, 0:L]-imlinn[0:H,0:L]))
    return ori

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
  

def gaussian_kernel(h, l, sigma = 'auto'):
    if sigma == 'auto':
        sigma_h = h/1.5
        sigma_l = l/1.5
    x = np.linspace(0, h-1, h)
    y = np.linspace(0, l-1, l)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(((X-h//2)/sigma_h)**2 + ((Y-l//2)/sigma_l)**2)/2)
    return Z

def assign_orientation(P_orientation, P_magnitude, s, N, n):
    '''
    Assigns the principal orientation to each of the n*n patches in the image
    patch size: w_loc*w_loc
    Inputs: -Patch_orientation: set of the gradient 
                orientation of each patch [n,n,w_loc, w_loc]
            -Patch_gradient: set of the gradient magnitude of each patch
                [n,n,w_loc, w_loc]
            -s smoothing parameter for the gradient magnitude (float)
            -N quantization of the historgam of angles (int)
    Output: principal orientation of each patch [n,n]
    
    '''
    #Assigning angle to histogram bin
    orientation_bin = ((P_orientation+np.pi)*N/np.pi)//2
    #Initialisation
    [H, L, h, l] = P_orientation.shape
    histo = np.zeros((H, L, N))
    histo_weight = np.zeros((H, L, N))
    kern = gaussian_kernel(h,l)
    #Give more weight to patch's center pixels
    P_magnitude = P_magnitude * kern
    
    #Generation of the histogram and weighted histogram
    for i in range(N):
        index_bin_i = orientation_bin == i
        histo[:,:,i] = np.count_nonzero(index_bin_i, (2,3))
        mag_copy = copy(P_magnitude)
        mag_copy[orientation_bin != i]=0
        histo_weight[:,:,i] = np.sum(mag_copy, axis = (2, 3))
        
    #smoothing the histogram by 6 circular convolutions
    for i in range(6):
        histo_weight = cconv(histo_weight, [1/3,1/3,1/3], 2)
        
    #extraction of the reference orientations
    theta = np.zeros((n,n))
    maxi = np.max(histo_weight, 2)
    for i in range(N):
        h = histo_weight[:,:,i]
        h_plus = histo_weight[:,:,(i+1)%N] 
        h_minus = histo_weight[:,:,(i-1)%N]
        d = h == maxi
        theta = np.maximum((360*(i+1/2)/N + 180/N * ((h_minus - h_plus)/(h_minus - 2*h + h_plus)))*d, theta)

    return theta


def cconv(x, h, d):
    """
        Circular convolution along dimension d.
        h should be small and with odd size
        from: https://github.com/gpeyre/numerical-tours/blob/master/python/nt_toolbox/signal.py
    """
    if d == 2:
        # apply to transposed matrix
        return np.transpose(cconv(np.transpose(x), h, 1))
    y = np.zeros(x.shape)
    p = len(h)
    pc = int(round( float((p - 1) / 2 )))
    for i in range(0, p):
        y = y + h[i] * np.roll(x, pc-i, axis=0)
    return y




def assign_orientation_patch(P_orientation, P_magnitude, w, N=100):
    '''
    Assigns the principal orientation to a patch
    
    '''
    #Assigning angle to histogram bin
    orientation_bin = ((P_orientation+np.pi)*N/np.pi)//2
    #Initialisation
    [h, l] = P_orientation.shape
    histo = np.zeros((N, 1))
    histo_weight = np.zeros((N, 1))
    kern = gaussian_kernel(l, h)
    #Give more weight to patch's center pixels
    P_magnitude = P_magnitude * kern
    #Generation of the histogram and weighted histogram
    for i in range(N):
        index_bin_i = orientation_bin == i
        histo[i] = np.count_nonzero(index_bin_i, (0,1))
        mag_copy = copy(P_magnitude)
        mag_copy[orientation_bin != i]=0
        histo_weight[i] = np.sum(mag_copy, axis = (0, 1))
    #smoothing by 6 circular convolutions
    for i in range(6):
        histo_weight = cconv(histo_weight, [1/3,1/3,1/3], 0)
    #extraction of the reference orientations
    theta = 0
    maxi = np.max(histo_weight)
    for i in range(N):
        h = histo_weight[i]
        h_plus = histo_weight[(i+1)%N] 
        h_minus = histo_weight[(i-1)%N]
        d = h == maxi
        theta = np.maximum((360*(i+1/2)/N + 180/N * ((h_minus - h_plus)/(h_minus - 2*h + h_plus)))*d, theta)

    return theta


def get_algae_im(xcoords, ycoords, imref):
    back = np.zeros((60,60))
    xmin = max(xcoords.min() - 5, 0)
    xmax = min(xcoords.max() + 5, imref.shape[0])
    ymin = max(ycoords.min() - 5, 0)
    ymax = min(ycoords.max() + 5, imref.shape[1])
    start = 15
    b = imref[xmin:xmax, ymin:ymax]
    P_orientation = gradient_orientation(b)
    P_magnitude = gradient_magnitude(b)
    theta0 = assign_orientation_patch(P_orientation, P_magnitude, 30)

    v = scipy.ndimage.interpolation.rotate(b,-float(theta0), axes=(1, 0), reshape=False,
                                           output=None, order=3, mode='constant', cval=0, prefilter=True)

    back[start:start + v.shape[0], start: start+v.shape[1]] = v#imref[xmin:xmax, ymin:ymax]
    #back = v
    #print(back.shape)
    return back#, v, theta0


def show_labelled(data, algae):

    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    x_a = data["items_dict"][algae]["x_coords"]
    y_a = data["items_dict"][algae]["y_coords"]
    imref = data["im_ref"]
    folder = data["folder"]
    filename = folder + "/video.tiff"
    p1 = read_image(filename, 250)
    p2 = read_image(filename, 270)
    r = get_ratio(p1,p2)    

    back = np.zeros((60,60))
    quenching = np.zeros((60,60))
    xmin = max(x_a.min() - 5, 0)
    xmax = min(x_a.max() + 5, imref.shape[0])
    ymin = max(y_a.min() - 5, 0)
    ymax = min(y_a.max() + 5, imref.shape[1])
    start = 15


    back[start:start + xmax-xmin, start: start+ymax-ymin] = imref[xmin:xmax, ymin:ymax]
    quenching[start:start + xmax-xmin, start: start+ymax-ymin] = r[xmin:xmax, ymin:ymax]

    axs[0].imshow(back)
    axs[0].set_title("fluorescence")
    axs[1].set_title("qE (250, 270)")
    axs[1].imshow(quenching, vmin = 0, vmax = 0.8)
    return fig, r[xmin:xmax, ymin:ymax]

def get_ratio(p1,p2):
    r = (p1-p2)/p2
    r[r!=r]=0
    r[r>=1] =0
    r[r==0.5] = 0
    return r
def read_image(filename, index):
        vid = imageio.get_reader(filename)
        frame = vid.get_data(index)
        vid.close()
        return frame

####################################################


"""
CONVERT TIFF VIDEO TO MP4
video_samples = []
for folder in glob.glob("."):
    for i, f in enumerate(glob.glob(folder + "/*qE_calib*")):
        print("yo0")
        file_path = f + "/video.tiff"
        if len(glob.glob(f+"/video.mp4"))==0:
            

            video = tiff.imread(file_path)
            writer = imageio.get_writer(f + "/video.mp4", fps=20)
            for i in range(video.shape[0]):
                writer.append_data(video[i])
            writer.close()
            print(file_path)
            video_samples.append(video[240:600,  x1:x2, y1:y2])#200:290, 570:670]
        
        
READ MP4
pic = []
fm_bis = []
fm = []
mini =[]

for folder in glob.glob("."):
    for i, f in enumerate(glob.glob(folder + "/*qE_calib*")):
        print("yo0")
        filename = f + "/video.mp4"
        
        vid = imageio.get_reader(filename,  'ffmpeg')
        fm_bis.append(vid.get_data(270))
        fm.append(vid.get_data(250))
        pic.append(vid.get_data(260))
        mini.append(vid.get_data(269))
        vid.close()


"""


#####################
def fit_biexp(time, trace, ref):
    inds = ref<ref.max()*0.4
    time = time[inds]
    trace = trace[inds]
    ys = mvgavg.mvgavg(trace, 3)
    start = np.argmin(ys[0:100])
    stop = min(start + np.argmax(ys[start:])+30, len(time)-1)
    
    t = time[start:stop] - time[start]
    y = trace[start:stop]
    half = (trace[start] + trace[stop])/2
    xhalf = (start+stop)/2
    #x0 = [0.03, 0.98, 20, 80]
    #lower_bounds = [-1e9]*len(x0)
    #lower_bounds[1] = 0

    #OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
    #                                args = (t, y, sigmoid))
    #parameters_estimated = OptimizeResult.x
    #xhalf = int(parameters_estimated[-1])
    #print("tau sigmoid", parameters_estimated[1])
    #ypred = sigmoid(parameters_estimated, t)
    """
    plt.figure()
    plt.plot(time, trace)
    plt.scatter(time[start], trace[start])
    plt.scatter(time[xhalf], trace[xhalf])
    plt.scatter(time[stop], trace[stop])
    plt.plot(t+time[start], ypred)
    """
    xeph = np.linspace(0, len(y)-1, len(y))
    
    p = np.polyfit(xeph, y, 5)
    #plt.figure()
    #plt.plot(time, trace)
    #plt.plot(time[xeph.astype(int)+start], np.polyval(p, t))
    p[-1] -= half

    roots = np.roots(p)
    roots = roots[np.isreal(roots)]
    root = roots[np.argmin(np.sqrt((roots-(xhalf+start)/2)**2))]
    xhalf = int(root)+start
    
    #plt.scatter(root, half)
    
    x0 = [1,1,1]
    lower_bounds = [-1e9]*len(x0)
    lower_bounds[1] = 0
    t = time[xhalf:stop] - time[xhalf]
    y = trace[xhalf:stop]
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
                                    args = (t, y, exp_decay))    
    
    parameters_estimated = OptimizeResult.x
    #print("tau exp", parameters_estimated[1])
    
    ypred = exp_decay(parameters_estimated, t)
    #plt.plot(t+time[xhalf], ypred)
    
    A, tau, y0 = parameters_estimated
    

    def biexp(params, xdata):
        A0, tau0, A1, y0, delay = params
        return A0*np.exp(-(xdata-delay)/tau0)+A1*np.exp(-(xdata-delay)/tau) + y0
    

    
    x0 = [A, 1, A, y0, 0]
    lower_bounds = [-1e9]*len(x0)
    lower_bounds[1] = 0  
    t = time[start:stop]-time[start]
    y = trace[start:stop]
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
                                    args = (t, y, biexp))        
    parameters_estimated = OptimizeResult.x
    print("tau  biexp", parameters_estimated[1])
    ypred = biexp(parameters_estimated, t)
    #plt.plot(t+time[start], ypred)
    A0, tau0, A1, y0, delay = parameters_estimated
    tau1 = tau
    
    """
    def biexponential(parameters, xdata):
        A0, tau0, A1, y0, delay, tau1 = parameters
        return A0*np.exp(-xdata/tau0) + A1*np.exp(-(xdata-delay)/tau1) + y0
    
    x0 = [1, 1, 1, 1, 1, 1]
    
    
    lower_bounds = [-1e9]*len(x0)
    lower_bounds[1] = 0  
    lower_bounds[-1] = 0
    t = time[start:stop]-time[start]
    y = trace[start:stop]
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
                                    args = (t, y, biexponential))        
    p = OptimizeResult.x
    print("tau  biexp", p[1])
    ypred = biexponential(p, t)
    #plt.plot(time, trace)
    #plt.plot(t + time[start], ypred, "o")
    A0, tau0, A1, y0, delay, tau1 = p
    """
    
    if tau1>tau0:
        tau2 = tau0
        tau0 = tau1
        tau1 = tau2
        
    return trace, time, ypred, t+time[start], A0, tau0, A1, y0, delay, tau1


def fit_exp(xdata, trace, ref):
    x0 = [1,1,1]
    inds = ref<ref.max()*0.4
    time = xdata[inds]
    start = np.argmax(trace[0:2])
    y = trace[inds][start:]
    ys = mvgavg.mvgavg(trace, 3)
    time = time[start:] - time[start]
    lower_bounds = [-1e9]*len(x0)
    lower_bounds[1] = 0
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
                                    args = (time,y , exp_decay_max))
    parameters_estimated = OptimizeResult.x
    ypred = exp_decay(parameters_estimated, time)
    
    tau = parameters_estimated[1]
    duration = time[-1]-time[0]
    convert = duration/len(time)
    if tau >  duration//2: #if too high
            tau =  duration//2
    if tau < 5: #if too low, increase it
            tau = 5
    x0 = parameters_estimated #initial guess: parameters from previous fit
    #second fit
    
    bound = int(tau*5/convert)
    #print(bound)
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
                                    args = (time[0:bound],y[0:bound] , exp_decay))
    ypred = exp_decay_max(parameters_estimated, time)
    
    return trace[inds], xdata[inds]-xdata[start], ypred, time, *parameters_estimated






def fit_monoexp(time, trace, ref):
    inds = ref<ref.max()*0.4
    time = time[inds]
    trace = trace[inds]
    ys = mvgavg.mvgavg(trace, 3)
    start = np.argmin(ys[0:100])
    stop = min(start + np.argmax(ys[start:])+30, len(time)-1)
    
    t = time[start:stop] - time[start]
    y = trace[start:stop]
    half = (trace[start] + trace[stop])/4
    xhalf = (start+stop)/4
    #x0 = [0.03, 0.98, 20, 80]
    #lower_bounds = [-1e9]*len(x0)
    #lower_bounds[1] = 0

    #OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
    #                                args = (t, y, sigmoid))
    #parameters_estimated = OptimizeResult.x
    #xhalf = int(parameters_estimated[-1])
    #print("tau sigmoid", parameters_estimated[1])
    #ypred = sigmoid(parameters_estimated, t)
    """
    plt.figure()
    plt.plot(time, trace)
    plt.scatter(time[start], trace[start])
    plt.scatter(time[xhalf], trace[xhalf])
    plt.scatter(time[stop], trace[stop])
    plt.plot(t+time[start], ypred)
    """
    xeph = np.linspace(start, stop-1, stop-start)
    
    p = np.polyfit(xeph, y, 5)
    #plt.figure()
    #plt.plot(time, trace)
    #plt.plot(time[xeph.astype(int)+start], np.polyval(p, t))
    p[-1] -= half

    roots = np.roots(p)
    roots = roots[np.isreal(roots)]
    roots = roots[(roots>start)*(roots<stop)]
    root = roots[np.argmin(np.sqrt((roots-xhalf)**2))]
    xhalf = int(root)
    
    #plt.scatter(root, half)
    
    x0 = [1,1,1]
    lower_bounds = [-1e9]*len(x0)
    lower_bounds[1] = 0
    t = time[xhalf:stop] - time[xhalf]
    y = trace[xhalf:stop]
    OptimizeResult  = optimize.least_squares(residuals,  x0, bounds = (lower_bounds,1e9),
                                    args = (t, y, exp_decay))    
    
    parameters_estimated = OptimizeResult.x
    #print("tau exp", parameters_estimated[1])
    
    ypred = exp_decay(parameters_estimated, t)
    #plt.plot(t+time[xhalf], ypred)
    
    A0, tau0, y0 = parameters_estimated
    
        
    return trace, time, ypred, t+time[xhalf], A0, tau0, y0



############################


from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd

#https://andrewmourcos.github.io/blog/2019/06/06/PCA.html
#https://jupyter.brynmawr.edu/services/public/dblank/CS371%20Cognitive%20Science/2016-Fall/PCA.ipynb
#https://stackoverflow.com/questions/36232334/plotting-3d-decision-boundary-from-linear-svm
#https://www.kaggle.com/pranathichunduru/svm-for-multiclass-classification

def make_svm(train_data, Y_train_label):

    #Libraries to Build Ensemble Model : Random Forest Classifier 
    # Create the parameter grid based on the results of random search 
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': [ 'poly'], 'C': [1, 10, 100, 1000]}]
    # Performing CV to tune parameters for best SVM fit 
    svm_model = GridSearchCV(SVC(), params_grid, cv=5)
    svm_model.fit(train_data, Y_train_label)

    # View the accuracy score
    print('Best score for training data:', svm_model.best_score_,"\n") 

    # View the best parameters for the model found using grid search
    print('Best C:',svm_model.best_estimator_.C,"\n") 
    print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    final_model = svm_model.best_estimator_
    
    Y_data = final_model.predict(train_data)
    print("training accuracy:", np.sum(Y_data==Y_train_label)/len(Y_data))
    
    
    return final_model

# Constructing a hyperplane using a formula.
#w = final_model.coef_[0]           # w consists of 2 elements
#b = final_model.intercept_[0]      # b consists of 1 element
#x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
#y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
#plt.plot(x_points, y_points)

# The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# Solve for w3 (z)

#z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]
#tmp = np.linspace(-0.5,1.5,30)
#boundary = np.array([tmp, tmp, z(tmp, tmp)])
#boundary = pca.transform(boundary.T)
def make_pca(n_components, train_data, Y_train_label):
    pca = PCA(n_components=n_components)

    pca.fit(train_data)
    X = pca.transform(train_data)
    print("PCA components:")
    plt.matshow(pca.components_)
    plt.figure()
    print("PCA explained variance:")
    plt.semilogy(pca.explained_variance_)
    plt.ylabel("eignevalue")
    plt.xlabel("component")

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=Y_train_label)##
    #plt.plot(boundary[:,0], boundary[:,1], c = 'k')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    return pca


#plt.figure()
#for k in range(0,4):
    #plt.figure()
#    position = (label==2)*(video_number==k)
#    test_data = (np.array([r1,r2,r3, r4, r5]).T)[position]
#    Y_data = final_model.predict(test_data)

#    X2 = pca.transform(test_data)

#    plt.scatter(X2[:,0], X2[:,1], c=Y_data)
#    print(np.sum(Y_data)/len(Y_data))
#    plt.xlabel("PC1")
#    plt.ylabel("PC2")
    #plt.plot(boundary[:,0], boundary[:,1], c = 'k')

#    plt.scatter(-1, -0.4, c="w")
#    plt.scatter(1.5, 0.8, c= 'w')



#image analysis
