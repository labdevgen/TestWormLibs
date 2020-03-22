from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import numpy as np

def fit_linear_regression(data, resolution,
                          starting_npoints = 5,
                          step = 1,
                          crop_min = -1.75,
                          crop_max = -0.05,
                          max_plot_dist=50000000):

    maxdist = min([len(i) for i in data.values()])
    distances = np.log(np.arange(maxdist)*resolution+1)
    coeffs = []
    regression = LinearRegression()

    assert maxdist-starting_npoints-1 > 1
    starts = range(1,maxdist-starting_npoints-1, step)

    plot_distances = []
    local_average = []
    for st in starts:
        npoints = min(starting_npoints + (st*resolution)//50000, 5000000 // resolution)
        end = st + npoints
        if end >= len(distances):
            break
        X = distances[st:end].reshape(end-st,1)
        curr_coefs = []
        for chr in data:
            Y = np.log(data[chr][st:end].reshape(end-st,1))
            reg = regression.fit(X,Y)
            curr_coefs.append(reg.coef_[0][0])
        #curr_coef = np.median(curr_coefs)
        curr_coef = np.average(curr_coefs)

        if (st*resolution >= max_plot_dist):
            break

        # check that curr coef is not too different from last coeff
        if curr_coef < crop_min:
            coeffs.append(crop_min)
            plot_distances.append(distances[st])
            continue
        elif curr_coef > crop_max:
            coeffs.append(crop_max)
            plot_distances.append(distances[st])
            continue


        if (st*resolution > 1000000 and len(coeffs) > 5) and False:
            # check diference
            av = np.average(coeffs[-4:-1])
            threashold = abs(av)/2
            if abs(curr_coef - av) > threashold or len(local_average)>0: # diff is too high, probably outlayer
                local_average.append(curr_coef)
                if np.median(local_average) > 0 or np.median(local_average) < -2:
                    break
                elif abs(np.median(local_average) - av) <= threashold:
                    coeffs.append(np.median(local_average))
                    plot_distances.append(distances[st])
                    local_average = []
                continue

        coeffs.append(curr_coef)
        plot_distances.append(distances[st])

    return np.exp(plot_distances), np.array(coeffs)

def fit_delta(data, npoints = 20, step = 2):
    maxdist = min([len(i) for i in data.values()])
    distances = np.log(np.arange(maxdist)*resolution+1)
    results = []

    assert maxdist-npoints-1 > 1
    starts = range(1,maxdist-npoints*2, step)

    plot_distances = []
    for st in starts:
        curr_coefs=[]
        for chr in data:
            delta = np.average([(np.log(data[chr][i])-np.log(data[chr][i+npoints]))/np.log(data[chr][i]) for i in range(st,st+npoints)])
            curr_coefs.append(delta)
        curr_coef = np.average(curr_coefs)
        if curr_coef < 0:
            results.append(abs(curr_coef))
            plot_distances.append(distances[st])

    return np.exp(plot_distances), np.array(results)

def plot_ps(data, resolution, maxdist=None):
    # maxdist = 20000000 // resolution
    if maxdist is None:
        maxdist = min([len(i) for i in data.values()])
    else:
        maxdist = min([len(i) for i in data.values()]+[maxdist // resolution])
    print (maxdist)
    expected = data[list(data)[0]]
    for e in data.values():
        if len(e) > len(expected):
            expected = e
    distances = np.arange(1,maxdist) * resolution + 1
    return distances, np.log(expected[1:len(distances)+1]) / np.log(10)

def fit_power_low (data, npoints = 20, step = 2):
    def power_low(x, a, b):
        return  (x ** a) * b

    expected = data[list(data)[0]]
    for e in data.values():
        if len(e) > len(expected):
            expected = e

    # expected = expected / sum(expected)
    distances = np.arange(len(expected))*resolution
    coeffs = []
    # TODO think about this "20" constant
    starts = range(4,len(distances)-10,step)
    plot_distances = []
    for st in starts:
        end = st + npoints
        p0 = [-1., expected[st] * distances[st]]
        try:
            popt, pcov = curve_fit(power_low,
                               distances[st:end],
                               expected[st:end],
                               p0=p0,
                               maxfev = 1000)
        except RuntimeError: #scippy couldn't fit curve
            continue
        if 0 > popt[0] > -2:
            coeffs.append(popt[0])
            plot_distances.append(distances[st])

        # uncomment to draw fit of the curve
        # predicted = power_low(distances[st:end], *popt)
        # plt.loglog(distances[st:end], predicted)
    return plot_distances, np.array(coeffs)
