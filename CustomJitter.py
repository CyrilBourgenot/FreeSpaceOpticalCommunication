import numpy as np
import pandas as pd
import pylab

def create_custom_Jitter(dim,seed = None):
    #### This function  creates a jitter type data from random number, by doing a cummulative sum on a random array to smooth the data and 
    #### to add some low frequency variation, then by doing rolling the array by an offset of "lag", and averaging it with the nominal to
    #### remove the final noise. 
    #### I have added a filter of the lowest frequency as well, to make sure that the just remains around the value zero, and does not wander
    #### too far.

    #### This way we have a jitter of frequency mox 1Hz (basically the interval of "dim"), and the jitter can be reproduced using the seed 
    #### parameter. 
     
    lag = 60
    filter = 3

    dimTime = dim + lag #because the cummulative sum replace the first pixels (number equivalent to lag), as NAN number, I extend the array by "lag".
    
    ### here we create the random numbers    
   
    np.random.seed(seed)
    random_normals = np.random.randn(dimTime-1)
    
    # the time serie is produced with random number. To remove the high frequency and smooth this up, the first stage is to use a cummulative sum. Thisis easly done with Panda
    cumulatedSum = pd.DataFrame(random_normals).cumsum()
    # Then to further smooth it up, a rolling average of size LAG is taken
    res = cumulatedSum.rolling(lag).mean()
    
    # convert pandas to numpy
    res.to_numpy()
    res = res[0]
    res = res[~np.isnan(res)] # remove the NAN 
   
    # Here we filter the low frequency - so we remove +/-3 pixels around the central peak of the FFT, then convert it back to a time series
    FFT = np.fft.fft(res)
    arg = np.real(np.sqrt(np.real(FFT)**2 + np.imag(FFT)**2))
    dimarg = arg.shape[0]
    argshifted = np.fft.fftshift(arg)
    argShiftedFiltered = argshifted.copy()
    
    argShiftedFiltered[dimarg//2-filter:dimarg//2+filter+1] = 0.
    phase = np.arctan2(np.imag(FFT),np.real(FFT))
    invFFTG = np.real(np.fft.ifft(np.fft.fftshift(argShiftedFiltered) * np.exp(1j*phase)))
    return invFFTG

if __name__ == '__main__':

    dim = 4021
    timeseries = np.arange(dim)
    resx = create_custom_Jitter(dim) # good to use seed = 100
    resy = create_custom_Jitter(dim) # good to use seed = 8

    maxResx = max(max(resx),abs(min(resx)))
    maxResy = max(max(resy),abs(min(resy)))
    # normalise to 0.3 degree
    resx *= 0.3 / maxResx
    resy *= 0.3 / maxResy
    pylab.plot(timeseries,resx, label = 'axis x')
    pylab.plot(timeseries,resy, label = 'axis y')
    pylab.xlabel('time in sec')
    pylab.ylabel('Angle in degrees')
    pylab.legend()
    pylab.show()

    pylab.scatter(resx,resy,marker = '.',c = timeseries)
    pylab.xlabel('Angle along x in degrees')
    pylab.ylabel('Angle along y in degrees')
    pylab.title('Angular jitter experienced by the satellite')
    pylab.show()
