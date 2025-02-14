import numpy as np 

def hann_window(size):    
    return (0.5 - 0.5*np.cos(2*np.pi*np.arange(size)/size))

def digitize(n, base):
    return (round(n/base))*base    

def myselect(data, level, base):
    data = [item if item>level else base for item in data]
    return np.array(data)

def sine(A, f, phi, dc, n, rate):
    return np.sin(2*np.pi*f*np.arange(n)/rate+phi*np.pi/180)+dc

def cosine(A, f, phi, dc, n, rate):
    return np.cos(2*np.pi*f*np.arange(n)/rate+phi*np.pi/180)+dc

# pythran export calculate_vumeter_data(float64[], int, bool, int, int, int)
def calculate_vumeter_data(signal, rate, window, fft_size, offset, base):
    size = len(signal)
    if size < fft_size: signal = np.hstack([signal, np.zeros(fft_size-size)])
    num_rows = int(len(signal)/fft_size)
    spectrogram = np.zeros((num_rows, fft_size)).astype(np.complex128)
    window = hann_window(fft_size) if window else np.ones(fft_size)
    for i in range(num_rows): # Bu daha kısa sürer. 
        spectrogram[i,:] = np.fft.fft(signal[i*fft_size:(i+1)*fft_size]*window)
    spectrum = 20*np.log10(np.mean(abs(spectrogram/fft_size), axis=0)) 
    spectrum[:int(fft_size//2)]
    data = np.array([spectrum[2**i] for i in range(10)]) + offset
    levels = np.array([digitize(item, base) for item in data])
    levels = myselect(levels, base, base)
    return levels
    
# pythran export calculate_vumeter_data_lock_in(float64[], int, bool, int, int, int)
def calculate_vumeter_data_lock_in(signal, arate, window, fft_size, offset, base):    
    amplitudes = []
    size = len(signal)
    freqs = np.array([30, 60, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    window = hann_window(size) if window else np.ones(size)
    for freq in freqs:
        re = np.mean(window*signal*sine(1, freq, 0, 0, size, arate))
        im = np.mean(window*signal*cosine(1, freq, 0, 0, size, arate))
        amplitude = (re**2+im**2)**0.5
        amplitudes.append(amplitude)
    amplitudes = np.array(amplitudes)
    levels = 20*np.log10(amplitudes) + offset
    levels = np.array([digitize(item, base) for item in levels])
    levels =  myselect(levels, base, base)
    return levels
