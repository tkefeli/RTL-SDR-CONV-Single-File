import numpy as np

def my_shift(data):
    index = int(len(data)//2)
    return np.hstack([data[index:], data[:index]])

def hann_window(size):    
    return (0.5 - 0.5*np.cos(2*np.pi*np.arange(size)/size))

# pythran export spectrum_partial_fft(complex128[], int, bool, int)
def spectrum_partial_fft(signal, rate, window, fft_size):
    num_rows = int(len(signal)/fft_size)
    spectrogram = np.zeros((num_rows, fft_size)).astype(np.complex128)
    window = hann_window(fft_size) if window else np.ones(fft_size)
    for i in range(num_rows): # Bu daha kısa sürer. 
        spectrogram[i,:] = np.fft.fft(signal[i*fft_size:(i+1)*fft_size]*window)
    spectrum = my_shift(20*np.log10(np.mean(abs(spectrogram/fft_size), axis=0))) 
    return spectrum

# pythran export spectrum_partial_fft_real(float[], int, bool, int)
def spectrum_partial_fft_real(signal, rate, window, fft_size):
    size = len(signal)
    if size < fft_size: signal = np.hstack([signal, np.zeros(fft_size-size)])
    num_rows = int(len(signal)/fft_size)
    spectrogram = np.zeros((num_rows, fft_size)).astype(np.complex128)
    window = hann_window(fft_size) if window else np.ones(fft_size)
    for i in range(num_rows): # Bu daha kısa sürer. 
        spectrogram[i,:] = np.fft.fft(signal[i*fft_size:(i+1)*fft_size]*window)
    spectrum = 20*np.log10(np.mean(abs(spectrogram/fft_size), axis=0)) 
    return spectrum[:int(fft_size//2)]