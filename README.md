# RTL-SDR-CONV-Single-File

The simple radio application with RTL-SDR using DearPyGUI window system. This is the highly modified version of "MyRadio" branch. The demodulator schemes use the time domain convolutions instead of frequency domain FFT calculations used in MyRadio branch. With using DearPyGUI package, the radio has an GUI, that all required parameters can be adjusted. The GUI has an waterfall, 10-Band vumeter screen, sequelch an audio noise reduction knobs, RF noise reduction option and an bias-T option for to use with active antenna amplifier or etc.

The pytran, numpy, scipy, DearPyGUI, pyrtlsdr, sounddevice and pyaudio (used by sounddevice package) packages must be exist in your system for running the radio application. 

Although "single file" words include in the name of the file, the radio application needs extra files, that compiled with pythran for accerelating purposes. I tried too many options
for speeding up the execution of the program, multiprocessing, shared memories etc. This program uses the threading module. The src folder contains the required python files. compile this files using pythran and 
copy the compiled library files to the same location of radio application file. The font files must be in the same location too.

İf all things go well, the radio application screen should be seen as below:

![image](https://github.com/user-attachments/assets/bb40d3f0-9cab-4d12-8089-759f7ccb35df)

