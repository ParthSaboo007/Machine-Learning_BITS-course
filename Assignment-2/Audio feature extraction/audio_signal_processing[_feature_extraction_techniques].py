# -*- coding: utf-8 -*-
"""Audio-signal Processing[ Feature Extraction techniques]

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16iSyTQGTTKI0WwMIacix0HksjnV5JMvy
"""

pip install geopandas

# Commented out IPython magic to ensure Python compatibility.
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Map 1 library
import plotly.express as px

# Map 2 libraries
import descartes
import geopandas as gpd # allows spatial operation on geometry types, i.e., operations are performed shapely
from shapely.geometry import Point, Polygon

# Librosa Libraries
""" Python module to analyze audio signals 
    in general but geared more towards music"""
import librosa
import librosa.display
import IPython.display as ipd

import sklearn

import warnings
warnings.filterwarnings('ignore')

from google.colab import files
uploaded = files.upload()

audio_data_amered = 'amered.mp3'
x, sr = librosa.load(audio_data_amered) # sr-> sampling rate,i.e., no. of samples of audio carried per second ( measured in Hz)
# x -> audio time series numpy array
print(type(x), type(sr)) # returns audio-time series as numpy array with a default sampling rate(sr) of 22 kHz mono
print(x.shape,sr)
librosa.load(audio_data_amered, sr=44100) # resampling at 44 kHz

# Playing the audio using IPythn.isplay.Audio

ipd.Audio(audio_data_amered)

audio_data_cangoo = 'cangoo.mp3'
librosa.load(audio_data_cangoo)
ipd.Audio(audio_data_cangoo)

audio_data_haiwoo = 'haiwoo.mp3'
librosa.load(audio_data_haiwoo)
ipd.Audio(audio_data_haiwoo)

audio_data_pingro = 'pingro.mp3'
librosa.load(audio_data_pingro)
ipd.Audio(audio_data_pingro)

audio_data_vesspa = 'vesspa.mp3'
librosa.load(audio_data_vesspa)
ipd.Audio(audio_data_vesspa)

"""### Sound Wave 2d- representation"""

# Visualizzing audio using librosa.display.waveplot

plt.figure(figsize = (15,5))
plt.xlabel("Time")
plt.ylabel("Amplitude")
librosa.display.waveplot(x,sr= sr)

librosa.display.waveplot(librosa.load(audio_data_cangoo)[0],sr= sr)

librosa.display.waveplot(librosa.load(audio_data_haiwoo)[0],sr= sr)

librosa.display.waveplot(librosa.load(audio_data_pingro)[0],sr= sr)

librosa.display.waveplot(librosa.load(audio_data_vesspa)[0],sr= sr)

# numpy nd arrays of audio time signal
audio_amered = librosa.load(audio_data_amered)[0]
audio_cangoo = librosa.load(audio_data_cangoo)[0]
audio_haiwoo = librosa.load(audio_data_haiwoo)[0]
audio_pingro = librosa.load(audio_data_pingro)[0]
audio_vesspa = librosa.load(audio_data_vesspa)[0]

"""###Fourier-Transform

.stft() converts data into short term Fourier transform. STFT converts signals such that we can know the amplitude of the given frequency at a given time. 

Using STFT we can determine the amplitude of various frequencies playing at a given time of an audio signal. 

.specshow is used to display a spectrogram.

"""

# Default FFT window size
n_fft = 2048 # FFT window size
hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

# Short-time Fourier transform (STFT)
D_amered = np.abs(librosa.stft(audio_amered, n_fft = n_fft, hop_length = hop_length))
D_cangoo = np.abs(librosa.stft(audio_cangoo, n_fft = n_fft, hop_length = hop_length))
D_haiwoo = np.abs(librosa.stft(audio_haiwoo, n_fft = n_fft, hop_length = hop_length))
D_pingro = np.abs(librosa.stft(audio_pingro, n_fft = n_fft, hop_length = hop_length))
D_vesspa = np.abs(librosa.stft(audio_vesspa, n_fft = n_fft, hop_length = hop_length))

print('Shape of D object:', np.shape(D_haiwoo))

"""### Spectogram

A visual representation of the spectrum of frequencies of a signal as it varies with time. When applied to an audio signal, spectrograms are sometimes called sonographs, voiceprints, or voicegrams.

 For audio signals, analyzing in the frequency domain is easier than in the time domain.
"""

plt.figure(figsize=(14, 5))
librosa.display.specshow(librosa.amplitude_to_db(D_amered), sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

plt.figure(figsize=(14, 5))
librosa.display.specshow(librosa.amplitude_to_db(D_cangoo), sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

plt.figure(figsize=(14, 5))
librosa.display.specshow(librosa.amplitude_to_db(D_haiwoo), sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

plt.figure(figsize=(14, 5))
librosa.display.specshow(librosa.amplitude_to_db(D_pingro), sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

plt.figure(figsize=(14, 5))
librosa.display.specshow(librosa.amplitude_to_db(D_vesspa), sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()



"""### Feature extraction from audio signal


*   Every audio signal consists of many features. However, we must extract the characteristics that are relevant to the problem we are trying to solve. The process of extracting features to use them for analysis is called feature extraction.

The spectral features (frequency-based features), which are obtained by 
converting the time-based signal into the frequency domain using the Fourier Transform, like fundamental frequency, frequency components, spectral centroid, spectral flux, spectral density, spectral roll-off, etc.

**1] Spectral Centroid**

-> indicates at which frequency the energy of a spectrum is centered upon or  It indicates where the ” center of mass” for a sound is located.
"""

import sklearn

x, sr = librosa.load(audio_data_amered)

"""
.spectral_centroid() will return an array with columns
 equal to the number of frames present in your sample."""
 
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0] 
spectral_centroids.shape

# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

#Plotting the Spectral Centroid along the waveform without normalising
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, spectral_centroids, color='b')

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform without normalising
plt.figure(figsize=(12, 4))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')

"""2.] Spectral Roloff

Is a measure of the shape of the signal. It represents the frequency below which a specified percentage of the total spectral energy (e.g. 85%) lies.

librosa.feature.spectral_rolloff computes the roll off frequency for each frame in a signal
"""

spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
spectral_rolloff[0:100]

# visualization
plt.figure(figsize=(15, 8))
librosa.display.waveplot(x, sr=sr, alpha=0.5)   # alpha denotes range of y-axis
plt.plot(t, normalize(spectral_rolloff), color='r')

"""### 3] Zero Crossing Rate

Tells rate at which the signal changes from positive to negative or back

-> A voice signal oscillates slowly — for example, a 100 Hz signal will cross zero 100 per second — whereas an unvoiced fricative can have 3000 zero crossings per second.

Observation-> It usually has higher values for highly percussive sounds like those in metal and rock. 
"""

zero_amered = librosa.zero_crossings(audio_amered, pad=False)
zero_amered

# Visualization
## Zooming in 

n0 = 1000
n1= 1100
plt.figure(figsize=(15,6))
plt.plot(x[n0:n1])
plt.grid()

# No. of zero crossings in selected frame
sum(librosa.zero_crossings(x[n0:n1], pad=False))



"""### 4] Mel-Frequency Capstral Coefficients [ MFCCs]

Definition: MFCCs are a compact representation of the spectrum(When a waveform is represented by a summation of possibly infinite number of sinusoids) of an audio signal.

->The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) that concisely describe the overall shape of a spectral envelope. 

-> It models the characteristics of the human voice [ Mel scale].

"""

mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)

# displaying mfcc feature vector
mfccs

# Visualization

plt.figure(figsize=(15,6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# y-axis denotes MFCC coefficients



"""## 5] Chroma feature

A chroma feature or vector is typically a 12-element feature vector indicating how much energy of each pitch class, {C, C#, D, D#, E, …, B}, is present in the signal. In short, It provides a robust way to describe a similarity measure between music pieces.

Library used: librosa.feature.chroma_stft
"""

chromagram = librosa.feature.chroma_stft(x, sr=sr)
# displaying obtained feature vectors
chromagram

# visualization
plt.figure(figsize=(15, 6))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

