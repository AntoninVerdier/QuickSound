# QuickSound

Quicksound is a simple yet powerful modular tool to build raw sounds files in .wav format. 

## General architecture

Each Sound is a python-object encapsulating diverse meta-information and a raw signal. Adding two Sound object together (by concatenation or overlapping, depending on needs) will enable creation of increasingly complex sound. 

Several methods are available to create the most basic sounds.

## Examples

Creation of a simple pure tone at 6kHz with a duration of 800ms. Amplitude is in dB.
```python
  s = Sound(samplerate=192000, amplitude=70)
  s.pure_tone(6e3, duration=800)
```
To save the corresponding sound
```python
s.save_wav(name='Pure_tone', path='../Samples/')
```
A more complex sound can be created by concatenating and overlaping already existing Sounds. However, they need to have the same samplerate.
```python
s = Sound(samplerate=192000, amplitude=70)
s.pure_tone(6e3, duration=800)

d = Sound(samplerate=192000, amplitude=60)
d.delay(duration=1200)

newSound = s + d

am = Sound(samplerate=192000, amplitude=80)
am.amplitude_modulation(1e3, 50, duration=2000)

OverlappingSounds = newSound * am
OverlappingSounds.save_wav(name='Complex_Sound', path='../Samples/')
```


## Dependencies

- Numpy
- Scipy
- Sklearn (to be removed)

