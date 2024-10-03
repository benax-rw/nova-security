import numpy as np
import soundfile as sf

# Function to generate a sine wave
def generate_tone(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave

# Set parameters for the tone
frequency = 1000  # Frequency in Hertz (1kHz tone)
duration = 0.001    # Duration in seconds
silence_duration = 0.001  # Duration of silence between beeps

# Generate the tone and silence
tone = generate_tone(frequency, duration)
silence = np.zeros(int(44100 * silence_duration))

# Concatenate the tone and silence three times
beep_sequence = np.concatenate([tone, silence, tone, silence, tone])

# Save the sequence as a .wav file
sf.write('alert.wav', beep_sequence, 44100)

print("Alert sound with three beeps generated and saved as 'alert.wav'")
