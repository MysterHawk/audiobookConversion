"""Performs automatic speed edits to audio books.

Example usage:

Assuming you have an audiobook book.aax on your Desktop:

1. Convert it to wav:
ffmpeg -i ~/Desktop/book.aax ~/Desktop/book.wav

2. Adjust the speed:
python fastbook.py \
--audio_path=~/Desktop/book.wav \
--output_path=~/Desktop/book-fast.wav \
--loud_speed=3 \
--quiet_speed=6
"""

import math
import os
import re
import shlex
import shutil
import subprocess
from tqdm import tqdm
import uuid

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter

import fire
import numpy as np
from scipy.io import wavfile

DEFAULT_FPS = 60
DEFAULT_SAMPLERATE = 44100
SILENCE_THRESHOLD = 0.030


def get_max_volume(samples):
  return np.max(np.abs(samples))

def extract_noise_events(samples, metadata):
  num_samples = samples.shape[0]
  max_volume = get_max_volume(samples)

  fps = metadata['fps']
  sample_rate = metadata['sample_rate']
  samples_per_frame = sample_rate / fps
  num_frames = int(math.ceil(num_samples / samples_per_frame))

  # Determine loud frames
  frame_is_loud = np.zeros((num_frames))
  loud_frame_indexes = []
  current_region_start = 0
  regions = []
  for frame_index in range(num_frames):
    frame_margin = 2
    start_frame_index = max(0, frame_index - frame_margin)
    end_frame_index = min(num_frames - 1, frame_index + frame_margin)
    start_sample_index = int(start_frame_index * samples_per_frame)
    end_sample_index = int((end_frame_index + 1) * samples_per_frame)
    frame_samples = samples[start_sample_index:end_sample_index]

    frame_max_volume = get_max_volume(frame_samples)
    if frame_max_volume / max_volume > SILENCE_THRESHOLD:
      frame_is_loud[frame_index] = 1
      loud_frame_indexes.append(frame_index)

    if frame_index > 0 and frame_is_loud[frame_index] != frame_is_loud[frame_index - 1]:
      regions.append((current_region_start, frame_index, frame_is_loud[frame_index - 1]))
      current_region_start = frame_index

  # Include the final region
  if current_region_start < num_frames:
    regions.append((current_region_start, num_frames, frame_is_loud[num_frames - 1]))
  return regions

def adjust_audio_speed(samples, sample_rate, speed, temp_dir):
  tmp1 = os.path.join(temp_dir, 'tmp1.wav')
  tmp2 = os.path.join(temp_dir, 'tmp2.wav')
  wavfile.write(tmp1, int(sample_rate), samples)
  with WavReader(tmp1) as reader:
    with WavWriter(tmp2, reader.channels, reader.samplerate) as writer:
      vocoder = phasevocoder(reader.channels, speed=speed)
      vocoder.run(reader, writer)
  _, output_samples = wavfile.read(tmp2)
  return output_samples

def get_region_samples(samples, metadata, start_frame_index, end_frame_index):
  fps = metadata['fps']
  sample_rate = metadata['sample_rate']
  samples_per_frame = sample_rate / fps

  start_sample_index = int(start_frame_index * samples_per_frame)
  end_sample_index = int(end_frame_index * samples_per_frame)
  return samples[start_sample_index:end_sample_index]

def compress_quiet_regions(samples, metadata, regions, temp_dir, loud_speed, quiet_speed):
  sample_rate = metadata['sample_rate']
  fps = metadata['fps']
  samples_per_frame = sample_rate / fps
  output_samples = np.zeros((0, samples.shape[1]))
  
  for region in regions:
    region_start_frame, region_end_frame, region_is_loud = region
    region_samples = get_region_samples(samples, metadata, region_start_frame, region_end_frame)
    if not region_is_loud:
      speed = quiet_speed
      region_samples = adjust_audio_speed(region_samples, sample_rate, speed=speed, temp_dir=temp_dir)
    else:
      speed = loud_speed
      region_samples = adjust_audio_speed(region_samples, sample_rate, speed=speed, temp_dir=temp_dir)
    region_samples = np.float64(region_samples)

    # Fade or zero-out the region samples.
    num_region_samples = region_samples.shape[0]
    num_fade_samples = int(0.01 * sample_rate)
    if num_region_samples < num_fade_samples:
      # Region too short; remove its audio.
      region_samples[:] = 0.0
    else:
      # Fade the region in and out.
      fade_mask = np.arange(num_fade_samples) / num_fade_samples
      fade_mask = np.repeat(fade_mask[:, np.newaxis], 2, axis=1)
      region_samples[:num_fade_samples] *= fade_mask
      region_samples[-num_fade_samples:] *= 1 - fade_mask

    # Save the region samples.
    output_start_frame = int(math.ceil(output_samples.shape[0] / samples_per_frame))
    output_samples = np.concatenate((output_samples, region_samples))
    output_end_frame = int(math.ceil(output_samples.shape[0] / samples_per_frame))

  return output_samples

def run(audio_path, output_path, loud_speed=1, quiet_speed=5):
  temp_dir = f'TEMP-{uuid.uuid4()}'
  os.makedirs(temp_dir)
  
  audio_path = os.path.expanduser(audio_path)
  output_path = os.path.expanduser(output_path)

  sample_rate, samples = wavfile.read(audio_path)
  metadata = {
      'fps': DEFAULT_FPS,
      'sample_rate': sample_rate,
  }
  regions = extract_noise_events(samples, metadata)
  output_samples = compress_quiet_regions(samples, metadata, regions, temp_dir, loud_speed, quiet_speed)
  output_samples /= get_max_volume(samples)

  wavfile.write(output_path, sample_rate, output_samples)
  shutil.rmtree(temp_dir)


if __name__ == '__main__':
  fire.Fire(run)
