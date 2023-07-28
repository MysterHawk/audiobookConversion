import os
import subprocess
import argparse
from tqdm import tqdm

# Define the arguments for the script
parser = argparse.ArgumentParser(description='Process audio files in a folder and apply the fastbook.py script')
parser.add_argument('folder_path', type=str, help='Path to the folder containing the audio files')
args = parser.parse_args()

audio_files = [f for f in os.listdir(args.folder_path) if f.lower().endswith('.wav')]

# Loop through each file in the folder
for filename in tqdm(audio_files, desc='Processing audio files'):
    print ('\n Audio file: %s' % filename)
    audio_path = os.path.join(args.folder_path, filename)
    output_path = os.path.join(args.folder_path, f'{os.path.splitext(filename)[0]}-fast.wav')
    # Call the fastbook.py script using subprocess
    subprocess.run(['python', 'fastbook.py', '--audio_path', audio_path, '--output_path', output_path, '--loud_speed', '1', '--quiet_speed', '5'])
