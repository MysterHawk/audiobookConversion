import os
import argparse
import subprocess
from tqdm import tqdm

def convert_to_opus(folder_path):
    # Get a list of all audio files in the folder
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]

    # Loop through each audio file and convert it to OPUS
    for audio_file in tqdm(audio_files, desc="Converting to OPUS"):
        input_file = os.path.join(folder_path, audio_file)
        output_file = os.path.join(folder_path, os.path.splitext(audio_file)[0] + '.opus')
        cmd = f'ffmpeg -i "{input_file}" -c:a libopus -b:a 48k "{output_file}"'
        subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert all WAV audio files in a folder to OPUS format.')
    parser.add_argument('folder_path', help='The path to the folder containing the audio files.')
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print('Error: folder path does not exist.')
        exit()

    convert_to_opus(args.folder_path)