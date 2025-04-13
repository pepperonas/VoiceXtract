# VoiceXtract

A Python tool for extracting vocals from music files using state-of-the-art audio source separation with the Demucs
library.

## Features

- Extract vocals from audio files (MP3, WAV, FLAC, OGG, M4A)
- Process individual files or entire directories
- Support for recursive directory scanning
- Clean separation of vocals and accompaniment tracks

## Requirements

- Python 3.x
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/pepperonas/VoiceXtract.git
   cd VoiceXtract
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

```
python3 voicextract.py input [-o OUTPUT] [-m MODEL] [-r]
```

Arguments:

- `input`: Input audio file or directory
- `-o, --output`: Output directory (default: "output")
- `-m, --model`: Model to use (default: "htdemucs")
- `-r, --recursive`: Search directories recursively

### Examples

Process a single file:

```
python3 voicextract.py song.mp3
```

Process a directory:

```
python3 voicextract.py music_folder -o extracted_vocals
```

Process a directory recursively:

```
python3 voicextract.py music_folder -o extracted_vocals -r
```

### Python API

```python
from voicextract import VocalExtractor

extractor = VocalExtractor()
result = extractor.extract_vocals("song.mp3", "output_directory")
print(result)  # {'vocals': 'output_directory/vocals.wav', 'accompaniment': 'output_directory/accompaniment.wav'}
```

## Models

VoiceXtract uses Demucs models for audio source separation. The default model is "htdemucs", which offers a good balance
between quality and performance.

## Output

For each processed audio file, two output files are created:

- `vocals.wav`: Contains only the extracted vocals
- `accompaniment.wav`: Contains the instrumental backing track

## License

[MIT License](LICENSE)