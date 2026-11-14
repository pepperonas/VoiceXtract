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
python3 voicextract.py input [-o OUTPUT] [-m MODEL] [-f {mp3,wav}] [-b BITRATE] [-s SECONDS] [-r]
```

Arguments:

- `input`: Input audio file or directory
- `-o, --output`: Output directory (default: "output")
- `-m, --model`: Model to use (default: "htdemucs")
- `-f, --format`: Output format, `mp3` (default, needs ffmpeg) or `wav`
- `-b, --bitrate`: MP3 bitrate (default: "192k")
- `-s, --segment`: Process in segments of this many seconds (lower memory use)
- `-r, --recursive`: Search directories recursively

When processing a directory, each file gets its own output folder named after
the file; files with the same name (e.g. `a/song.mp3` and `b/song.mp3` with
`-r`) get `song`, `song (2)`, … instead of overwriting each other.

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
print(result)  # {'vocals': 'output_directory/vocals.mp3', 'accompaniment': 'output_directory/accompaniment.mp3'}
```

## Models

VoiceXtract uses Demucs models for audio source separation. The default model is "htdemucs", which offers a good balance
between quality and performance.

## Output

For each processed audio file, two output files are created:

- `vocals.mp3`: Contains only the extracted vocals
- `accompaniment.mp3`: Contains the instrumental backing track

With `-f wav` (or when ffmpeg is missing or fails) the files are `.wav`; the
returned paths always point to the files that were actually written.

## Tests

31 unit tests cover everything that does not need the neural network:
segmenting long audio, the crossfade that stitches segments back together
(it must reconstruct the input exactly), clipping and channel order when
saving, the MP3 → WAV fallback, the vocals/accompaniment split (with a fake
model), file discovery and unique output folders. torch and demucs are not
installed for the tests — `tests/conftest.py` provides a numpy-backed stand-in
for the few tensor operations involved.

```bash
python3 -m venv .venv-test
.venv-test/bin/pip install -r requirements-dev.txt
.venv-test/bin/python -m pytest -q tests
```

GitHub Actions runs the suite on every push (`.github/workflows/tests.yml`).

## License

[MIT License](LICENSE)