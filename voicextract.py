# voicextract.py
import os
import torch
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio
import argparse
from pathlib import Path
import tqdm


class VocalExtractor:
    def __init__(self, model_name="htdemucs"):
        """
        Initialisiert den Vocal Extractor mit demucs

        Args:
            model_name: Name des zu verwendenden demucs Modells
                        (htdemucs ist ein gutes Standard-Modell)
        """
        self.model = get_model(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Verwende Gerät: {self.device}")
        self.model.to(self.device)

    def extract_vocals(self, input_file, output_dir):
        """
        Extrahiert Vocals aus einer Audiodatei

        Args:
            input_file: Pfad zur Eingabe-Audiodatei
            output_dir: Verzeichnis für die Ausgabedateien

        Returns:
            Dictionary mit Pfaden zu den extrahierten Dateien
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Audio laden
        audio = AudioFile(input_file).read(samplerate=self.model.samplerate)
        # Auf richtiges Format bringen (batch, channels, time)
        audio = torch.tensor(audio, device=self.device).unsqueeze(0)

        # Separation durchführen
        with torch.no_grad():
            sources = self.model.forward(audio)

        # Sources transformieren [batch, source, channels, time]
        sources = sources.cpu().numpy()

        # Pfade definieren
        vocal_path = os.path.join(output_dir, 'vocals.wav')
        accompaniment_path = os.path.join(output_dir, 'accompaniment.wav')

        # Indizes für demucs sources: 0=drums, 1=bass, 2=other, 3=vocals
        vocal_idx = 3  # Index für vocals

        # Vocals speichern
        save_audio(sources[0, vocal_idx], vocal_path, self.model.samplerate)

        # Begleitung (alles außer vocals) zusammenmischen
        # Nehme drums (0), bass (1) und other (2) und summiere
        accompaniment = sources[0, :vocal_idx].sum(axis=0)
        save_audio(accompaniment, accompaniment_path, self.model.samplerate)

        return {
            'vocals': vocal_path,
            'accompaniment': accompaniment_path
        }

    def batch_process(self, input_files, output_dir):
        """
        Verarbeitet mehrere Dateien im Batch-Modus

        Args:
            input_files: Liste von Eingabedateipfaden
            output_dir: Basisverzeichnis für die Ausgabe

        Returns:
            Dictionary mit Eingabedateien als Schlüssel und Ausgabepfaden als Werten
        """
        results = {}
        for input_file in tqdm.tqdm(input_files, desc="Verarbeite Dateien"):
            file_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0])
            results[input_file] = self.extract_vocals(input_file, file_output_dir)

        return results


# cli.py (Kommandozeilen-Schnittstelle)
def main():
    parser = argparse.ArgumentParser(description='VoiceXtract - Extrahiere Vocals aus Musikdateien')
    parser.add_argument('input', help='Eingabedatei oder Verzeichnis mit Audiodateien')
    parser.add_argument('-o', '--output', default='output', help='Ausgabeverzeichnis (Standard: output)')
    parser.add_argument('-m', '--model', default='htdemucs',
                        help='Name des zu verwendenden Modells (Standard: htdemucs)')
    parser.add_argument('-r', '--recursive', action='store_true', help='Verzeichnisse rekursiv durchsuchen')

    args = parser.parse_args()

    extractor = VocalExtractor(model_name=args.model)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        # Einzelne Datei verarbeiten
        print(f"Verarbeite Datei: {input_path}")
        result = extractor.extract_vocals(str(input_path), str(output_dir))
        print(f"Extrahierte Dateien: {result}")
    else:
        # Verzeichnis verarbeiten
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']

        if args.recursive:
            audio_files = [str(f) for f in input_path.glob('**/*') if
                           f.is_file() and f.suffix.lower() in audio_extensions]
        else:
            audio_files = [str(f) for f in input_path.glob('*') if f.is_file() and f.suffix.lower() in audio_extensions]

        if not audio_files:
            print(f"Keine Audiodateien in {input_path} gefunden.")
            return

        print(f"Verarbeite {len(audio_files)} Audiodateien...")
        results = extractor.batch_process(audio_files, str(output_dir))
        print(f"Verarbeitung abgeschlossen. Ausgabe in: {output_dir}")