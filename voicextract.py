# voicextract.py
import os
import torch
import numpy as np
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile
import argparse
from pathlib import Path
import tqdm
import soundfile as sf


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
        print(f"Modell geladen: {model_name}, Quellen: {self.model.sources}")
        self.model.to(self.device)

    def save_audio(self, wav, path, sample_rate):
        """
        Speichert Audio-Daten als WAV-Datei
        """
        # Sicherstellen, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Clip-Werte zwischen -1 und 1
        wav = np.clip(wav, -1, 1)

        # Speichern mit soundfile
        sf.write(path, wav.T, sample_rate)

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

        try:
            # Audio laden - AudioFile.read gibt bereits einen Tensor zurück in neueren Versionen
            wav = AudioFile(input_file).read(channels=self.model.audio_channels, samplerate=self.model.samplerate)

            # Prüfen, ob wav bereits ein Tensor ist
            if not isinstance(wav, torch.Tensor):
                wav = torch.tensor(wav)

            # Sicherstellen, dass Audio das richtige Format hat (batch, channels, time)
            if wav.dim() == 2:  # Wenn es (channels, time) ist
                wav = wav.unsqueeze(0)  # Zu (1, channels, time) erweitern
            elif wav.dim() == 1:  # Wenn es nur (time) ist
                wav = wav.unsqueeze(0).unsqueeze(0)  # Zu (1, 1, time) erweitern

            # Zum richtigen Gerät verschieben
            wav = wav.to(self.device)

            print(f"Audio-Shape: {wav.shape}")

            # Verwende apply_model statt separate, da BagOfModels keine separate-Methode hat
            with torch.no_grad():
                sources = apply_model(self.model, wav)

            # Pfade definieren
            vocal_path = os.path.join(output_dir, 'vocals.wav')
            accompaniment_path = os.path.join(output_dir, 'accompaniment.wav')

            # Die Ausgabe von apply_model hat die Form (batch, source, channels, time)
            stem_names = self.model.sources
            print(f"Verfügbare Stems: {stem_names}")

            vocal_idx = stem_names.index('vocals') if 'vocals' in stem_names else -1

            if vocal_idx != -1:
                # Vocals speichern
                vocals = sources[:, vocal_idx].cpu().numpy()
                self.save_audio(vocals[0], vocal_path, self.model.samplerate)

                # Begleitung erstellen (alle Quellen außer vocals)
                accompaniment = torch.zeros_like(sources[:, 0])
                for idx, name in enumerate(stem_names):
                    if idx != vocal_idx:
                        accompaniment += sources[:, idx]

                self.save_audio(accompaniment[0].cpu().numpy(), accompaniment_path, self.model.samplerate)

                return {
                    'vocals': vocal_path,
                    'accompaniment': accompaniment_path
                }
            else:
                print("Warnung: Keine Vocals-Stem gefunden in den verfügbaren Stems!")
                return {}

        except Exception as e:
            print(f"Fehler bei der Extraktion: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

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


if __name__ == "__main__":
    main()