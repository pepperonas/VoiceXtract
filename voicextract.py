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
import tempfile
import subprocess


class VocalExtractor:
    def __init__(self, model_name="htdemucs", segment_size=None):
        """
        Initialisiert den Vocal Extractor mit demucs

        Args:
            model_name: Name des zu verwendenden demucs Modells
                        (htdemucs ist ein gutes Standard-Modell)
            segment_size: Größe der Audio-Segmente für Chunk-Verarbeitung (in Sekunden)
                          None = gesamte Datei wird auf einmal verarbeitet
        """
        self.model = get_model(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.segment_size = segment_size
        print(f"Verwende Gerät: {self.device}")
        print(f"Modell geladen: {model_name}, Quellen: {self.model.sources}")
        self.model.to(self.device)

    def save_audio(self, wav, path, sample_rate, format="mp3", bitrate="192k"):
        """
        Speichert Audio-Daten als MP3- oder WAV-Datei mit ffmpeg

        Args:
            wav: Audio-Daten als NumPy-Array
            path: Ausgabepfad
            sample_rate: Abtastrate
            format: Ausgabeformat ("mp3" oder "wav")
            bitrate: Bitrate für MP3-Kompression
        """
        # Sicherstellen, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Clip-Werte zwischen -1 und 1
        wav = np.clip(wav, -1, 1)

        # Temporäre WAV-Datei
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        # WAV speichern (mit Transposition für richtige Kanalreihenfolge)
        sf.write(temp_wav_path, wav.T, sample_rate)

        if format.lower() == "mp3":
            try:
                # Mit ffmpeg zu MP3 konvertieren
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_wav_path,
                    "-b:a", bitrate, path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"Fehler beim Konvertieren zu MP3: {e}")
                print(f"ffmpeg Fehler: {e.stderr.decode('utf-8') if e.stderr else 'Unbekannt'}")
                # Fallback: Speichere als WAV wenn ffmpeg fehlschlägt
                import shutil
                path = os.path.splitext(path)[0] + ".wav"
                print(f"Speichere stattdessen als WAV: {path}")
                shutil.copy(temp_wav_path, path)
            except FileNotFoundError:
                print("ffmpeg nicht gefunden. Bitte installieren Sie ffmpeg.")
                # Fallback: Speichere als WAV
                import shutil
                path = os.path.splitext(path)[0] + ".wav"
                print(f"Speichere stattdessen als WAV: {path}")
                shutil.copy(temp_wav_path, path)
        else:
            # Einfach die WAV-Datei verschieben
            import shutil
            shutil.copy(temp_wav_path, path)

        # Temporäre WAV-Datei löschen
        try:
            os.unlink(temp_wav_path)
        except Exception as e:
            print(f"Warnung: Konnte temporäre Datei nicht löschen: {e}")

    def extract_vocals(self, input_file, output_dir, format="mp3", bitrate="192k"):
        """
        Extrahiert Vocals aus einer Audiodatei

        Args:
            input_file: Pfad zur Eingabe-Audiodatei
            output_dir: Verzeichnis für die Ausgabedateien
            format: Ausgabeformat ("mp3" oder "wav")
            bitrate: Bitrate für MP3-Kompression

        Returns:
            Dictionary mit Pfaden zu den extrahierten Dateien
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # Audio laden
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

            # Pfade definieren (mit richtiger Dateiendung)
            extension = f".{format.lower()}"
            vocal_path = os.path.join(output_dir, f'vocals{extension}')
            accompaniment_path = os.path.join(output_dir, f'accompaniment{extension}')

            # Segmentierte Verarbeitung oder gesamte Datei auf einmal
            if self.segment_size is not None and wav.shape[2] > self.segment_size * self.model.samplerate:
                # Verarbeite Audio in Segmenten
                segment_samples = int(self.segment_size * self.model.samplerate)
                overlap_samples = segment_samples // 4  # 25% Überlappung

                # Initialisiere Ausgabe-Tensoren
                all_vocals = []
                all_accompaniment = []

                # Berechne Anzahl der Segmente
                total_samples = wav.shape[2]
                segments = []

                # Erstelle überlappende Segmente
                start = 0
                while start < total_samples:
                    end = min(start + segment_samples, total_samples)
                    segments.append((start, end))
                    start += segment_samples - overlap_samples

                print(f"Audio wird in {len(segments)} Segmenten verarbeitet")

                # Verarbeite jedes Segment
                for i, (start, end) in enumerate(tqdm.tqdm(segments, desc="Verarbeite Segmente")):
                    segment = wav[:, :, start:end]

                    # Trenne Audio
                    with torch.no_grad():
                        segment_sources = apply_model(self.model, segment)

                    # Indizes für demucs sources
                    stem_names = self.model.sources
                    vocal_idx = stem_names.index('vocals')

                    # Vocals extrahieren
                    segment_vocals = segment_sources[:, vocal_idx]

                    # Begleitung erstellen
                    segment_accompaniment = torch.zeros_like(segment_sources[:, 0])
                    for idx, name in enumerate(stem_names):
                        if idx != vocal_idx:
                            segment_accompaniment += segment_sources[:, idx]

                    # Speichern der Segmente in Listen
                    all_vocals.append(segment_vocals.cpu())
                    all_accompaniment.append(segment_accompaniment.cpu())

                # Zusammenfügen der Segmente mit Überblendung
                vocal_output = self.crossfade_segments(all_vocals, overlap_samples)
                accompaniment_output = self.crossfade_segments(all_accompaniment, overlap_samples)

                # Speichern der Dateien
                self.save_audio(vocal_output[0].numpy(), vocal_path, self.model.samplerate, format, bitrate)
                self.save_audio(accompaniment_output[0].numpy(), accompaniment_path, self.model.samplerate, format,
                                bitrate)
            else:
                # Verarbeite die gesamte Datei auf einmal
                with torch.no_grad():
                    sources = apply_model(self.model, wav)

                # Indizes für demucs sources
                stem_names = self.model.sources
                print(f"Verfügbare Stems: {stem_names}")

                vocal_idx = stem_names.index('vocals') if 'vocals' in stem_names else -1

                if vocal_idx != -1:
                    # Vocals speichern
                    vocals = sources[:, vocal_idx].cpu().numpy()
                    self.save_audio(vocals[0], vocal_path, self.model.samplerate, format, bitrate)

                    # Begleitung erstellen (alle Quellen außer vocals)
                    accompaniment = torch.zeros_like(sources[:, 0])
                    for idx, name in enumerate(stem_names):
                        if idx != vocal_idx:
                            accompaniment += sources[:, idx]

                    self.save_audio(accompaniment[0].cpu().numpy(), accompaniment_path, self.model.samplerate, format,
                                    bitrate)
                else:
                    print("Warnung: Keine Vocals-Stem gefunden in den verfügbaren Stems!")
                    return {}

            return {
                'vocals': vocal_path,
                'accompaniment': accompaniment_path
            }

        except Exception as e:
            print(f"Fehler bei der Extraktion: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def crossfade_segments(self, segments, overlap_samples):
        """
        Fügt Audio-Segmente mit Überblendung zusammen

        Args:
            segments: Liste von Audio-Segmenten als Tensoren
            overlap_samples: Anzahl der überlappenden Samples

        Returns:
            Zusammengeführter Audio-Tensor
        """
        if len(segments) == 1:
            return segments[0]

        # Erstelle Fenster für Überblendung
        fade_in = torch.linspace(0., 1., overlap_samples)
        fade_out = 1. - fade_in

        # Erstelle ersten Teil des Ausgabetensors
        result = segments[0][:, :, :-overlap_samples]

        for i in range(len(segments) - 1):
            # Aktuelles und nächstes Segment
            current = segments[i]
            next_seg = segments[i + 1]

            # Überlappungsbereich
            curr_overlap = current[:, :, -overlap_samples:]
            next_overlap = next_seg[:, :, :overlap_samples]

            # Überblendung berechnen
            blended = fade_out.view(1, 1, -1) * curr_overlap + fade_in.view(1, 1, -1) * next_overlap

            # Anfügen an Ergebnis
            result = torch.cat([result, blended], dim=2)

            # Rest des nächsten Segments anfügen, wenn es nicht das letzte ist
            if i < len(segments) - 2:
                result = torch.cat([result, next_seg[:, :, overlap_samples:-overlap_samples]], dim=2)
            else:
                # Für das letzte Segment den Rest komplett anfügen
                result = torch.cat([result, next_seg[:, :, overlap_samples:]], dim=2)

        return result

    def batch_process(self, input_files, output_dir, format="mp3", bitrate="192k"):
        """
        Verarbeitet mehrere Dateien im Batch-Modus

        Args:
            input_files: Liste von Eingabedateipfaden
            output_dir: Basisverzeichnis für die Ausgabe
            format: Ausgabeformat ("mp3" oder "wav")
            bitrate: Bitrate für MP3-Kompression

        Returns:
            Dictionary mit Eingabedateien als Schlüssel und Ausgabepfaden als Werten
        """
        results = {}
        for input_file in tqdm.tqdm(input_files, desc="Verarbeite Dateien"):
            file_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0])
            results[input_file] = self.extract_vocals(input_file, file_output_dir, format, bitrate)

        return results


def main():
    parser = argparse.ArgumentParser(description='VoiceXtract - Extrahiere Vocals aus Musikdateien')
    parser.add_argument('input', help='Eingabedatei oder Verzeichnis mit Audiodateien')
    parser.add_argument('-o', '--output', default='output', help='Ausgabeverzeichnis (Standard: output)')
    parser.add_argument('-m', '--model', default='htdemucs',
                        help='Name des zu verwendenden Modells (Standard: htdemucs)')
    parser.add_argument('-f', '--format', default='mp3', choices=['mp3', 'wav'], help='Ausgabeformat (Standard: mp3)')
    parser.add_argument('-b', '--bitrate', default='192k', help='Bitrate für MP3 (Standard: 192k)')
    parser.add_argument('-r', '--recursive', action='store_true', help='Verzeichnisse rekursiv durchsuchen')
    parser.add_argument('-s', '--segment', type=int,
                        help='Segmentgröße in Sekunden für Chunk-Verarbeitung (reduziert Speicherverbrauch)')

    args = parser.parse_args()

    # Überprüfe ob ffmpeg installiert ist, wenn MP3 als Format gewählt wurde
    if args.format.lower() == "mp3":
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("WARNUNG: ffmpeg wurde nicht gefunden! MP3-Ausgabe wird nicht möglich sein.")
            print("Bitte installieren Sie ffmpeg (z.B. mit 'brew install ffmpeg' unter macOS)")
            print("Setze Format auf WAV...")
            args.format = "wav"

    extractor = VocalExtractor(model_name=args.model, segment_size=args.segment)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        # Einzelne Datei verarbeiten
        print(f"Verarbeite Datei: {input_path}")
        result = extractor.extract_vocals(str(input_path), str(output_dir), args.format, args.bitrate)
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
        results = extractor.batch_process(audio_files, str(output_dir), args.format, args.bitrate)
        print(f"Verarbeitung abgeschlossen. Ausgabe in: {output_dir}")


if __name__ == "__main__":
    main()