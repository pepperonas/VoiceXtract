import subprocess

import numpy as np
import pytest
import soundfile as sf

import voicextract as vx


# --- segment_bounds ---------------------------------------------------------

@pytest.mark.parametrize("total,seg,ov", [
    (180, 100, 25), (170, 100, 25), (101, 100, 25), (175, 100, 25),
    (1000, 100, 25), (44100 * 7 + 3, 44100, 11025), (250, 100, 0),
])
def test_segments_cover_everything_with_exact_overlap(total, seg, ov):
    bounds = vx.segment_bounds(total, seg, ov)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == total
    for (s1, e1), (s2, e2) in zip(bounds, bounds[1:]):
        assert e1 - s2 == ov              # neighbours overlap by exactly `ov`
        assert e1 - s1 == seg             # only the last segment may be shorter
    # every segment is longer than the crossfade it takes part in
    assert all(e - s > ov for s, e in bounds[1:])


def test_no_redundant_tail_segment():
    # Used to produce a third segment (150, 170) of 20 samples, shorter than
    # the 25-sample overlap -> crossfade shape error.
    assert vx.segment_bounds(170, 100, 25) == [(0, 100), (75, 170)]


def test_short_audio_is_one_segment():
    assert vx.segment_bounds(60, 100, 25) == [(0, 60)]
    assert vx.segment_bounds(0, 100, 25) == []


def test_invalid_segment_size():
    with pytest.raises(ValueError):
        vx.segment_bounds(100, 25, 25)    # would never advance


# --- crossfade_segments ------------------------------------------------------

def crossfade(segments, ov):
    # crossfade_segments does not use the model, so skip __init__.
    ex = vx.VocalExtractor.__new__(vx.VocalExtractor)
    return ex.crossfade_segments(segments, ov)


def split(signal, seg, ov):
    return [signal[:, :, s:e] for s, e in vx.segment_bounds(signal.shape[2], seg, ov)]


@pytest.mark.parametrize("total", [101, 170, 180, 333, 1000])
def test_crossfade_reconstructs_the_signal(total):
    rng = np.random.default_rng(total)
    signal = rng.standard_normal((1, 2, total))
    out = crossfade(split(signal, 100, 25), 25)
    assert out.shape == signal.shape
    # fade_in + fade_out == 1, so identical overlaps blend back to the original
    assert np.allclose(out, signal)


def test_crossfade_blends_linearly():
    a = np.ones((1, 1, 8))
    b = np.zeros((1, 1, 8))
    out = crossfade([a, b], 4)
    assert out.shape == (1, 1, 12)
    assert np.allclose(out[0, 0, 4:8], [1, 2 / 3, 1 / 3, 0])
    assert np.allclose(out[0, 0, :4], 1) and np.allclose(out[0, 0, 8:], 0)


def test_single_segment_is_returned_unchanged():
    seg = np.arange(10.0).reshape(1, 1, 10)
    assert crossfade([seg], 3) is seg


# --- save_audio --------------------------------------------------------------

def stereo(n=441):
    t = np.linspace(0, 1, n)
    return np.stack([np.sin(t), -np.sin(t)]) * 0.5


def test_save_wav_transposes_channels(tmp_path):
    ex = vx.VocalExtractor.__new__(vx.VocalExtractor)
    wav = stereo()
    path = ex.save_audio(wav, str(tmp_path / "sub" / "v.wav"), 44100, format="wav")
    assert path == str(tmp_path / "sub" / "v.wav")
    data, sr = sf.read(path)
    assert sr == 44100
    assert data.shape == (441, 2)                     # (time, channels) on disk
    assert np.allclose(data.T, wav, atol=1e-4)


def test_save_clips_to_minus_one_plus_one(tmp_path, monkeypatch):
    written = []
    real_write = vx.sf.write
    monkeypatch.setattr(vx.sf, "write",
                        lambda path, data, sr: written.append(data.copy()) or real_write(path, data, sr))
    ex = vx.VocalExtractor.__new__(vx.VocalExtractor)
    ex.save_audio(np.array([[3.0, -3.0, 0.5]]), str(tmp_path / "c.wav"), 8000, format="wav")
    # clipping happens before writing, not by chance inside the WAV encoder
    assert np.allclose(written[0].ravel(), [1.0, -1.0, 0.5])


def test_mp3_calls_ffmpeg_with_bitrate(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vx.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    ex = vx.VocalExtractor.__new__(vx.VocalExtractor)
    out = str(tmp_path / "v.mp3")
    assert ex.save_audio(stereo(), out, 44100, format="mp3", bitrate="320k") == out
    cmd = calls[0]
    assert cmd[0] == "ffmpeg" and cmd[-1] == out
    assert cmd[cmd.index("-b:a") + 1] == "320k"
    assert not list(tmp_path.glob("*.wav"))           # temp file cleaned up


@pytest.mark.parametrize("error", [
    FileNotFoundError(),
    subprocess.CalledProcessError(1, "ffmpeg", stderr=b"boom"),
])
def test_mp3_falls_back_to_wav_and_reports_real_path(tmp_path, monkeypatch, error):
    def fail(cmd, **kw):
        raise error
    monkeypatch.setattr(vx.subprocess, "run", fail)
    ex = vx.VocalExtractor.__new__(vx.VocalExtractor)
    path = ex.save_audio(stereo(), str(tmp_path / "v.mp3"), 44100, format="mp3")
    assert path == str(tmp_path / "v.wav")
    assert sf.read(path)[0].shape == (441, 2)


# --- extract_vocals with a fake model ---------------------------------------

class FakeModel:
    sources = ["drums", "bass", "other", "vocals"]
    audio_channels = 2
    samplerate = 100


def make_extractor(monkeypatch, length, segment_size=None, sources=None):
    signal = np.random.default_rng(0).standard_normal((2, length))

    class FakeAudioFile:
        def __init__(self, path):
            pass

        def read(self, channels, samplerate):
            return signal

    def fake_apply(model, mix):
        # stems: drums = 0.1*mix, bass = 0.2*mix, other = 0.3*mix, vocals = 0.4*mix
        factors = np.array([0.1, 0.2, 0.3, 0.4])[: len(model.sources)]
        return factors[None, :, None, None] * mix[:, None]

    monkeypatch.setattr(vx, "AudioFile", FakeAudioFile)
    monkeypatch.setattr(vx, "apply_model", fake_apply)
    ex = vx.VocalExtractor.__new__(vx.VocalExtractor)
    ex.model = FakeModel()
    if sources:
        ex.model.sources = sources
    ex.device = "cpu"
    ex.segment_size = segment_size
    return ex, signal


@pytest.mark.parametrize("segment_size", [None, 1])
def test_extract_vocals_splits_vocals_and_accompaniment(tmp_path, monkeypatch, segment_size):
    # 170 samples at 100 Hz with 1-s segments hits the former tail-segment crash
    ex, signal = make_extractor(monkeypatch, 170, segment_size)
    monkeypatch.setattr(vx.tqdm, "tqdm", lambda it, **kw: it)
    result = ex.extract_vocals("song.mp3", str(tmp_path), format="wav")
    vocals, _ = sf.read(result["vocals"])
    accomp, _ = sf.read(result["accompaniment"])
    assert np.allclose(vocals.T, np.clip(0.4 * signal, -1, 1), atol=1e-4)
    assert np.allclose(accomp.T, np.clip(0.6 * signal, -1, 1), atol=1e-4)


def test_extract_without_vocal_stem_returns_empty(tmp_path, monkeypatch):
    ex, _ = make_extractor(monkeypatch, 50, sources=["drums", "bass"])
    assert ex.extract_vocals("x.wav", str(tmp_path), format="wav") == {}


def test_extract_reports_fallback_path(tmp_path, monkeypatch):
    ex, _ = make_extractor(monkeypatch, 50)

    def no_ffmpeg(cmd, **kw):
        raise FileNotFoundError
    monkeypatch.setattr(vx.subprocess, "run", no_ffmpeg)
    result = ex.extract_vocals("x.wav", str(tmp_path), format="mp3")
    assert result["vocals"].endswith("vocals.wav")
    assert all(__import__("os").path.exists(p) for p in result.values())


# --- file discovery and naming -----------------------------------------------

def test_find_audio_files(tmp_path):
    for rel in ["b.MP3", "a.wav", "notes.txt", "sub/c.flac", "sub/deep/d.ogg", "e.m4a"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    (tmp_path / "dir.mp3").mkdir()                    # a directory, not a file
    names = lambda files: [f.replace(str(tmp_path) + "/", "") for f in files]
    assert names(vx.find_audio_files(tmp_path)) == ["a.wav", "b.MP3", "e.m4a"]
    assert names(vx.find_audio_files(tmp_path, recursive=True)) == [
        "a.wav", "b.MP3", "e.m4a", "sub/c.flac", "sub/deep/d.ogg"]


def test_unique_output_names():
    used = set()
    names = [vx.unique_output_name(f, used) for f in
             ["a/song.mp3", "b/song.mp3", "song.wav", "Song.flac", "other.mp3"]]
    assert names == ["song", "song (2)", "song (3)", "Song (4)", "other"]


def test_batch_process_keeps_same_named_files_apart(tmp_path, monkeypatch):
    ex = vx.VocalExtractor.__new__(vx.VocalExtractor)
    seen = []
    monkeypatch.setattr(ex, "extract_vocals", lambda f, out, *a: seen.append(out) or {"f": f})
    monkeypatch.setattr(vx.tqdm, "tqdm", lambda it, **kw: it)
    results = ex.batch_process(["x/song.mp3", "y/song.mp3"], str(tmp_path))
    assert len(set(seen)) == 2
    assert set(results) == {"x/song.mp3", "y/song.mp3"}


def test_main_directory_without_audio(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vx, "VocalExtractor", lambda **kw: object())
    vx.main([str(tmp_path), "-f", "wav"])
    assert "Keine Audiodateien" in capsys.readouterr().out


def test_main_without_ffmpeg_switches_to_wav(tmp_path, monkeypatch):
    (tmp_path / "s.mp3").write_bytes(b"")
    calls = []

    class Fake:
        def __init__(self, **kw):
            calls.append(("init", kw))

        def extract_vocals(self, path, out, fmt, bitrate):
            calls.append(("extract", fmt))
            return {}

    def no_ffmpeg(cmd, **kw):
        raise FileNotFoundError
    monkeypatch.setattr(vx.subprocess, "run", no_ffmpeg)
    monkeypatch.setattr(vx, "VocalExtractor", Fake)
    vx.main([str(tmp_path / "s.mp3"), "-s", "30"])
    assert calls[0] == ("init", {"model_name": "htdemucs", "segment_size": 30})
    assert calls[1] == ("extract", "wav")
