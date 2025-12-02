"""
Extração de features de áudio.
"""

import librosa
import numpy as np


class ExtratorFeatures:
    """Extrai características de músicas."""

    def __init__(self, duracao_max=210):
        self.duracao_max = duracao_max

    def carregar_audio(self, caminho_arquivo):
        """Carrega um arquivo de áudio."""
        y, sr = librosa.load(caminho_arquivo, duration=self.duracao_max)
        return y, sr

    def extrair_mfcc(self, y, sr):
        """Extrai MFCCs."""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return [float(np.mean(mfcc[i])) for i in range(len(mfcc))]

    def extrair_spectral_centroid(self, y, sr):
        """Extrai centróide espectral."""
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        return float(np.mean(centroid))

    def extrair_spectral_rolloff(self, y, sr):
        """Extrai spectral rolloff."""
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        return float(np.mean(rolloff))

    def extrair_zero_crossing_rate(self, y):
        """Extrai taxa de cruzamento de zero."""
        zcr = librosa.feature.zero_crossing_rate(y)
        return float(np.mean(zcr))

    def extrair_chroma(self, y, sr):
        """Extrai características cromáticas."""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return [float(np.mean(chroma[i])) for i in range(len(chroma))]

    def extrair_tempo(self, y, sr):
        """Estima o tempo (BPM)."""
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo) if isinstance(tempo, np.ndarray) else tempo

    def extrair_todas_features(self, caminho_arquivo):
        """Extrai todas as features de um arquivo com normalização."""
        y, sr = self.carregar_audio(caminho_arquivo)

        features = []

        # MFCCs (13 valores) - já estão em escala razoável
        mfccs = self.extrair_mfcc(y, sr)
        features.extend(mfccs)

        # Spectral centroid - normalizar por sr/2 (Nyquist)
        centroid = self.extrair_spectral_centroid(y, sr)
        features.append(centroid / (sr / 2))

        # Spectral rolloff - normalizar por sr/2
        rolloff = self.extrair_spectral_rolloff(y, sr)
        features.append(rolloff / (sr / 2))

        # Zero crossing rate - já está entre 0 e 1
        zcr = self.extrair_zero_crossing_rate(y)
        features.append(zcr)

        # Chroma (12 valores) - já estão normalizados (0 a 1)
        chroma = self.extrair_chroma(y, sr)
        features.extend(chroma)

        # Tempo (BPM) - normalizar dividindo por 200
        tempo = self.extrair_tempo(y, sr)
        features.append(tempo / 200.0)

        return features
