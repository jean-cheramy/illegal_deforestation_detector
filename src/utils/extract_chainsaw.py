import librosa
import numpy as np


def extract_chainsaw_features(
        arr,
        sr=12000,
        n_mfcc=20,
        f_ref_low=(100, 250),  # typical idle pitch band for chainsaw
        f_ref_high=(2000, 4000)  # typical higher freq engine noise band
):
    """
    Extract ~100D feature vector specialized for chainsaw detection.

    Parameters
    ----------
    arr : np.ndarray
        Audio time-series (3 seconds at sr=12kHz => ~36000 samples).
    sr : int
        Sample rate in Hz. Default 12000 for rainforest problem.
    n_mfcc : int
        Number of MFCCs to compute. (We'll also compute deltas.)
    f_ref_low : (float, float)
        Frequency band (low-mid range) where chainsaw fundamental often sits.
    f_ref_high : (float, float)
        Frequency band (upper range) capturing typical engine overtones / noise.

    Returns
    -------
    features : np.ndarray
        1D array of length ~100 containing domain-specific features
        for chainsaw detection.
    """

    hop_length = 512
    n_fft = 1024

    # 2. Pitch estimation: using librosa's YIN
    #    -> we get an f0 contour over time
    fmin = 50  # Hz, below typical chainsaw fundamentals
    fmax = 600  # Hz, above typical rev idle
    f0_series = librosa.yin(y=arr, sr=sr, fmin=fmin, fmax=fmax, frame_length=n_fft, hop_length=hop_length)

    # If YIN fails to find a pitch, it may produce NaNs; replace them:
    f0_series = np.nan_to_num(f0_series, nan=0.0)

    # Basic pitch statistics
    pitch_mean = np.mean(f0_series)
    pitch_std = np.std(f0_series)
    pitch_min = np.min(f0_series)
    pitch_max = np.max(f0_series)
    pitch_range = pitch_max - pitch_min
    # Count how many frames YIN flagged as unvoiced (f0=0)
    pitch_unvoiced_count = np.sum(f0_series < 1.0)

    # Compute the STFT once for efficiency:
    stft_data = librosa.stft(arr, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(stft_data)  # magnitude spectrogram

    # Standard spectral features from librosa
    spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    spec_flux = librosa.onset.onset_strength(S=librosa.power_to_db(S ** 2, ref=np.max), sr=sr, hop_length=hop_length)

    # 2D shape for consistent stats
    spec_flux = np.expand_dims(spec_flux, axis=0)

    # Summarize each feature with mean & std across time
    def feature_mean_std(feat):
        return np.array([np.mean(feat), np.std(feat)])

    centroid_stats = feature_mean_std(spec_centroid)
    bandwidth_stats = feature_mean_std(spec_bandwidth)
    rolloff_stats = feature_mean_std(spec_rolloff)
    flux_stats = feature_mean_std(spec_flux)

    # Zero-crossing rate (time-domain)
    zcr = librosa.feature.zero_crossing_rate(y=arr, frame_length=n_fft, hop_length=hop_length)
    zcr_stats = feature_mean_std(zcr)

    # Short-time energy
    frame_energy = np.sum(S ** 2, axis=0)
    energy_mean = np.mean(frame_energy)
    energy_std = np.std(frame_energy)

    # 4. Energy ratio in specific frequency bands:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # Indices for each band
    idx_low = np.where((freqs >= f_ref_low[0]) & (freqs <= f_ref_low[1]))[0]
    idx_high = np.where((freqs >= f_ref_high[0]) & (freqs <= f_ref_high[1]))[0]

    # Integrate magnitude-squared in each band across time, then ratio
    band_energy_low = np.sum(S[idx_low, :] ** 2)
    band_energy_high = np.sum(S[idx_high, :] ** 2)
    total_energy = np.sum(S ** 2)
    ratio_low = band_energy_low / (total_energy + 1e-9)
    ratio_high = band_energy_high / (total_energy + 1e-9)

    # MFCCs: including delta and possibly delta-delta
    mfcc = librosa.feature.mfcc(y=arr, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # We'll take the mean across time for each coefficient, for each set:
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)
    mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
    mfcc_delta2_std = np.std(mfcc_delta2, axis=1)

    # Combine them (this yields 6*n_mfcc features):
    mfcc_features = np.concatenate([
        mfcc_mean, mfcc_std,
        mfcc_delta_mean, mfcc_delta_std,
        mfcc_delta2_mean, mfcc_delta2_std
    ])

    # 6. Harmonic detection: ratio of harmonic energies
    harmonic_ratios = []
    if pitch_mean > 0:
        # We define a small helper that sums energy near a given freq ± some tolerance
        def harmonic_energy(mag_spectrogram, base_freq, sr, freq_tolerance=20.0):
            freqs_local = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            idx = np.where((freqs_local >= base_freq - freq_tolerance) &
                           (freqs_local <= base_freq + freq_tolerance))[0]
            if len(idx) == 0:
                return 0.0
            return np.sum(mag_spectrogram[idx, :] ** 2)

        # Fundamental
        energy_fund = harmonic_energy(S, pitch_mean, sr)
        # Up to 4 harmonics
        for h in [2, 3, 4, 5]:
            freq_h = h * pitch_mean
            e_harm = harmonic_energy(S, freq_h, sr)
            ratio_h = e_harm / (energy_fund + 1e-9)
            harmonic_ratios.append(ratio_h)
    else:
        # If pitch_mean is zero, no stable fundamental
        harmonic_ratios = [0.0, 0.0, 0.0, 0.0]

    harmonic_ratios = np.array(harmonic_ratios)

    # 7. Concatenate all features into a single ~100D vector

    # Pitch stats
    pitch_features = np.array([
        pitch_mean, pitch_std, pitch_min, pitch_max, pitch_range, pitch_unvoiced_count
    ])

    # Basic spectral shape stats (each has 2 => mean/std)
    spectral_shape_features = np.concatenate([
        centroid_stats,
        bandwidth_stats,
        rolloff_stats,
        flux_stats,
        zcr_stats
    ])

    # Energy stats
    energy_features = np.array([energy_mean, energy_std, ratio_low, ratio_high])

    # Concatenate all              # nb_of_features
    features = np.concatenate([
        pitch_features,  # 6
        spectral_shape_features,  # 10
        energy_features,  # 4
        mfcc_features,  # 6*n_mfcc
        harmonic_ratios  # 4
    ])

    return features
