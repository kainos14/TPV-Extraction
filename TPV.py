import numpy as np
import pandas as pd
from gudhi import CubicalComplex
from scipy.stats import skew, kurtosis, entropy

def compute_persistence(signal):
    """
    Compute persistence diagram for a 1D signal using sublevel set filtration.
    """
    signal = np.array(signal, dtype=float)
    n = len(signal)
    # Reshape for CubicalComplex: needs a 2D grid (reshape to (n,1))
    cc = CubicalComplex(top_dimensional_cells=signal.reshape(-1, 1))
    cc.persistence(homology_coeff_field=2, min_persistence=0)
    diag = cc.persistence_intervals_in_dimension(0)
    return diag

def extract_tpv_features(signal, top_k=5):
    """
    Extract 33 TPV features from a single 1D signal segment.
    """
    diag = compute_persistence(signal)
    if len(diag) == 0:
        # No features, return zeros
        return [0.0]*33

    births = diag[:, 0]
    deaths = diag[:, 1]
    lifetimes = deaths - births

    # Basic stats
    mean_life = np.mean(lifetimes)
    std_life = np.std(lifetimes)
    max_life = np.max(lifetimes)
    min_life = np.min(lifetimes)
    skew_life = skew(lifetimes)
    kurt_life = kurtosis(lifetimes)

    # Energy-based
    life_energy = np.sum(lifetimes**2)
    norm_energy = life_energy / (len(lifetimes) + 1e-8)

    # Entropy of lifetimes
    hist, _ = np.histogram(lifetimes, bins=10, density=True)
    life_entropy = entropy(hist + 1e-8)

    # Birth stats
    mean_birth = np.mean(births)
    std_birth = np.std(births)
    max_birth = np.max(births)
    min_birth = np.min(births)

    # Death stats
    mean_death = np.mean(deaths)
    std_death = np.std(deaths)
    max_death = np.max(deaths)
    min_death = np.min(deaths)

    # Top-K features
    top_k_life = sorted(lifetimes, reverse=True)[:top_k]
    top_k_life += [0.0]*(top_k - len(top_k_life))

    top_k_birth = sorted(births, reverse=True)[:top_k]
    top_k_birth += [0.0]*(top_k - len(top_k_birth))

    top_k_death = sorted(deaths, reverse=True)[:top_k]
    top_k_death += [0.0]*(top_k - len(top_k_death))

    # Assemble 33 features
    features = [
        mean_life, std_life, max_life, min_life, skew_life, kurt_life, life_energy, norm_energy, life_entropy,
        mean_birth, std_birth, max_birth, min_birth,
        mean_death, std_death, max_death, min_death
    ] + top_k_life + top_k_birth + top_k_death

    return features

def extract_tpv_from_dataframe(df, signal_col, segment_length=2100):
    """
    Extract TPV features from all rows in a DataFrame (PPG/ECG signal segments).
    """
    tpv_feature_list = []
    for idx, row in df.iterrows():
        signal = row[signal_col]
        features = extract_tpv_features(signal)
        tpv_feature_list.append(features)
    tpv_df = pd.DataFrame(tpv_feature_list, columns=[
        "mean_life", "std_life", "max_life", "min_life", "skew_life", "kurt_life", "life_energy", "norm_energy", "life_entropy",
        "mean_birth", "std_birth", "max_birth", "min_birth",
        "mean_death", "std_death", "max_death", "min_death"
    ] + [f"top_life_{i+1}" for i in range(5)] +
        [f"top_birth_{i+1}" for i in range(5)] +
        [f"top_death_{i+1}" for i in range(5)]
    )
    return tpv_df

# Example Usage
if __name__ == "__main__":
    # Generate dummy PPG signal for testing
    test_signal = np.sin(np.linspace(0, 4*np.pi, 2100)) + 0.1*np.random.randn(2100)
    
    features = extract_tpv_features(test_signal)
    print("Extracted TPV (33 features):")
    print(features)
