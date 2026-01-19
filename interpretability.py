import matplotlib
matplotlib.use('Agg') 
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

class GradientAnalyzer:
    def __init__(self, pl_model, device='cuda'):

        self.model = pl_model.to(device)
        self.model.eval()
        self.device = device

        for param in self.model.parameters():
            param.requires_grad = False


    def save_gradients(self, gradients, filename='gradients.npy'):
        np.save(filename, gradients)

    def compute_input_gradients(self, loader, num_batches=5):
        all_gradients = []

        original_log = self.model.log
        self.model.log = lambda *args, **kwargs: None

        try:
            with torch.enable_grad():
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= num_batches:
                        break

                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)

                    batch['eeg'].requires_grad_(True)

                    eeg_z, img_z_proj, _ = self.model(batch)

                    eeg_z_norm = F.normalize(eeg_z, p=2, dim=-1)
                    img_z_proj_norm = F.normalize(img_z_proj, p=2, dim=-1)

                    similarity_scores = (eeg_z_norm * img_z_proj_norm).sum(dim=1)
                    total_score = similarity_scores.sum()

                    self.model.zero_grad()
                    total_score.backward()

                    if batch['eeg'].grad is not None:
                        grads = batch['eeg'].grad.data.cpu().numpy()
                        all_gradients.append(np.abs(grads))
                    else:
                        print(f"Warning: Batch {batch_idx} has no gradient.")

        finally:
            self.model.log = original_log

        if len(all_gradients) > 0:
            all_gradients = np.concatenate(all_gradients, axis=0)
            print(f"Gradient calculation completed Shape: {all_gradients.shape}")
            return all_gradients
        else:
            raise RuntimeError("Failed to calculate any gradient successfully.")

    def plot_temporal_analysis(self, gradients, fs=250, save_path='temporal_gradient.png'):
        temporal_importance = np.mean(gradients, axis=(0, 1))

        if temporal_importance.max() != temporal_importance.min():
            temporal_importance = (temporal_importance - temporal_importance.min()) / (
                        temporal_importance.max() - temporal_importance.min())

        time_axis = np.arange(len(temporal_importance)) * (1000 / fs)  # ms

        plt.figure(figsize=(10, 5))
        plt.plot(time_axis, temporal_importance, label='Gradient Energy', color='teal', linewidth=2)
        plt.fill_between(time_axis, temporal_importance, alpha=0.3, color='teal')

        plt.title("Temporal Importance (Gradient Energy across Time)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Normalized Importance")
        plt.grid(True, linestyle='--', alpha=0.5)

        # 0-200ms
        plt.axvspan(0, 200, color='yellow', alpha=0.2, label='Early Visual Processing (0-200ms)')
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        try:
            plt.show()
        except Exception as e:
            print(f"Unable to display images on the screen {e}")
        plt.close()

    def plot_spectral_analysis(self, gradients, fs=250, save_path='spectral_gradient.png'):
        n_per_seg = min(gradients.shape[-1], 256)
        freqs, psd = welch(gradients, fs=fs, nperseg=n_per_seg, axis=-1)
        mean_psd = np.mean(psd, axis=(0, 1))

        bands = {
            'Delta\n(0-4Hz)': (0, 4),
            'Theta\n(4-8Hz)': (4, 8),
            'Alpha\n(8-12Hz)': (8, 12),
            'Beta\n(12-30Hz)': (12, 30),
            'Gamma\n(>30Hz)': (30, 100)
        }

        band_energies = []
        band_names = []

        for name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            if np.sum(idx) > 0:
                energy = np.mean(mean_psd[idx])
            else:
                energy = 0
            band_energies.append(energy)
            band_names.append(name)

        plt.figure(figsize=(8, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = plt.bar(band_names, band_energies, color=colors, edgecolor='black', alpha=0.8)

        plt.title("Spectral Importance (PSD of Gradients)")
        plt.ylabel("Power Spectral Density (Gradient Magnitude)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        try:
            plt.show()
        except Exception:
            pass
        plt.close()