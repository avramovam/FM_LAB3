"""
Вспомогательные функции для отчета ЛР3 — полная версия
Все графики для всех экспериментов
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
import os
from config import OutputParams, DisplayParams

# === НАСТРОЙКИ ГРАФИКОВ (читаемость при печати А4) ===
plt.rcParams['figure.figsize'] = (14, 9)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

COLORS = {'orig': '#2E86AB', 'noisy': '#A23B72', 'filt': '#06A77D', 'mask': '#000000'}
LINES = {'orig': '-', 'noisy': '--', 'filt': '-.'}


def create_rect_pulse(t, a, t1, t2):
    g = np.zeros_like(t)
    g[(t >= t1) & (t <= t2)] = a
    return g


def create_noisy_signal(g, t, b=0, c=0, d=0):
    u = g.copy()
    if b > 0:
        u += b * (2 * np.random.random(len(t)) - 1)
    if c > 0 and d > 0:
        u += c * np.sin(d * t)
    return u


def apply_freq_filter(signal, freqs, mask_func):
    S = fftshift(fft(signal))
    mask = mask_func(freqs)
    S_f = S * mask
    return np.real(ifft(ifftshift(S_f))), mask, S, S * mask


# === Маски фильтров ===
def lpf_mask(freqs, cutoff): return np.abs(freqs) <= cutoff
def hpf_mask(freqs, cutoff): return np.abs(freqs) >= cutoff
def notch_mask(freqs, center, width):
    m = np.ones_like(freqs, dtype=bool)
    m[(np.abs(freqs) >= center-width) & (np.abs(freqs) <= center+width)] = False
    return m
def combined_mask(freqs, lpf_cut, notch_c, notch_w):
    return lpf_mask(freqs, lpf_cut) & notch_mask(freqs, notch_c, notch_w)
def bandpass_mask(freqs, low, high):
    m = np.zeros_like(freqs, dtype=bool)
    m[(np.abs(freqs) >= low) & (np.abs(freqs) <= high)] = True
    return m


def calc_mse(orig, filt): return np.mean((orig - filt) ** 2)


def _save(fig, path):
    if path and OutputParams.save_figures:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=OutputParams.figures_dpi, bbox_inches='tight')
    if not DisplayParams.SHOW_PLOTS: plt.close(fig)


def _style(ax, xlabel, ylabel, title, legend_loc='upper right'):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11, loc=legend_loc)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# === Графики для отчета ===

def plot_time_three(t, g, u, uf, title, save_path, xlim=(-3, 3)):
    """3 сигнала во времени"""
    fig, ax = plt.subplots()
    ax.plot(t, g, color=COLORS['orig'], linestyle=LINES['orig'], linewidth=1.8, label='Исходный $g(t)$')
    ax.plot(t, u, color=COLORS['noisy'], linestyle=LINES['noisy'], linewidth=1, alpha=0.8, label='Зашумленный $u(t)$')
    ax.plot(t, uf, color=COLORS['filt'], linestyle=LINES['filt'], linewidth=1.8, label='Отфильтрованный')
    _style(ax, 'Время, с', 'Амплитуда', title)
    ax.set_xlim(xlim)
    _save(fig, save_path)


def plot_spectrum_before(freqs, G, U, title, save_path, xlim=(-30, 30)):
    """Спектры до фильтрации"""
    fig, ax = plt.subplots()
    ax.plot(freqs, np.abs(G), color=COLORS['orig'], linestyle=LINES['orig'], linewidth=1.8, label='$|G(\\nu)|$')
    ax.plot(freqs, np.abs(U), color=COLORS['noisy'], linestyle=LINES['noisy'], linewidth=1, alpha=0.8, label='$|U(\\nu)|$')
    _style(ax, 'Частота, Гц', '$|\\mathcal{F}\\{\\cdot\\}|$', title)
    ax.set_xlim(xlim)
    _save(fig, save_path)


def plot_spectrum_after(freqs, G, Uf, title, save_path, xlim=(-30, 30)):
    """Спектры после фильтрации"""
    fig, ax = plt.subplots()
    ax.plot(freqs, np.abs(G), color=COLORS['orig'], linestyle=LINES['orig'], linewidth=1.8, label='$|G(\\nu)|$')
    ax.plot(freqs, np.abs(Uf), color=COLORS['filt'], linestyle=LINES['filt'], linewidth=1.8, label='$|U_f(\\nu)|$')
    _style(ax, 'Частота, Гц', '$|\\mathcal{F}\\{\\cdot\\}|$', title)
    ax.set_xlim(xlim)
    _save(fig, save_path)


def plot_mask(freqs, mask, title, save_path, xlim=(-30, 30)):
    """АЧХ фильтра"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(freqs, mask.astype(float), color=COLORS['mask'], linewidth=2, label='Коэффициент передачи')
    ax.fill_between(freqs, 0, mask.astype(float), alpha=0.2, color='gray')
    _style(ax, 'Частота, Гц', '', title, legend_loc='upper right')
    ax.set_xlim(xlim)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    _save(fig, save_path)


def plot_mse_line(x, y, xlabel, title, save_path, log_y=False):
    """MSE vs параметр"""
    fig, ax = plt.subplots()
    if log_y:
        ax.semilogy(x, y, 'bo-', linewidth=1.8, markersize=8, label='Эксперимент')
        # Добавим теоретическую линию ~x² для наглядности
        x_fit = np.array(x)
        y_fit = y[0] * (x_fit / x[0])**2
        ax.plot(x_fit, y_fit, 'r--', linewidth=1, alpha=0.7, label='$\\propto x^2$')
    else:
        ax.plot(x, y, 'bo-', linewidth=1.8, markersize=8)
    _style(ax, xlabel, 'MSE', title)
    ax.grid(True, alpha=0.3)
    if log_y: ax.legend(fontsize=10)
    _save(fig, save_path)


def plot_mse_heatmap(b_vals, c_vals, mse_mat, save_path):
    """Тепловая карта MSE"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mse_mat, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(im, label='MSE', ax=ax)
    ax.set_xlabel('Амплитуда гармоники $c$', fontsize=13)
    ax.set_ylabel('Амплитуда шума $b$', fontsize=13)
    ax.set_title('MSE при разных шумах', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(np.arange(len(c_vals)))
    ax.set_yticks(np.arange(len(b_vals)))
    ax.set_xticklabels([f'{c}' for c in c_vals])
    ax.set_yticklabels([f'{b}' for b in b_vals])
    for i in range(len(b_vals)):
        for j in range(len(c_vals)):
            col = 'white' if mse_mat[i, j] > 0.1 else 'black'
            ax.text(j, i, f'{mse_mat[i, j]:.4f}', ha='center', va='center', color=col, fontsize=10)
    _save(fig, save_path)


def plot_comparison_three(t, g, u_lpf, u_notch, u_comb, mse_lpf, mse_notch, mse_comb, title, save_path, xlim=(-2, 2)):
    """Сравнение трёх фильтров на одном графике"""
    fig, ax = plt.subplots()
    ax.plot(t, g, 'k:', linewidth=1, label='Исходный $g(t)$', alpha=0.6)
    ax.plot(t, u_lpf, 'b--', linewidth=1.2, label=f'ФНЧ (MSE={mse_lpf:.4f})', alpha=0.8)
    ax.plot(t, u_notch, 'r--', linewidth=1.2, label=f'Режектор (MSE={mse_notch:.4f})', alpha=0.8)
    ax.plot(t, u_comb, 'g-', linewidth=1.8, label=f'Комбинированный (MSE={mse_comb:.4f})')
    _style(ax, 'Время, с', 'Амплитуда', title)
    ax.set_xlim(xlim)
    _save(fig, save_path)


def ensure_dirs():
    os.makedirs(OutputParams.figures_dir, exist_ok=True)
    os.makedirs(OutputParams.audio_dir, exist_ok=True)