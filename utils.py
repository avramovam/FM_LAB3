"""
Вспомогательные функции для всех заданий
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
import os
from config import OutputParams, DisplayParams

# Настройка стиля графиков
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def create_rect_pulse(t, a, t1, t2):
    """Создание прямоугольного импульса"""
    g = np.zeros_like(t)
    g[(t >= t1) & (t <= t2)] = a
    return g


def add_uniform_noise(signal, b):
    """Добавление равномерного белого шума"""
    noise = 2 * np.random.random(len(signal)) - 1
    return signal + b * noise, noise


def add_sinusoidal_interference(signal, c, d, t):
    """Добавление синусоидальной помехи"""
    interference = c * np.sin(d * t)
    return signal + interference, interference


def create_noisy_signal(g, t, b=0, c=0, d=0):
    """Создание зашумленного сигнала"""
    u = g.copy()
    noise_components = {}

    if b > 0:
        u, noise = add_uniform_noise(u, b)
        noise_components['uniform'] = noise

    if c > 0 and d > 0:
        u, interference = add_sinusoidal_interference(u, c, d, t)
        noise_components['sinusoidal'] = interference

    return u, noise_components


def apply_freq_filter(signal, freqs, filter_func):
    """Применение частотного фильтра"""
    S = fftshift(fft(signal))
    mask = filter_func(freqs)
    S_filtered = S * mask
    signal_filtered = np.real(ifft(ifftshift(S_filtered)))
    return signal_filtered, mask, S, S_filtered


def create_lpf_mask(freqs, cutoff):
    """Фильтр нижних частот"""
    return np.abs(freqs) <= cutoff


def create_hpf_mask(freqs, cutoff):
    """Фильтр верхних частот"""
    return np.abs(freqs) >= cutoff


def create_notch_mask(freqs, center, width):
    """Режекторный фильтр"""
    mask = np.ones_like(freqs, dtype=bool)
    mask[(np.abs(freqs) >= (center - width)) &
         (np.abs(freqs) <= (center + width))] = False
    return mask


def create_bandpass_mask(freqs, low, high):
    """Полосовой фильтр"""
    mask = np.zeros_like(freqs, dtype=bool)
    mask[(np.abs(freqs) >= low) & (np.abs(freqs) <= high)] = True
    return mask


def calculate_mse(original, filtered):
    """Вычисление среднеквадратичной ошибки"""
    return np.mean((original - filtered) ** 2)


def calculate_snr(original, noisy):
    """Вычисление отношения сигнал/шум (дБ)"""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - noisy) ** 2)
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    return float('inf')


def save_and_close(fig, save_path):
    """Сохранение и закрытие фигуры"""
    if save_path and OutputParams.save_figures:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=OutputParams.figures_dpi,
                   format=OutputParams.figures_format, bbox_inches='tight')
    if not DisplayParams.SHOW_PLOTS:
        plt.close(fig)


def plot_time_domain(t, g, u, u_filtered, title="", save_path=None,
                     xlim=(-3, 3), show_noisy=True):
    """График временной области"""
    fig = plt.figure(figsize=(16, 8))

    plt.plot(t, g, 'g-', linewidth=3, label='Исходный сигнал g(t)')
    if show_noisy:
        plt.plot(t, u, 'r-', alpha=0.6, linewidth=1.5, label='Зашумленный сигнал u(t)')
    plt.plot(t, u_filtered, 'b-', linewidth=2.5, label='Отфильтрованный сигнал')

    plt.xlabel('Время (с)', fontsize=16)
    plt.ylabel('Амплитуда', fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(xlim)
    plt.tight_layout()

    save_and_close(fig, save_path)
    if DisplayParams.SHOW_PLOTS:
        plt.show()
    return fig


def plot_time_comparison(t, g, u_filtered, title="", save_path=None, xlim=(-3, 3)):
    """График сравнения исходного и фильтрованного"""
    fig = plt.figure(figsize=(16, 8))

    plt.plot(t, g, 'g-', linewidth=3, label='Исходный сигнал g(t)')
    plt.plot(t, u_filtered, 'b--', linewidth=2.5, label='Отфильтрованный сигнал')

    plt.xlabel('Время (с)', fontsize=16)
    plt.ylabel('Амплитуда', fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(xlim)
    plt.tight_layout()

    save_and_close(fig, save_path)
    if DisplayParams.SHOW_PLOTS:
        plt.show()
    return fig


def plot_spectrum_magnitude(freqs, G, U, title="", save_path=None, xlim=(-30, 30)):
    """График модулей спектров"""
    fig = plt.figure(figsize=(16, 8))

    plt.plot(freqs, np.abs(G), 'g-', linewidth=2.5, label='|G(ν)| - исходный')
    plt.plot(freqs, np.abs(U), 'r-', alpha=0.7, linewidth=1.5, label='|U(ν)| - зашумленный')

    plt.xlabel('Частота (Гц)', fontsize=16)
    plt.ylabel('Модуль спектра', fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(xlim)
    plt.tight_layout()

    save_and_close(fig, save_path)
    if DisplayParams.SHOW_PLOTS:
        plt.show()
    return fig


def plot_filtered_spectrum(freqs, G, U_filtered, title="", save_path=None, xlim=(-30, 30)):
    """График сравнения исходного и фильтрованного спектров"""
    fig = plt.figure(figsize=(16, 8))

    plt.plot(freqs, np.abs(G), 'g-', linewidth=2.5, label='|G(ν)| - исходный')
    plt.plot(freqs, np.abs(U_filtered), 'b-', linewidth=2, label='|U_filtered(ν)| - фильтрованный')

    plt.xlabel('Частота (Гц)', fontsize=16)
    plt.ylabel('Модуль спектра', fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(xlim)
    plt.tight_layout()

    save_and_close(fig, save_path)
    if DisplayParams.SHOW_PLOTS:
        plt.show()
    return fig


def plot_filter_mask(freqs, mask, title="", save_path=None, xlim=(-30, 30)):
    """График маски фильтра"""
    fig = plt.figure(figsize=(16, 6))

    plt.plot(freqs, mask.astype(float), 'k-', linewidth=2)
    plt.fill_between(freqs, 0, mask.astype(float), alpha=0.3, color='gray')

    plt.xlabel('Частота (Гц)', fontsize=16)
    plt.ylabel('Коэффициент передачи', fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(xlim)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()

    save_and_close(fig, save_path)
    if DisplayParams.SHOW_PLOTS:
        plt.show()
    return fig


def plot_mse_analysis(x_values, y_values, xlabel, title="", save_path=None):
    """График анализа MSE"""
    fig = plt.figure(figsize=(14, 8))

    plt.plot(x_values, y_values, 'bo-', linewidth=2.5, markersize=10)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('MSE', fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_and_close(fig, save_path)
    if DisplayParams.SHOW_PLOTS:
        plt.show()
    return fig


def plot_results_separate(t, g, u, u_filtered, freqs, G, U, U_filtered,
                          base_title="", save_base=""):
    """Набор отдельных графиков для отчета"""

    # 1. Временная область
    plot_time_domain(
        t, g, u, u_filtered,
        title=f"{base_title}\nСигналы во временной области",
        save_path=f"{OutputParams.figures_dir}/{save_base}_time_all.png"
    )

    # 2. Спектры до фильтрации
    plot_spectrum_magnitude(
        freqs, G, U,
        title=f"{base_title}\nСпектры до фильтрации",
        save_path=f"{OutputParams.figures_dir}/{save_base}_spectrum_before.png"
    )

    # 3. Спектры после фильтрации
    plot_filtered_spectrum(
        freqs, G, U_filtered,
        title=f"{base_title}\nСпектры после фильтрации",
        save_path=f"{OutputParams.figures_dir}/{save_base}_spectrum_after.png"
    )


def ensure_directories():
    """Создание необходимых директорий"""
    os.makedirs(OutputParams.figures_dir, exist_ok=True)
    os.makedirs(OutputParams.audio_dir, exist_ok=True)