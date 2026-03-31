"""
Вспомогательные функции для всех заданий
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
import os
from config import OutputParams, DisplayParams

# контрастная палитра для черно-белой печати
COLORS = {
    'orig': '#000000',      # черный
    'noisy': '#D62728',     # красный
    'filt': '#1F77B4',      # синий
    'mask': '#000000',      # черный
    'highlight': '#FF7F0E', # оранжевый
    'green': '#2CA02C',     # зеленый
    'gray': '#7F7F7F'       # серый
}

LINE_STYLES = {
    'orig': '-',
    'noisy': '--',
    'filt': '-.'
}

LINE_WIDTHS = {
    'orig': 2.5,
    'noisy': 2.0,
    'filt': 2.5
}

# настройки для печати
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.2


def create_rect_pulse(t, a, t1, t2):
    """Формирует прямоугольный импульс амплитуды a на интервале [t1, t2]"""
    g = np.zeros_like(t)
    g[(t >= t1) & (t <= t2)] = a
    return g


def create_noisy_signal(g, t, b=0, c=0, d=0):
    """Добавляет белый шум и гармоническую помеху к сигналу"""
    u = g.copy()
    if b > 0:
        u += b * (2 * np.random.random(len(t)) - 1)
    if c > 0 and d > 0:
        u += c * np.sin(2 * np.pi * d * t)
    return u


def apply_freq_filter(signal, freqs, mask_func):
    """Применяет фильтр с заданной маской в частотной области"""
    S = fftshift(fft(signal))
    mask = mask_func(freqs)
    S_f = S * mask
    return np.real(ifft(ifftshift(S_f))), mask, S, S_f


def lpf_mask(freqs, cutoff):
    """Маска фильтра нижних частот"""
    return np.abs(freqs) <= cutoff


def hpf_mask(freqs, cutoff):
    """Маска фильтра верхних частот"""
    return np.abs(freqs) >= cutoff


def notch_mask(freqs, center, width):
    """Маска режекторного фильтра"""
    m = np.ones_like(freqs, dtype=bool)
    m[(np.abs(freqs) >= (center - width)) & (np.abs(freqs) <= (center + width))] = False
    return m


def combined_mask(freqs, lpf_cut, notch_c, notch_w):
    """Комбинированная маска: ФНЧ + режектор"""
    return lpf_mask(freqs, lpf_cut) & notch_mask(freqs, notch_c, notch_w)


def bandpass_mask(freqs, low, high):
    """Маска полосового фильтра"""
    m = np.zeros_like(freqs, dtype=bool)
    m[(np.abs(freqs) >= low) & (np.abs(freqs) <= high)] = True
    return m


def calculate_mse(original, filtered):
    """Среднеквадратичная ошибка"""
    return np.mean((original - filtered) ** 2)


def _save(fig, path):
    """Сохраняет фигуру, если сохранение включено"""
    if path and OutputParams.save_figures:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=OutputParams.figures_dpi, bbox_inches='tight')
    if not DisplayParams.SHOW_PLOTS:
        plt.close(fig)


def _style(ax, xlabel, ylabel, title, legend_loc='upper right'):
    """Стиль оформления графиков"""
    ax.set_xlabel(xlabel, fontsize=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=22, fontweight='bold', pad=15)
    ax.legend(fontsize=18, loc=legend_loc, framealpha=0.9, fancybox=True)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.2, length=6, labelsize=18)


def plot_time_three(t, g, u, uf, title, save_path, xlim=(-3, 3), ylim=None):
    """Сравнение трех сигналов во временной области"""
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(t, g, color=COLORS['orig'], linewidth=LINE_WIDTHS['orig'],
            linestyle=LINE_STYLES['orig'], label='Исходный g(t)', alpha=0.9, zorder=1)

    ax.plot(t, u, color=COLORS['noisy'], linewidth=LINE_WIDTHS['noisy'],
            linestyle=LINE_STYLES['noisy'], alpha=0.85, label='Зашумленный u(t)', zorder=2)

    ax.plot(t, uf, color=COLORS['filt'], linewidth=LINE_WIDTHS['filt'],
            linestyle=LINE_STYLES['filt'], label='Отфильтрованный', alpha=0.95, zorder=3)

    _style(ax, 'Время, с', 'Амплитуда', title)
    ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    _save(fig, save_path)


def plot_spectrum_before(freqs, G, U, title, save_path, xlim=(-30, 30)):
    """Спектры сигналов до фильтрации"""
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(freqs, np.abs(G), color=COLORS['orig'], linewidth=3.0,
            linestyle='-', label='|G(ν)| исходный', alpha=0.9)
    ax.plot(freqs, np.abs(U), color=COLORS['noisy'], linewidth=2.5,
            linestyle='--', label='|U(ν)| зашумленный', alpha=0.85)

    _style(ax, 'Частота, Гц', '|Спектр|', title)
    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    _save(fig, save_path)


def plot_spectrum_after(freqs, G, Uf, title, save_path, xlim=(-30, 30)):
    """Спектры сигналов после фильтрации"""
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(freqs, np.abs(G), color=COLORS['orig'], linewidth=3.0,
            linestyle='-', label='|G(ν)| исходный', alpha=0.9)
    ax.plot(freqs, np.abs(Uf), color=COLORS['filt'], linewidth=3.0,
            linestyle='-.', label='|U_f(ν)| фильтрованный', alpha=0.95)

    _style(ax, 'Частота, Гц', '|Спектр|', title)
    ax.set_xlim(xlim)
    ax.set_ylim(bottom=0)
    _save(fig, save_path)


def plot_mask(freqs, mask, title, save_path, xlim=(-30, 30)):
    """АЧХ фильтра"""
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(freqs, mask.astype(float), color=COLORS['mask'], linewidth=3,
            label='Коэффициент передачи')
    ax.fill_between(freqs, 0, mask.astype(float), alpha=0.3, color='gray')

    _style(ax, 'Частота, Гц', '', title, legend_loc='upper right')
    ax.set_xlim(xlim)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'], fontsize=18)
    _save(fig, save_path)


def plot_mse_line(x, y, xlabel, title, save_path, log_y=False):
    """График зависимости MSE от параметра"""
    fig, ax = plt.subplots(figsize=(16, 10))

    if log_y:
        ax.semilogy(x, y, 'bo-', linewidth=3.0, markersize=12, label='Эксперимент')
        x_fit = np.array(x)
        y_fit = y[0] * (x_fit / x[0])**2
        ax.plot(x_fit, y_fit, 'r--', linewidth=3, alpha=0.8, label='∝ x²')
    else:
        ax.plot(x, y, 'bo-', linewidth=3.0, markersize=12)

    _style(ax, xlabel, 'MSE', title)
    ax.grid(True, alpha=0.4)
    if log_y:
        ax.legend(fontsize=18, framealpha=0.9)

    ax.set_ylim(bottom=0)
    if not log_y:
        ax.set_ylim(top=max(y) * 1.2)

    _save(fig, save_path)


def plot_mse_heatmap(b_vals, c_vals, mse_mat, save_path):
    """Тепловая карта MSE при разных соотношениях шумов"""
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(mse_mat, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(im, label='MSE', ax=ax, shrink=0.8)

    ax.set_xlabel('Амплитуда гармоники c', fontsize=20, fontweight='bold')
    ax.set_ylabel('Амплитуда шума b', fontsize=20, fontweight='bold')
    ax.set_title('MSE при разных соотношениях шумов', fontsize=22, fontweight='bold', pad=15)

    ax.set_xticks(np.arange(len(c_vals)))
    ax.set_yticks(np.arange(len(b_vals)))
    ax.set_xticklabels([f'{c}' for c in c_vals], fontsize=18)
    ax.set_yticklabels([f'{b}' for b in b_vals], fontsize=18)

    for i in range(len(b_vals)):
        for j in range(len(c_vals)):
            col = 'white' if mse_mat[i, j] > 0.1 else 'black'
            ax.text(j, i, f'{mse_mat[i, j]:.4f}', ha='center', va='center',
                   color=col, fontsize=16, fontweight='bold')

    _save(fig, save_path)


def plot_comparison_filters(t, g, u_lpf, u_notch, u_comb,
                           mse_lpf, mse_notch, mse_comb,
                           title, save_path, xlim=(-2, 2)):
    """Сравнение трех фильтров на одном графике"""
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(t, u_comb, color=COLORS['green'], linewidth=3.0, linestyle='-',
            label=f'Комбинированный (MSE={mse_comb:.4f})', alpha=0.9, zorder=1)

    ax.plot(t, u_notch, color=COLORS['noisy'], linewidth=2.5, linestyle='--',
            label=f'Режектор (MSE={mse_notch:.4f})', alpha=0.9, zorder=2)

    ax.plot(t, u_lpf, color=COLORS['filt'], linewidth=3.5, linestyle=':',
            label=f'ФНЧ (MSE={mse_lpf:.4f})', alpha=1.0, zorder=3)

    ax.plot(t, g, color=COLORS['gray'], linewidth=2.0, linestyle='-',
            label='Исходный g(t)', alpha=0.5, zorder=0)

    _style(ax, 'Время, с', 'Амплитуда', title)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.3, 1.3)
    _save(fig, save_path)


def ensure_directories():
    """Создает необходимые директории для сохранения результатов"""
    os.makedirs(OutputParams.figures_dir, exist_ok=True)
    os.makedirs(OutputParams.audio_dir, exist_ok=True)