"""
Пункт 1.7: Комбинированная фильтрация
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    create_lpf_mask, create_notch_mask, calculate_mse,
    plot_time_domain, plot_spectrum_magnitude, plot_filtered_spectrum,
    plot_filter_mask
)


def create_combined_mask(freqs, lpf_cutoff, notch_center, notch_width):
    """Создание комбинированной маски"""
    mask = np.ones_like(freqs, dtype=bool)
    mask[np.abs(freqs) > lpf_cutoff] = False
    notch_region = (np.abs(freqs) >= (notch_center - notch_width)) & \
                   (np.abs(freqs) <= (notch_center + notch_width))
    mask[notch_region] = False
    return mask


def run():
    """Выполнение пункта 1.7"""

    t = SignalParams.get_time_array()
    freqs, _ = SignalParams.get_freq_array(len(t))

    g = create_rect_pulse(t, SignalParams.a, SignalParams.t1, SignalParams.t2)
    G = fftshift(fft(g))

    b_val = 0.2
    c_val = 0.5
    d_val = 20.0

    print("\n1. Сравнение различных типов фильтров")
    print("-" * 40)

    u, _ = create_noisy_signal(g, t, b=b_val, c=c_val, d=d_val)
    U = fftshift(fft(u))

    # Сравнительный график всех фильтров
    plt.figure(figsize=(16, 10))
    plt.plot(t, g, 'k-', linewidth=2.5, label='Исходный', alpha=0.8)

    # Только ФНЧ
    u_lpf, _, _, _ = apply_freq_filter(u, freqs, lambda f: create_lpf_mask(f, 12))
    mse_lpf = calculate_mse(g, u_lpf)
    plt.plot(t, u_lpf, 'b--', linewidth=1.5, label=f'Только ФНЧ (MSE={mse_lpf:.4f})')

    # Только режектор
    u_notch, _, _, _ = apply_freq_filter(u, freqs, lambda f: create_notch_mask(f, d_val, 2))
    mse_notch = calculate_mse(g, u_notch)
    plt.plot(t, u_notch, 'r--', linewidth=1.5, label=f'Только режектор (MSE={mse_notch:.4f})')

    # Комбинированный
    u_comb, mask, _, U_comb = apply_freq_filter(
        u, freqs, lambda f: create_combined_mask(f, 12, d_val, 2)
    )
    mse_comb = calculate_mse(g, u_comb)
    plt.plot(t, u_comb, 'g-', linewidth=2, label=f'Комбинированный (MSE={mse_comb:.4f})')

    plt.xlabel('Время (с)', fontsize=16)
    plt.ylabel('Амплитуда', fontsize=16)
    plt.title('Сравнение различных типов фильтров', fontsize=18, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([-2, 2])
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task1_7_filters_comparison.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    print("\n2. Детальный анализ комбинированного фильтра")
    print("-" * 40)

    # Временная область
    plot_time_domain(
        t, g, u, u_comb,
        title="Комбинированный фильтр: сигналы во временной области",
        save_path=f"{OutputParams.figures_dir}/task1_7_combined_time.png"
    )

    # Спектры до
    plot_spectrum_magnitude(
        freqs, G, U,
        title="Комбинированный фильтр: спектры до фильтрации",
        save_path=f"{OutputParams.figures_dir}/task1_7_combined_spectrum_before.png"
    )

    # Спектры после
    plot_filtered_spectrum(
        freqs, G, U_comb,
        title="Комбинированный фильтр: спектры после фильтрации",
        save_path=f"{OutputParams.figures_dir}/task1_7_combined_spectrum_after.png"
    )

    # Маска
    plot_filter_mask(
        freqs, mask,
        title="АЧХ комбинированного фильтра (ФНЧ 12 Гц + режектор 20±2 Гц)",
        save_path=f"{OutputParams.figures_dir}/task1_7_combined_mask.png"
    )

    print("\n3. Влияние соотношения шумов")
    print("-" * 40)

    b_values = [0.1, 0.2, 0.5]
    c_values = [0.2, 0.5, 1.0]

    mse_matrix = np.zeros((len(b_values), len(c_values)))

    for i, b_test in enumerate(b_values):
        for j, c_test in enumerate(c_values):
            u, _ = create_noisy_signal(g, t, b=b_test, c=c_test, d=d_val)
            u_filtered, _, _, _ = apply_freq_filter(
                u, freqs, lambda f: create_combined_mask(f, 12, d_val, 2)
            )
            mse = calculate_mse(g, u_filtered)
            mse_matrix[i, j] = mse

    # Тепловая карта
    plt.figure(figsize=(12, 10))
    plt.imshow(mse_matrix, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(label='MSE')
    plt.xlabel('Амплитуда гармонической помехи c', fontsize=16)
    plt.ylabel('Амплитуда белого шума b', fontsize=16)
    plt.title('MSE в зависимости от соотношения шумов', fontsize=18, fontweight='bold')
    plt.xticks(np.arange(len(c_values)), [str(c) for c in c_values])
    plt.yticks(np.arange(len(b_values)), [str(b) for b in b_values])

    for i in range(len(b_values)):
        for j in range(len(c_values)):
            color = 'white' if mse_matrix[i, j] > 0.1 else 'black'
            plt.text(j, i, f'{mse_matrix[i, j]:.4f}',
                    ha='center', va='center', color=color, fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task1_7_mse_heatmap.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # Графики зависимостей
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for i, b_test in enumerate(b_values):
        ax1.plot(c_values, mse_matrix[i, :], 'o-', linewidth=2.5,
                markersize=8, label=f'b={b_test}')
    ax1.set_xlabel('Амплитуда помехи c', fontsize=14)
    ax1.set_ylabel('MSE', fontsize=14)
    ax1.set_title('Зависимость MSE от амплитуды помехи', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    for j, c_test in enumerate(c_values):
        ax2.plot(b_values, mse_matrix[:, j], 'o-', linewidth=2.5,
                markersize=8, label=f'c={c_test}')
    ax2.set_xlabel('Амплитуда шума b', fontsize=14)
    ax2.set_ylabel('MSE', fontsize=14)
    ax2.set_title('Зависимость MSE от амплитуды шума', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task1_7_mse_dependencies.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("ВЫВОДЫ ПО ПУНКТУ 1.7")
    print("="*50)
    print("""
    1. Комбинированный фильтр эффективнее простых
    2. Оптимальные параметры: ФНЧ 12 Гц, режектор 20±2 Гц
    3. MSE растет с увеличением амплитуд шумов
    4. Гармоническая помеха вносит больший вклад
    """)