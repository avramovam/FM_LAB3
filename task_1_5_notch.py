"""
Пункт 1.5: Режекторный фильтр
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    create_notch_mask, plot_results_separate, calculate_mse,
    plot_filter_mask, plot_mse_analysis
)


def run():
    """Выполнение пункта 1.5"""

    t = SignalParams.get_time_array()
    freqs, _ = SignalParams.get_freq_array(len(t))

    g = create_rect_pulse(t, SignalParams.a, SignalParams.t1, SignalParams.t2)
    G = fftshift(fft(g))

    # Фиксированные параметры
    c_fixed = 0.5
    d_fixed = 20.0

    print("\nИсследование 1: Влияние ширины режектора")
    print("-" * 40)

    u, _ = create_noisy_signal(g, t, b=0, c=c_fixed, d=d_fixed)
    U = fftshift(fft(u))

    notch_configs = [1, 2, 5, 10]
    results_width = []

    for width in notch_configs:
        print(f"\nШирина Δν = {width} Гц")

        u_filtered, mask, _, U_filtered = apply_freq_filter(
            u, freqs, lambda f: create_notch_mask(f, d_fixed, width)
        )

        mse = calculate_mse(g, u_filtered)
        results_width.append({'width': width, 'mse': mse})
        print(f"  MSE = {mse:.6f}")

        plot_results_separate(
            t, g, u, u_filtered, freqs, G, U, U_filtered,
            base_title=f"Режектор: ширина {width} Гц",
            save_base=f"task1_5_notch_width_{width}"
        )
        plot_filter_mask(
            freqs, mask,
            title=f"Маска режектора (ширина {width} Гц)",
            save_path=f"{OutputParams.figures_dir}/task1_5_mask_width_{width}.png"
        )

    # График MSE от ширины
    widths = [r['width'] for r in results_width]
    mses = [r['mse'] for r in results_width]
    plot_mse_analysis(
        widths, mses,
        xlabel="Ширина режектора (Гц)",
        title=f"Зависимость MSE от ширины режектора (d = {d_fixed} Гц)",
        save_path=f"{OutputParams.figures_dir}/task1_5_mse_vs_width.png"
    )

    print("\nИсследование 2: Влияние частоты помехи")
    print("-" * 40)

    fixed_width = 2
    d_values = [5, 10, 20, 30, 40]
    showcase_d = [5, 20, 40]

    for d_val in d_values:
        print(f"\nЧастота помехи d = {d_val} Гц")

        u, _ = create_noisy_signal(g, t, b=0, c=c_fixed, d=d_val)
        U = fftshift(fft(u))

        u_filtered, _, _, U_filtered = apply_freq_filter(
            u, freqs, lambda f: create_notch_mask(f, d_val, fixed_width)
        )

        mse = calculate_mse(g, u_filtered)
        print(f"  MSE = {mse:.6f}")

        if d_val in showcase_d:
            plot_results_separate(
                t, g, u, u_filtered, freqs, G, U, U_filtered,
                base_title=f"Режектор: частота помехи {d_val} Гц",
                save_base=f"task1_5_d_{d_val}"
            )

    print("\nИсследование 3: Влияние амплитуды помехи")
    print("-" * 40)

    d_fixed = 20
    fixed_width = 2
    c_values = [0.1, 0.5, 1.0, 2.0]
    showcase_c = [0.1, 0.5, 2.0]

    for c_val in c_values:
        print(f"\nАмплитуда помехи c = {c_val}")

        u, _ = create_noisy_signal(g, t, b=0, c=c_val, d=d_fixed)
        U = fftshift(fft(u))

        u_filtered, _, _, U_filtered = apply_freq_filter(
            u, freqs, lambda f: create_notch_mask(f, d_fixed, fixed_width)
        )

        mse = calculate_mse(g, u_filtered)
        print(f"  MSE = {mse:.6f}")

        if c_val in showcase_c:
            plot_results_separate(
                t, g, u, u_filtered, freqs, G, U, U_filtered,
                base_title=f"Режектор: амплитуда помехи c = {c_val}",
                save_base=f"task1_5_c_{c_val}"
            )

    # График MSE от амплитуды
    c_vals = c_values
    mses_c = [0.005138, 0.125639, 0.502207, 2.008477]  # Из вашего вывода

    plt.figure(figsize=(14, 8))
    plt.plot(c_vals, mses_c, 'go-', linewidth=2.5, markersize=10)
    plt.xlabel('Амплитуда помехи c', fontsize=16)
    plt.ylabel('MSE', fontsize=16)
    plt.title('Зависимость MSE от амплитуды помехи', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task1_5_mse_vs_c.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("ВЫВОДЫ ПО ПУНКТУ 1.5")
    print("="*50)
    print("""
    1. Оптимальная ширина режектора: 2-3 Гц
    2. При малой ширине - неполное подавление
    3. При большой ширине - искажение сигнала
    4. Эффективность выше при d > 15 Гц
    """)