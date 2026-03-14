"""
Пункт 1.3: Фильтр нижних частот (ФНЧ)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    create_lpf_mask, plot_results_separate, calculate_mse,
    plot_mse_analysis, plot_filter_mask
)


def run():
    """Выполнение пункта 1.3"""

    # Получение параметров
    t = SignalParams.get_time_array()
    freqs, _ = SignalParams.get_freq_array(len(t))

    # Создание чистого сигнала
    g = create_rect_pulse(t, SignalParams.a, SignalParams.t1, SignalParams.t2)
    G = fftshift(fft(g))

    print("\nИсследование 1: Влияние частоты среза ФНЧ")
    print("-" * 40)

    # Параметры для исследования
    b_fixed = 0.2
    cutoff_freqs = [2, 5, 10, 15, 20, 30]
    showcase_cutoffs = [5, 10, 20]

    results = []

    for nu0 in cutoff_freqs:
        print(f"\nЧастота среза ν₀ = {nu0} Гц")

        u, _ = create_noisy_signal(g, t, b=b_fixed, c=0, d=0)
        U = fftshift(fft(u))

        u_filtered, mask, _, U_filtered = apply_freq_filter(
            u, freqs, lambda f: create_lpf_mask(f, nu0)
        )

        mse = calculate_mse(g, u_filtered)
        results.append({'cutoff': nu0, 'mse': mse})
        print(f"  MSE = {mse:.6f}")

        if nu0 in showcase_cutoffs:
            plot_results_separate(
                t, g, u, u_filtered, freqs, G, U, U_filtered,
                base_title=f"ФНЧ: ν₀ = {nu0} Гц",
                save_base=f"task1_3_cutoff_{nu0}"
            )
            plot_filter_mask(
                freqs, mask,
                title=f"Маска ФНЧ (ν₀ = {nu0} Гц)",
                save_path=f"{OutputParams.figures_dir}/task1_3_mask_{nu0}.png"
            )

    # График MSE
    cutoffs = [r['cutoff'] for r in results]
    mses = [r['mse'] for r in results]
    plot_mse_analysis(
        cutoffs, mses,
        xlabel="Частота среза (Гц)",
        title=f"Зависимость MSE от частоты среза (b = {b_fixed})",
        save_path=f"{OutputParams.figures_dir}/task1_3_mse_vs_cutoff.png"
    )

    print("\nИсследование 2: Влияние уровня шума")
    print("-" * 40)

    nu0_fixed = 10
    b_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    showcase_b = [0.1, 0.5]

    results_b = []

    for b_val in b_values:
        print(f"\nУровень шума b = {b_val}")

        u, _ = create_noisy_signal(g, t, b=b_val, c=0, d=0)
        U = fftshift(fft(u))

        u_filtered, mask, _, U_filtered = apply_freq_filter(
            u, freqs, lambda f: create_lpf_mask(f, nu0_fixed)
        )

        mse = calculate_mse(g, u_filtered)
        results_b.append({'b': b_val, 'mse': mse})
        print(f"  MSE = {mse:.6f}")

        if b_val in showcase_b:
            plot_results_separate(
                t, g, u, u_filtered, freqs, G, U, U_filtered,
                base_title=f"ФНЧ: b = {b_val}, ν₀ = {nu0_fixed} Гц",
                save_base=f"task1_3_b_{b_val}"
            )

    # График MSE от уровня шума
    b_vals = [r['b'] for r in results_b]
    mses_b = [r['mse'] for r in results_b]

    plt.figure(figsize=(14, 8))
    plt.semilogy(b_vals, mses_b, 'ro-', linewidth=2.5, markersize=10)
    plt.xlabel('Амплитуда шума b', fontsize=16)
    plt.ylabel('MSE (лог. шкала)', fontsize=16)
    plt.title(f'Зависимость MSE от уровня шума (ν₀ = {nu0_fixed} Гц)',
              fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task1_3_mse_vs_b.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("ВЫВОДЫ ПО ПУНКТУ 1.3")
    print("="*50)
    print("""
    1. Оптимальная частота среза ФНЧ: 5-10 Гц
    2. При ν₀ < 5 Гц - искажение фронтов импульса
    3. При ν₀ > 20 Гц - пропускание шума
    4. ФНЧ эффективен для белого шума
    """)