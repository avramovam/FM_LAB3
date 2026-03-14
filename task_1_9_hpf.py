"""
Пункт 1.9: Фильтр верхних частот (ФВЧ)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    create_hpf_mask, plot_results_separate, calculate_mse,
    plot_filter_mask, plot_mse_analysis
)


def run():
    """Выполнение пункта 1.9"""

    t = SignalParams.get_time_array()
    freqs, _ = SignalParams.get_freq_array(len(t))

    g = create_rect_pulse(t, SignalParams.a, SignalParams.t1, SignalParams.t2)
    G = fftshift(fft(g))

    b_val = 0.1
    u, _ = create_noisy_signal(g, t, b=b_val, c=0, d=0)
    U = fftshift(fft(u))

    print("\nИсследование влияния ФВЧ")
    print("-" * 40)

    hpf_cutoffs = [0.5, 2, 5, 10, 20]
    showcase_cutoffs = [0.5, 5, 20]
    results = []

    for nu0 in hpf_cutoffs:
        print(f"\nЧастота среза ν₀ = {nu0} Гц")

        u_filtered, mask, _, U_filtered = apply_freq_filter(
            u, freqs, lambda f: create_hpf_mask(f, nu0)
        )

        mse = calculate_mse(g, u_filtered)
        results.append({'cutoff': nu0, 'mse': mse})
        print(f"  MSE = {mse:.6f}")

        if nu0 in showcase_cutoffs:
            plot_results_separate(
                t, g, u, u_filtered, freqs, G, U, U_filtered,
                base_title=f"ФВЧ: ν₀ = {nu0} Гц",
                save_base=f"task1_9_hpf_{nu0}"
            )
            plot_filter_mask(
                freqs, mask,
                title=f"Маска ФВЧ (ν₀ = {nu0} Гц)",
                save_path=f"{OutputParams.figures_dir}/task1_9_mask_{nu0}.png"
            )

    # График MSE
    cutoffs = [r['cutoff'] for r in results]
    mses = [r['mse'] for r in results]
    plot_mse_analysis(
        cutoffs, mses,
        xlabel="Частота среза ФВЧ (Гц)",
        title="Зависимость MSE от частоты среза ФВЧ",
        save_path=f"{OutputParams.figures_dir}/task1_9_mse_vs_cutoff.png"
    )

    print("\n" + "="*50)
    print("ВЫВОДЫ ПО ПУНКТУ 1.9")
    print("="*50)
    print("""
    1. ФВЧ удаляет низкочастотный сигнал
    2. При малых ν₀ - сильные искажения
    3. При ν₀ > 10 Гц - полное подавление
    4. ФВЧ не подходит для данной задачи
    """)