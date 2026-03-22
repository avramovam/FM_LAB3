"""
Пункт 1.9: Фильтр верхних частот (ФВЧ)
ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
import numpy as np
from scipy.fft import fft, fftshift
from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    hpf_mask, calculate_mse,
    plot_time_three, plot_mask, plot_mse_line, ensure_directories
)


def run():
    """Выполнение пункта 1.9"""
    ensure_directories()
    np.random.seed(42)

    t = SignalParams.get_time_array()
    freqs, _ = SignalParams.get_freq_array(len(t))

    g = create_rect_pulse(t, SignalParams.a, SignalParams.t1, SignalParams.t2)

    b_val = 0.1
    u = create_noisy_signal(g, t, b=b_val)

    print("\n[1.9] ФВЧ: влияние частоты среза")
    print(f"{'ν₀, Гц':<10} {'MSE':<15} {'Характеристика'}")
    print("-" * 40)

    cutoffs = [0.5, 2, 5, 10, 20]
    results = []

    for nu0 in cutoffs:
        uf, mask, _, _ = apply_freq_filter(
            u, freqs, lambda f: hpf_mask(f, nu0)
        )
        mse = calculate_mse(g, uf)
        results.append((nu0, mse))

        desc = "форма угадывается" if nu0 < 1 else ("только фронты" if nu0 < 5 else "сигнал подавлен")
        print(f"{nu0:<10} {mse:<15.6f} {desc}")

        # График для КАЖДОГО ν₀
        plot_time_three(
            t, g, u, uf,
            title=f"ФВЧ: ν₀ = {nu0} Гц",
            save_path=f"{OutputParams.figures_dir}/task1_9_time_nu0{nu0}.png",
            ylim=(-1.0, 2.0)
        )

        plot_mask(
            freqs, mask,
            title=f"АЧХ ФВЧ (ν₀ = {nu0} Гц)",
            save_path=f"{OutputParams.figures_dir}/task1_9_mask_nu0{nu0}.png"
        )

    plot_mse_line(
        [r[0] for r in results], [r[1] for r in results],
        "Частота среза ФВЧ ν₀, Гц", "Зависимость MSE от ν₀ (ФВЧ)",
        f"{OutputParams.figures_dir}/task1_9_mse.png"
    )

    print("\n[ВЫВОДЫ 1.9]")
    print("• ФВЧ разрушает низкочастотный сигнал")
    print("• При ν₀ > 5 Гц — полное подавление")
    print("• Неприменим для данной задачи")