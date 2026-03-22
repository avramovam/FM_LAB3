"""Пункт 1.9: ФВЧ — все эксперименты для отчета"""
import numpy as np
from scipy.fft import fft, fftshift
from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    hpf_mask, calc_mse,
    plot_time_three, plot_mask, plot_mse_line, ensure_dirs
)

def run():
    ensure_dirs()
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
        uf, mask, _, _ = apply_freq_filter(u, freqs, lambda f: hpf_mask(f, nu0))
        mse = calc_mse(g, uf)
        results.append((nu0, mse))

        desc = "форма угадывается" if nu0 < 1 else ("только фронты" if nu0 < 5 else "сигнал подавлен")
        print(f"{nu0:<10} {mse:<15.6f} {desc}")

        # График для КАЖДОГО ν₀
        plot_time_three(t, g, u, uf, f"ФВЧ: $\\nu_0$ = {nu0} Гц",
                      f"{OutputParams.figures_dir}/task1_9_time_nu0{nu0}.png")
        plot_mask(freqs, mask, f"АЧХ ФВЧ ($\\nu_0$ = {nu0} Гц)",
                 f"{OutputParams.figures_dir}/task1_9_mask_nu0{nu0}.png")

    plot_mse_line([r[0] for r in results], [r[1] for r in results],
                 "Частота среза ФВЧ $\\nu_0$, Гц", "Зависимость MSE от $\\nu_0$ (ФВЧ)",
                 f"{OutputParams.figures_dir}/task1_9_mse.png")

    print("\n[Анализ] ФВЧ удаляет низкочастотный сигнал — для прямоугольного импульса это разрушительно.")
    print("При ν₀ > 5 Гц полезный сигнал полностью подавлен, MSE выходит на плато ≈ 0.2.")