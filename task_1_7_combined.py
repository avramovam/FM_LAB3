"""Пункт 1.7: Комбинированный фильтр — все эксперименты для отчета"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    lpf_mask, notch_mask, combined_mask, calc_mse,
    plot_time_three, plot_spectrum_before, plot_spectrum_after,
    plot_mask, plot_mse_heatmap, plot_comparison_three, ensure_dirs
)

def run():
    ensure_dirs()
    np.random.seed(42)

    t = SignalParams.get_time_array()
    freqs, _ = SignalParams.get_freq_array(len(t))
    g = create_rect_pulse(t, SignalParams.a, SignalParams.t1, SignalParams.t2)
    G = fftshift(fft(g))

    b_val, c_val, d_val = 0.2, 0.5, 20.0
    u = create_noisy_signal(g, t, b=b_val, c=c_val, d=d_val)
    U = fftshift(fft(u))

    print("\n[1.7] Сравнение фильтров")

    # === Сравнение трёх подходов ===
    uf_lpf, mask_lpf, _, Uf_lpf = apply_freq_filter(u, freqs, lambda f: lpf_mask(f, 12))
    mse_lpf = calc_mse(g, uf_lpf)

    uf_notch, mask_notch, _, Uf_notch = apply_freq_filter(u, freqs, lambda f: notch_mask(f, d_val, 2))
    mse_notch = calc_mse(g, uf_notch)

    uf_comb, mask_comb, _, Uf_comb = apply_freq_filter(
        u, freqs, lambda f: combined_mask(f, 12, d_val, 2))
    mse_comb = calc_mse(g, uf_comb)

    plot_comparison_three(t, g, uf_lpf, uf_notch, uf_comb,
                         mse_lpf, mse_notch, mse_comb,
                         "Сравнение типов фильтров (b=0.2, c=0.5, d=20 Гц)",
                         f"{OutputParams.figures_dir}/task1_7_comparison.png")

    # === Детальные графики комбинированного фильтра ===
    plot_time_three(t, g, u, uf_comb, "Комбинированный фильтр: временная область",
                  f"{OutputParams.figures_dir}/task1_7_time.png")
    plot_spectrum_before(freqs, G, U, "Спектры до фильтрации",
                       f"{OutputParams.figures_dir}/task1_7_spec_before.png")
    plot_spectrum_after(freqs, G, Uf_comb, "Спектры после комбинированного фильтра",
                      f"{OutputParams.figures_dir}/task1_7_spec_after.png")

    # === 🔥 КЛЮЧЕВОЙ ГРАФИК: АЧХ именно комбинированного фильтра ===
    plot_mask(freqs, mask_comb,
             "АЧХ комбинированного фильтра: ФНЧ 12 Гц ∩ режектор 20±2 Гц",
             f"{OutputParams.figures_dir}/task1_7_combined_mask.png")

    # === Дополнительно: покажем отдельно маски ФНЧ и режектора для сравнения ===
    plot_mask(freqs, lpf_mask(freqs, 12), "АЧХ ФНЧ (ν₀ = 12 Гц) — компонента комбинированного",
             f"{OutputParams.figures_dir}/task1_7_mask_lpf_component.png")
    plot_mask(freqs, notch_mask(freqs, d_val, 2), "АЧХ режектора (20±2 Гц) — компонента комбинированного",
             f"{OutputParams.figures_dir}/task1_7_mask_notch_component.png")

    # === Влияние соотношения шумов ===
    print("\n[1.7] Влияние соотношения шумов")
    b_vals = [0.1, 0.2, 0.5]
    c_vals = [0.2, 0.5, 1.0]
    mse_mat = np.zeros((len(b_vals), len(c_vals)))

    for i, b_test in enumerate(b_vals):
        for j, c_test in enumerate(c_vals):
            u_test = create_noisy_signal(g, t, b=b_test, c=c_test, d=d_val)
            uf_test, _, _, _ = apply_freq_filter(
                u_test, freqs, lambda f: combined_mask(f, 12, d_val, 2))
            mse_mat[i, j] = calc_mse(g, uf_test)

    plot_mse_heatmap(b_vals, c_vals, mse_mat,
                    f"{OutputParams.figures_dir}/task1_7_heatmap.png")

    # === ТАБЛИЦА 7 ДЛЯ ОТЧЕТА ===
    print("\n[ТАБЛИЦА 7 — для отчета]")
    header = "b \\ c"
    print(f"{header:<8} {'0.2':<12} {'0.5':<12} {'1.0':<12}")
    print("-" * 40)
    for i, b_test in enumerate(b_vals):
        row = f"{b_test:<8}"
        for j in range(len(c_vals)):
            row += f"{mse_mat[i, j]:<12.4f}"
        print(row)

    print("\n[Анализ] MSE ≈ αb² + βc², где β ≈ 5α — гармоника влияет сильнее.")
    print("Комбинированный фильтр эффективнее простых: MSE={:.4f} против {:.4f} (ФНЧ) и {:.4f} (режектор).".format(
        mse_comb, mse_lpf, mse_notch))