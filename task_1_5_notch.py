"""
Пункт 1.5: Режекторный фильтр
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    notch_mask, calculate_mse,
    plot_time_three, plot_spectrum_before, plot_spectrum_after,
    plot_mask, plot_mse_line, ensure_directories
)


def run():
    """Выполнение пункта 1.5"""
    ensure_directories()
    np.random.seed(42)

    t = SignalParams.get_time_array()
    freqs, _ = SignalParams.get_freq_array(len(t))

    g = create_rect_pulse(t, SignalParams.a, SignalParams.t1, SignalParams.t2)
    G = fftshift(fft(g))

    c_fixed, d_fixed = 0.5, 20.0

    # влияние ширины режектора
    print("\n[1.5] Влияние ширины режектора (d=20 Гц, c=0.5)")
    print(f"{'Δν, Гц':<10} {'MSE':<15}")
    print("-" * 30)

    widths = [1, 2, 5, 10]
    w_results = []

    for w in widths:
        u = create_noisy_signal(g, t, b=0, c=c_fixed, d=d_fixed)
        U = fftshift(fft(u))

        uf, mask, _, Uf = apply_freq_filter(
            u, freqs, lambda f: notch_mask(f, d_fixed, w)
        )
        mse = calculate_mse(g, uf)
        w_results.append((w, mse))

        print(f"{w:<10} {mse:<15.6f}")

        plot_time_three(
            t, g, u, uf,
            title=f"Режектор: Δν = {w} Гц, d = {d_fixed} Гц",
            save_path=f"{OutputParams.figures_dir}/task1_5_time_w{w}.png",
            ylim=(-0.8, 1.8)
        )

        plot_spectrum_after(
            freqs, G, Uf,
            title=f"Спектр после режектора (Δν={w} Гц)",
            save_path=f"{OutputParams.figures_dir}/task1_5_spec_after_w{w}.png"
        )

        plot_mask(
            freqs, mask,
            title=f"АЧХ режектора (Δν = {w} Гц, центр {d_fixed} Гц)",
            save_path=f"{OutputParams.figures_dir}/task1_5_mask_w{w}.png"
        )

    plot_mse_line(
        [r[0] for r in w_results], [r[1] for r in w_results],
        "Ширина режектора Δν, Гц", "Зависимость MSE от Δν (d=20 Гц)",
        f"{OutputParams.figures_dir}/task1_5_mse_width.png"
    )

    # влияние частоты помехи
    print(f"\n[1.5] Влияние частоты помехи (c=0.5, Δν=2 Гц)")
    print(f"{'d, Гц':<10} {'MSE':<15}")
    print("-" * 30)

    d_vals = [5, 10, 20, 30, 40]
    for d_val in d_vals:
        u = create_noisy_signal(g, t, b=0, c=c_fixed, d=d_val)
        U = fftshift(fft(u))

        uf, mask, _, Uf = apply_freq_filter(
            u, freqs, lambda f: notch_mask(f, d_val, 2)
        )
        mse = calculate_mse(g, uf)

        print(f"{d_val:<10} {mse:<15.6f}")

        if d_val in [5, 20, 40]:
            plot_time_three(
                t, g, u, uf,
                title=f"Режектор: d = {d_val} Гц",
                save_path=f"{OutputParams.figures_dir}/task1_5_time_d{d_val}.png",
                ylim=(-0.8, 1.8)
            )

            xlim_val = max(30, d_val + 15)
            plot_mask(
                freqs, mask,
                title=f"АЧХ режектора (d = {d_val} Гц)",
                save_path=f"{OutputParams.figures_dir}/task1_5_mask_d{d_val}.png",
                xlim=(-xlim_val, xlim_val)
            )

    # влияние амплитуды помехи
    print(f"\n[1.5] Влияние амплитуды помехи (d=20 Гц, Δν=2 Гц)")
    print(f"{'c':<10} {'MSE':<15} {'MSE/c²':<15}")
    print("-" * 40)

    c_vals = [0.1, 0.5, 1.0, 2.0]
    c_results = []

    b_small = 0.01

    for c_val in c_vals:
        np.random.seed(42)
        u = create_noisy_signal(g, t, b=b_small, c=c_val, d=d_fixed)
        U = fftshift(fft(u))

        uf, _, _, Uf = apply_freq_filter(
            u, freqs, lambda f: notch_mask(f, d_fixed, 2)
        )

        mse = calculate_mse(g, uf)
        c_results.append((c_val, mse))

        ratio = mse / (c_val ** 2) if c_val > 0 else 0
        print(f"{c_val:<10} {mse:<15.6f} {ratio:.4f}")

        if c_val in [0.1, 0.5, 2.0]:
            plot_time_three(
                t, g, u, uf,
                title=f"Режектор: c = {c_val}",
                save_path=f"{OutputParams.figures_dir}/task1_5_time_c{c_val}.png",
                ylim=(-1.5, 2.5) if c_val > 1 else (-0.8, 1.8)
            )

    # зависимость MSE от амплитуды помехи
    c_vals_plot = [r[0] for r in c_results]
    mses_c = [r[1] for r in c_results]

    plt.figure(figsize=(14, 9))
    plt.plot(c_vals_plot, mses_c, 'go-', linewidth=2.5, markersize=10, label='Эксперимент')

    c_fit = np.linspace(0.1, 2.0, 100)
    mse_fit = mses_c[0] * (c_fit / c_vals_plot[0]) ** 2
    plt.plot(c_fit, mse_fit, 'r--', linewidth=2, alpha=0.7, label='∝ c²')

    plt.xlabel('Амплитуда помехи c', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title('Зависимость MSE от амплитуды помехи', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task1_5_mse_c.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # результаты для отчета
    print("\n[ТАБЛИЦА 1] Зависимость MSE от ширины режектора:")
    for w, mse in w_results:
        print(f"{w} Гц & {mse:.6f} \\\\")

    print("\n[ТАБЛИЦА 2] Зависимость MSE от амплитуды помехи:")
    for c_val, mse in c_results:
        print(f"{c_val} & {mse:.6f} & {mse/(c_val**2):.4f} \\\\")