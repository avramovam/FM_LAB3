"""
Пункт 1.7: Комбинированная фильтрация
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    lpf_mask, notch_mask, combined_mask, calculate_mse,
    plot_time_three, plot_spectrum_before, plot_spectrum_after,
    plot_mask, plot_mse_heatmap, plot_comparison_filters, ensure_directories
)


def run():
    """Выполнение пункта 1.7"""
    ensure_directories()
    np.random.seed(42)

    t = SignalParams.get_time_array()
    freqs, _ = SignalParams.get_freq_array(len(t))

    g = create_rect_pulse(t, SignalParams.a, SignalParams.t1, SignalParams.t2)
    G = fftshift(fft(g))

    b_val, c_val, d_val = 0.2, 0.5, 20.0
    u = create_noisy_signal(g, t, b=b_val, c=c_val, d=d_val)
    U = fftshift(fft(u))

    print("\n[1.7] Сравнение фильтров")
    print("-" * 40)

    # применение трех типов фильтров
    u_lpf, mask_lpf, _, U_lpf = apply_freq_filter(
        u, freqs, lambda f: lpf_mask(f, 12)
    )
    mse_lpf = calculate_mse(g, u_lpf)

    u_notch, mask_notch, _, U_notch = apply_freq_filter(
        u, freqs, lambda f: notch_mask(f, d_val, 2)
    )
    mse_notch = calculate_mse(g, u_notch)

    u_comb, mask_comb, _, U_comb = apply_freq_filter(
        u, freqs, lambda f: combined_mask(f, 12, d_val, 2)
    )
    mse_comb = calculate_mse(g, u_comb)

    print(f"ФНЧ: MSE = {mse_lpf:.4f}")
    print(f"Режектор: MSE = {mse_notch:.4f}")
    print(f"Комбинированный: MSE = {mse_comb:.4f}")

    # сравнение фильтров во временной области
    plot_comparison_filters(
        t, g, u_lpf, u_notch, u_comb,
        mse_lpf, mse_notch, mse_comb,
        title="Сравнение типов фильтров (b=0.2, c=0.5, d=20 Гц)",
        save_path=f"{OutputParams.figures_dir}/task1_7_filters_comparison.png",
        xlim=(-2, 2)
    )

    # комбинированный фильтр: временная область
    plot_time_three(
        t, g, u, u_comb,
        title="Комбинированный фильтр: временная область",
        save_path=f"{OutputParams.figures_dir}/task1_7_time.png",
        ylim=(-0.3, 1.3)
    )

    # спектры до фильтрации
    plot_spectrum_before(
        freqs, G, U,
        title="Спектры до фильтрации",
        save_path=f"{OutputParams.figures_dir}/task1_7_spec_before.png"
    )

    # спектры после комбинированной фильтрации
    plot_spectrum_after(
        freqs, G, U_comb,
        title="Спектры после комбинированного фильтра",
        save_path=f"{OutputParams.figures_dir}/task1_7_spec_after.png"
    )

    # АЧХ комбинированного фильтра (ФНЧ 25 Гц для наглядности)
    mask_comb_viz = combined_mask(freqs, 25, d_val, 2)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(freqs, mask_comb_viz.astype(float), color='#000000', linewidth=3,
            label='Комбинированный фильтр')
    ax.fill_between(freqs, 0, mask_comb_viz.astype(float), alpha=0.3, color='gray')

    ax.axvspan(d_val-2, d_val+2, alpha=0.3, color='red', label='Режектор 20±2 Гц')
    ax.axvspan(-d_val-2, -d_val+2, alpha=0.3, color='red')
    ax.axvline(x=25, color='blue', linestyle='--', linewidth=2, label='ФНЧ 25 Гц')
    ax.axvline(x=-25, color='blue', linestyle='--', linewidth=2)

    ax.set_xlabel('Частота, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('Коэффициент передачи', fontsize=20, fontweight='bold')
    ax.set_title('АЧХ комбинированного фильтра: ФНЧ + режектор',
                fontsize=22, fontweight='bold', pad=15)
    ax.legend(fontsize=18, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.2, length=6, labelsize=18)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1])

    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task1_7_combined_mask.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # компоненты комбинированного фильтра
    plot_mask(
        freqs, lpf_mask(freqs, 12),
        title="АЧХ ФНЧ (компонента комбинированного)",
        save_path=f"{OutputParams.figures_dir}/task1_7_mask_lpf_component.png"
    )

    plot_mask(
        freqs, notch_mask(freqs, d_val, 2),
        title="АЧХ режектора (компонента комбинированного)",
        save_path=f"{OutputParams.figures_dir}/task1_7_mask_notch_component.png"
    )

    # влияние соотношения белого шума и гармонической помехи
    print("\n[1.7] Влияние соотношения шумов")
    b_vals = [0.1, 0.2, 0.5]
    c_vals = [0.2, 0.5, 1.0]
    mse_mat = np.zeros((len(b_vals), len(c_vals)))

    for i, b_test in enumerate(b_vals):
        for j, c_test in enumerate(c_vals):
            u_test = create_noisy_signal(g, t, b=b_test, c=c_test, d=d_val)
            uf_test, _, _, _ = apply_freq_filter(
                u_test, freqs, lambda f: combined_mask(f, 12, d_val, 2)
            )
            mse_mat[i, j] = calculate_mse(g, uf_test)

    plot_mse_heatmap(b_vals, c_vals, mse_mat,
                    f"{OutputParams.figures_dir}/task1_7_heatmap.png")

    # результаты для отчета
    print("\n[ТАБЛИЦА] Зависимость MSE от b и c:")
    header = "b \\ c"
    print(f"{header:<8} {'0.2':<12} {'0.5':<12} {'1.0':<12}")
    print("-" * 40)
    for i, b_test in enumerate(b_vals):
        row = f"{b_test:<8}"
        for j in range(len(c_vals)):
            row += f"{mse_mat[i, j]:<12.4f}"
        print(row)


if __name__ == "__main__":
    run()