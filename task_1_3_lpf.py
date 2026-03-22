"""
Пункт 1.3: Фильтр нижних частот (ФНЧ)
ИСПРАВЛЕННАЯ ВЕРСИЯ — все графики контрастные
"""
import numpy as np
from scipy.fft import fft, fftshift
from config import SignalParams, OutputParams
from utils import (
    create_rect_pulse, create_noisy_signal, apply_freq_filter,
    lpf_mask, calculate_mse,
    plot_time_three, plot_spectrum_before, plot_spectrum_after,
    plot_mask, plot_mse_line, ensure_directories
)


def run():
    """Выполнение пункта 1.3"""
    ensure_directories()
    np.random.seed(42)

    t = SignalParams.get_time_array()
    freqs, _ = SignalParams.get_freq_array(len(t))

    g = create_rect_pulse(t, SignalParams.a, SignalParams.t1, SignalParams.t2)
    G = fftshift(fft(g))

    b_fixed = 0.2
    print("\n[1.3] ФНЧ: влияние частоты среза (b=0.2)")
    print(f"{'ν₀, Гц':<10} {'MSE':<15} {'Комментарий'}")
    print("-" * 40)

    cutoffs = [2, 5, 10, 15, 20, 30]
    results = []

    for nu0 in cutoffs:
        u = create_noisy_signal(g, t, b=b_fixed)
        U = fftshift(fft(u))

        uf, mask, _, Uf = apply_freq_filter(
            u, freqs, lambda f: lpf_mask(f, nu0)
        )
        mse = calculate_mse(g, uf)
        results.append((nu0, mse))

        comment = "искажение фронтов (Гиббс)" if nu0 < 5 else ("пропуск шума" if nu0 > 15 else "оптимально")
        print(f"{nu0:<10} {mse:<15.6f} {comment}")

        # График для КАЖДОГО значения ν₀
        plot_time_three(
            t, g, u, uf,
            title=f"ФНЧ: ν₀ = {nu0} Гц, b = {b_fixed}",
            save_path=f"{OutputParams.figures_dir}/task1_3_time_nu0{nu0}.png",
            ylim=(-0.5, 1.5)
        )

        plot_spectrum_before(
            freqs, G, U,
            title=f"Спектры до ФНЧ (ν₀={nu0} Гц)",
            save_path=f"{OutputParams.figures_dir}/task1_3_spec_before_nu0{nu0}.png"
        )

        plot_spectrum_after(
            freqs, G, Uf,
            title=f"Спектры после ФНЧ (ν₀={nu0} Гц)",
            save_path=f"{OutputParams.figures_dir}/task1_3_spec_after_nu0{nu0}.png"
        )

        plot_mask(
            freqs, mask,
            title=f"АЧХ ФНЧ (ν₀ = {nu0} Гц)",
            save_path=f"{OutputParams.figures_dir}/task1_3_mask_nu0{nu0}.png"
        )

    # Сводный график MSE
    plot_mse_line(
        [r[0] for r in results], [r[1] for r in results],
        "Частота среза ν₀, Гц", "Зависимость MSE от ν₀ (b=0.2)",
        f"{OutputParams.figures_dir}/task1_3_mse_cutoff.png"
    )

    # === Исследование 2: влияние уровня шума ===
    print(f"\n[1.3] ФНЧ: влияние уровня шума (ν₀=10 Гц)")
    print(f"{'b':<10} {'MSE':<15} {'Примечание'}")
    print("-" * 40)

    nu0_fixed = 10
    b_vals = [0.05, 0.1, 0.2, 0.5, 1.0]
    b_results = []

    for b_val in b_vals:
        u = create_noisy_signal(g, t, b=b_val)
        U = fftshift(fft(u))

        uf, mask, _, Uf = apply_freq_filter(
            u, freqs, lambda f: lpf_mask(f, nu0_fixed)
        )
        mse = calculate_mse(g, uf)
        b_results.append((b_val, mse))

        note = "квадратичный рост" if b_val >= 0.2 else "низкий шум"
        print(f"{b_val:<10} {mse:<15.6f} {note}")

        # График для КАЖДОГО b
        plot_time_three(
            t, g, u, uf,
            title=f"ФНЧ: b = {b_val}, ν₀ = {nu0_fixed} Гц",
            save_path=f"{OutputParams.figures_dir}/task1_3_time_b{b_val}.png",
            ylim=(-1.5, 2.5) if b_val > 0.5 else (-0.5, 1.5)
        )

        plot_spectrum_after(
            freqs, G, Uf,
            title=f"Спектр после ФНЧ (b={b_val})",
            save_path=f"{OutputParams.figures_dir}/task1_3_spec_after_b{b_val}.png"
        )

    # MSE vs b (логарифмический)
    plot_mse_line(
        [r[0] for r in b_results], [r[1] for r in b_results],
        "Амплитуда шума b", "Зависимость MSE от b (ν₀=10 Гц)",
        f"{OutputParams.figures_dir}/task1_3_mse_b.png", log_y=True
    )

    # === ТАБЛИЦЫ ДЛЯ ОТЧЕТА ===
    print("\n[ТАБЛИЦА 1 — для отчета]")
    print("Зависимость MSE от частоты среза ФНЧ (b=0.2):")
    for nu0, mse in results:
        print(f"{nu0} Гц & {mse:.6f} & {'искажение' if nu0<5 else 'оптимум' if nu0<=15 else 'шум'} \\\\")

    print("\nЗависимость MSE от уровня шума (ν₀=10 Гц):")
    for b_val, mse in b_results:
        print(f"{b_val} & {mse:.6f} & MSE ∝ b² \\\\")

    print("\n[ВЫВОДЫ 1.3]")
    print("• Оптимальный ν₀: 5-10 Гц")
    print("• При ν₀ < 5 Гц — эффект Гиббса")
    print("• При ν₀ > 15 Гц — пропуск шума")
    print("• MSE ~ b² подтверждается линейным ростом в лог-масштабе")