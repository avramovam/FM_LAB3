"""
Пункт 2: Фильтрация звука
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftshift, fftfreq
from config import AudioParams, OutputParams
from utils import apply_freq_filter, bandpass_mask, ensure_directories


def run():
    """Выполнение пункта 2"""
    ensure_directories()
    np.random.seed(42)

    # настройки шрифтов для печати
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
        'figure.figsize': (18, 10),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })

    # загрузка аудиофайла
    try:
        fs, audio = wavfile.read('MUHA.wav')
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
        print(f"Файл MUHA.wav загружен (fs = {fs} Гц, длительность = {len(audio)/fs:.2f} с)")
    except FileNotFoundError:
        print("Файл MUHA.wav не найден, генерирую тестовый сигнал...")
        fs = 44100
        t = np.arange(0, 5, 1/fs)
        voice = (0.5 * np.sin(2*np.pi*500*t) * np.exp(-t) +
                 0.3 * np.sin(2*np.pi*1000*t) * np.exp(-t/2))
        noise_low = 0.2 * np.sin(2*np.pi*50*t)
        noise_high = 0.1 * (2*np.random.random(len(t)) - 1)
        audio = voice + noise_low + noise_high
        audio = audio / np.max(np.abs(audio))

    t = np.arange(len(audio)) / fs
    N = len(audio)

    freqs = fftshift(fftfreq(N, 1/fs))
    AUDIO = fftshift(fft(audio))

    total_energy = np.mean(audio**2)

    # исходный сигнал
    print("\n[2.1] Исходный сигнал")
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_len = min(5000, len(audio))
    ax.plot(t[:plot_len], audio[:plot_len], 'b-', linewidth=2.5)
    ax.set_xlabel('Время, с', fontsize=20, fontweight='bold')
    ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
    ax.set_title('Исходный аудиосигнал (фрагмент 0.1 с)', fontsize=22, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
    ax.tick_params(labelsize=18, width=1.2, length=6)
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_audio_orig.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # спектр исходного сигнала
    print("[2.2] Спектр исходного сигнала")
    fig, ax = plt.subplots(figsize=(18, 10))
    pos_mask = freqs >= 0
    ax.plot(freqs[pos_mask], 20*np.log10(np.abs(AUDIO[pos_mask]) + 1e-10), 'r-', linewidth=2.5)
    ax.axvspan(300, 3400, alpha=0.2, color='green', label='Диапазон голоса 300-3400 Гц')
    ax.set_xlabel('Частота, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('Спектр, дБ', fontsize=20, fontweight='bold')
    ax.set_title('Спектр исходного аудиосигнала', fontsize=22, fontweight='bold', pad=15)
    ax.legend(fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
    ax.tick_params(labelsize=18, width=1.2, length=6)
    ax.set_xlim([0, 5000])
    ax.set_ylim([-80, 20])
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_spectrum_orig.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # фильтрация для всех диапазонов
    print("\n[2.3] Фильтрация звука: сравнение диапазонов")
    print("-" * 60)

    for low, high, name in AudioParams.voice_ranges:
        print(f"\n{name}: {low}-{high} Гц")

        audio_filt, mask, _, AUDIO_filt = apply_freq_filter(
            audio, freqs, lambda f: bandpass_mask(f, low, high))

        filt_energy = np.mean(audio_filt**2)
        energy_ratio = 100 * filt_energy / total_energy if total_energy > 0 else 0

        # временная область
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.plot(t[:plot_len], audio[:plot_len], 'b-', alpha=0.5, linewidth=2.0, label='Исходный')
        ax.plot(t[:plot_len], audio_filt[:plot_len], 'g-', linewidth=2.5, label='Отфильтрованный')
        ax.set_xlabel('Время, с', fontsize=20, fontweight='bold')
        ax.set_ylabel('Амплитуда', fontsize=20, fontweight='bold')
        ax.set_title(f'Результат фильтрации: {name} ({low}-{high} Гц)\nСохранено энергии: {energy_ratio:.2f}%',
                    fontsize=22, fontweight='bold', pad=15)
        ax.legend(fontsize=18, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
        ax.tick_params(labelsize=18, width=1.2, length=6)

        max_val = max(np.max(np.abs(audio[:plot_len])), np.max(np.abs(audio_filt[:plot_len])))
        if max_val > 0:
            ax.set_ylim([-max_val*1.2, max_val*1.2])

        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_time_{low}_{high}.png",
                    dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # спектры до и после фильтрации
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.plot(freqs[pos_mask], 20*np.log10(np.abs(AUDIO[pos_mask]) + 1e-10),
               'r-', alpha=0.6, linewidth=2.0, label='Исходный')
        ax.plot(freqs[pos_mask], 20*np.log10(np.abs(AUDIO_filt[pos_mask]) + 1e-10),
               'g-', linewidth=2.5, label='Отфильтрованный')
        ax.axvspan(low, high, alpha=0.2, color='green', label=f'Полоса {low}-{high} Гц')
        ax.set_xlabel('Частота, Гц', fontsize=20, fontweight='bold')
        ax.set_ylabel('Спектр, дБ', fontsize=20, fontweight='bold')
        ax.set_title(f'Спектры: {name}', fontsize=22, fontweight='bold', pad=15)
        ax.legend(fontsize=18, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
        ax.tick_params(labelsize=18, width=1.2, length=6)
        ax.set_xlim([0, 5000])
        ax.set_ylim([-80, 20])
        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_spec_{low}_{high}.png",
                    dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # АЧХ фильтра
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.plot(freqs, mask.astype(float), color='#000000', linewidth=3.0, label='Коэффициент передачи')
        ax.fill_between(freqs, 0, mask.astype(float), alpha=0.3, color='gray')
        ax.set_xlabel('Частота, Гц', fontsize=20, fontweight='bold')
        ax.set_ylabel('Коэффициент передачи', fontsize=20, fontweight='bold')
        ax.set_title(f'АЧХ полосового фильтра ({low}-{high} Гц)',
                    fontsize=22, fontweight='bold', pad=15)
        ax.legend(fontsize=18, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
        ax.tick_params(labelsize=18, width=1.2, length=6)
        ax.set_xlim([0, 5000])
        ax.set_ylim([-0.15, 1.15])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'], fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_mask_{low}_{high}.png",
                    dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # сохранение отфильтрованного аудио
        audio_int = np.int16(audio_filt * 32767)
        wavfile.write(f"{OutputParams.audio_dir}/MUHA_{low}_{high}.wav", fs, audio_int)

    # оптимальная АЧХ
    print("\n[2.4] Оптимальная АЧХ")
    fig, ax = plt.subplots(figsize=(18, 8))
    optimal_mask = bandpass_mask(freqs, 300, 3400)
    ax.plot(freqs, optimal_mask.astype(float), color='#000000', linewidth=3.0)
    ax.fill_between(freqs, 0, optimal_mask.astype(float), alpha=0.3, color='green')
    ax.set_xlabel('Частота, Гц', fontsize=20, fontweight='bold')
    ax.set_ylabel('Коэффициент передачи', fontsize=20, fontweight='bold')
    ax.set_title('АЧХ оптимального полосового фильтра (300-3400 Гц)',
                fontsize=22, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
    ax.tick_params(labelsize=18, width=1.2, length=6)
    ax.set_xlim([0, 5000])
    ax.set_ylim([-0.15, 1.15])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'], fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_optimal_mask.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    print(f"\nГрафики сохранены в: {OutputParams.figures_dir}/")
    print(f"Аудиофайлы сохранены в: {OutputParams.audio_dir}/")


if __name__ == "__main__":
    run()