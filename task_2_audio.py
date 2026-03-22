"""Пункт 2: Фильтрация звука — все диапазоны для отчета"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftshift, fftfreq
from config import AudioParams, OutputParams
from utils import apply_freq_filter, bandpass_mask, plot_mask
from utils import ensure_directories as ensure_dirs

def run():
    ensure_dirs()
    np.random.seed(42)

    # Загрузка или генерация
    try:
        fs, audio = wavfile.read('MUHA.wav')
        if len(audio.shape) > 1: audio = audio[:, 0]
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
    except:
        fs = 44100
        t = np.arange(0, 5, 1/fs)
        voice = 0.5*np.sin(2*np.pi*500*t)*np.exp(-t) + 0.3*np.sin(2*np.pi*1000*t)*np.exp(-t/2)
        audio = voice + 0.2*np.sin(2*np.pi*50*t) + 0.1*(2*np.random.random(len(t))-1)
        audio = audio / np.max(np.abs(audio))

    t = np.arange(len(audio)) / fs
    freqs = fftshift(fftfreq(len(audio), 1/fs))
    AUDIO = fftshift(fft(audio))

    # === Исходный сигнал ===
    fig, ax = plt.subplots()
    ax.plot(t[:5000], audio[:5000], 'b-', linewidth=0.8)
    ax.set_xlabel('Время, с', fontsize=13)
    ax.set_ylabel('Амплитуда', fontsize=13)
    ax.set_title('Исходный аудиосигнал (фрагмент 0.1 с)', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_audio_orig.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # === Спектр исходного ===
    fig, ax = plt.subplots()
    ax.plot(freqs[freqs >= 0], 20*np.log10(np.abs(AUDIO[freqs >= 0]) + 1e-10), 'r-', linewidth=1)
    ax.axvspan(300, 3400, alpha=0.2, color='green', label='Диапазон голоса')
    ax.set_xlabel('Частота, Гц', fontsize=13)
    ax.set_ylabel('Спектр, дБ', fontsize=13)
    ax.set_title('Спектр исходного сигнала', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5000])
    ax.set_ylim([-60, 20])
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_spectrum_orig.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # === Фильтрация для ВСЕХ диапазонов из config ===
    print("\n[2] Фильтрация звука: сравнение диапазонов")
    for low, high, name in AudioParams.voice_ranges:
        print(f"  {name}: {low}-{high} Гц")

        audio_filt, mask, _, AUDIO_filt = apply_freq_filter(
            audio, freqs, lambda f: bandpass_mask(f, low, high))

        # Временная область
        fig, ax = plt.subplots()
        ax.plot(t[:5000], audio[:5000], 'b-', alpha=0.5, linewidth=0.8, label='Исходный')
        ax.plot(t[:5000], audio_filt[:5000], 'g-', linewidth=1.2, label='Отфильтрованный')
        ax.set_xlabel('Время, с', fontsize=13)
        ax.set_ylabel('Амплитуда', fontsize=13)
        ax.set_title(f'Результат: {name} ({low}-{high} Гц)', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_time_{low}_{high}.png",
                    dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # Спектры
        fig, ax = plt.subplots()
        ax.plot(freqs[freqs >= 0], 20*np.log10(np.abs(AUDIO[freqs >= 0]) + 1e-10),
               'r-', alpha=0.6, linewidth=1, label='Исходный')
        ax.plot(freqs[freqs >= 0], 20*np.log10(np.abs(AUDIO_filt[freqs >= 0]) + 1e-10),
               'g-', linewidth=1.2, label='Отфильтрованный')
        ax.axvspan(low, high, alpha=0.2, color='green', label='Полоса')
        ax.set_xlabel('Частота, Гц', fontsize=13)
        ax.set_ylabel('Спектр, дБ', fontsize=13)
        ax.set_title(f'Спектры: {name}', fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 5000])
        ax.set_ylim([-60, 20])
        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_spec_{low}_{high}.png",
                    dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # АЧХ фильтра
        plot_mask(freqs, mask, f"АЧХ полосового фильтра ({low}-{high} Гц)",
                 f"{OutputParams.figures_dir}/task2_mask_{low}_{high}.png")

        # Сохранение аудио
        audio_int = np.int16(audio_filt * 32767)
        wavfile.write(f"{OutputParams.audio_dir}/MUHA_{low}_{high}.wav", fs, audio_int)

    print("\n[Анализ] Оптимальный диапазон 300-3400 Гц соответствует стандарту телефонной связи.")
    print("Подавлены НЧ-гул (<300 Гц) и ВЧ-шипение (>3400 Гц), голос сохранён.")