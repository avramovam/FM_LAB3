"""
Пункт 2: Фильтрация звука
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftshift, fftfreq
import os

from config import AudioParams, OutputParams
from utils import (
    apply_freq_filter, create_bandpass_mask, ensure_directories,
    plot_time_domain, plot_spectrum_magnitude, plot_filtered_spectrum,
    plot_filter_mask
)


def load_audio(filename):
    """Загрузка аудиофайла"""
    try:
        fs, audio = wavfile.read(filename)

        if len(audio.shape) > 1:
            audio = audio[:, 0]

        audio = audio.astype(np.float32) / np.max(np.abs(audio))

        dt = 1 / fs
        t = np.arange(len(audio)) * dt

        print(f"\nФайл {filename} загружен")
        print(f"  Частота: {fs} Гц")
        print(f"  Длительность: {len(audio)/fs:.2f} с")
        print(f"  Отсчетов: {len(audio)}")

        return audio, fs, t

    except FileNotFoundError:
        print(f"\nФайл {filename} не найден. Создаю тестовый сигнал...")
        fs = 44100
        t = np.arange(0, 5, 1/fs)
        voice = 0.5 * np.sin(2*np.pi*500*t) * np.exp(-t) + \
                0.3 * np.sin(2*np.pi*1000*t) * np.exp(-t/2)
        noise_low = 0.2 * np.sin(2*np.pi*50*t)
        noise_high = 0.1 * (2*np.random.random(len(t)) - 1)
        audio = voice + noise_low + noise_high
        audio = audio / np.max(np.abs(audio))
        return audio, fs, t


def analyze_audio(audio, fs, t):
    """Анализ аудиосигнала"""
    freqs = fftshift(fftfreq(len(audio), 1/fs))
    AUDIO = fftshift(fft(audio))

    # Полный сигнал
    plt.figure(figsize=(16, 8))
    plt.plot(t, audio, 'b-', linewidth=0.8)
    plt.xlabel('Время (с)', fontsize=16)
    plt.ylabel('Амплитуда', fontsize=16)
    plt.title('Исходный аудиосигнал', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_audio_full.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # Фрагмент
    plt.figure(figsize=(16, 8))
    plt.plot(t[:5000], audio[:5000], 'b-', linewidth=1.2)
    plt.xlabel('Время (с)', fontsize=16)
    plt.ylabel('Амплитуда', fontsize=16)
    plt.title('Фрагмент сигнала (первые 5000 отсчетов)', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_audio_fragment.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # Спектр
    plt.figure(figsize=(16, 8))
    plt.plot(freqs[freqs >= 0],
             20*np.log10(np.abs(AUDIO[freqs >= 0]) + 1e-10), 'r-', linewidth=1.2)
    plt.axvspan(AudioParams.voice_low, AudioParams.voice_high,
                alpha=0.2, color='green', label='Диапазон голоса')
    plt.xlabel('Частота (Гц)', fontsize=16)
    plt.ylabel('Спектр (дБ)', fontsize=16)
    plt.title('Спектр сигнала', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, min(5000, fs/2)])
    plt.ylim([-60, 20])
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_spectrum_log.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    return freqs, AUDIO


def run():
    """Выполнение пункта 2"""

    ensure_directories()

    audio, fs, t = load_audio('MUHA.wav')
    freqs, AUDIO = analyze_audio(audio, fs, t)

    print("\nПрименение полосовых фильтров")
    print("-" * 40)

    for low, high, name in AudioParams.voice_ranges:
        print(f"\n{name}: {low}-{high} Гц")

        def bandpass_mask(f):
            mask = np.zeros_like(f, dtype=bool)
            mask[(f >= low) & (f <= high)] = True
            mask[(f >= -high) & (f <= -low)] = True
            return mask

        audio_filtered, mask, _, AUDIO_filtered = apply_freq_filter(
            audio, freqs, bandpass_mask
        )

        # Временная область
        plt.figure(figsize=(16, 8))
        plt.plot(t[:10000], audio[:10000], 'b-', alpha=0.7, linewidth=1.2, label='Исходный')
        plt.plot(t[:10000], audio_filtered[:10000], 'g-', linewidth=1.5, label=f'Фильтрованный')
        plt.xlabel('Время (с)', fontsize=16)
        plt.ylabel('Амплитуда', fontsize=16)
        plt.title(f'Результат фильтрации: {name}', fontsize=18, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_{name.replace(' ', '_')}_time.png",
                   dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # Спектры
        plt.figure(figsize=(16, 8))
        plt.plot(freqs[freqs >= 0],
                20*np.log10(np.abs(AUDIO[freqs >= 0]) + 1e-10),
                'r-', alpha=0.7, linewidth=1.2, label='Исходный')
        plt.plot(freqs[freqs >= 0],
                20*np.log10(np.abs(AUDIO_filtered[freqs >= 0]) + 1e-10),
                'b-', linewidth=1.5, label='Фильтрованный')
        plt.axvspan(low, high, alpha=0.2, color='green', label='Полоса')
        plt.xlabel('Частота (Гц)', fontsize=16)
        plt.ylabel('Спектр (дБ)', fontsize=16)
        plt.title(f'Спектры: {name}', fontsize=18, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, min(5000, fs/2)])
        plt.ylim([-60, 20])
        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_{name.replace(' ', '_')}_spectrum.png",
                   dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # Сохранение аудио
        audio_filtered_int = np.int16(audio_filtered * 32767)
        wavfile.write(f"{OutputParams.audio_dir}/MUHA_filtered_{low}_{high}.wav",
                     fs, audio_filtered_int)
        print(f"  Сохранено: MUHA_filtered_{low}_{high}.wav")

    # Маска оптимального фильтра
    mask_opt = create_bandpass_mask(freqs, 300, 3400)
    plot_filter_mask(
        freqs, mask_opt,
        title="АЧХ полосового фильтра (300-3400 Гц)",
        save_path=f"{OutputParams.figures_dir}/task2_optimal_mask.png"
    )

    print("\n" + "="*50)
    print("ВЫВОДЫ ПО ПУНКТУ 2")
    print("="*50)
    print("""
    1. Оптимальный диапазон: 300-3400 Гц
    2. Подавляет низкочастотные и высокочастотные шумы
    3. Преимущества: прямоугольная АЧХ, нет фазовых искажений
    4. Недостатки: эффект Гиббса, не для онлайн-обработки
    """)