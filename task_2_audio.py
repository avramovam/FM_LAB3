"""
Пункт 2: Фильтрация звука — исправленная версия
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

    # === Загрузка или генерация тестового сигнала ===
    try:
        fs, audio = wavfile.read('MUHA.wav')
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Берём один канал
        audio = audio.astype(np.float32) / np.max(np.abs(audio))
    except FileNotFoundError:
        # Генерация тестового сигнала если файла нет
        fs = 44100
        t = np.arange(0, 5, 1/fs)
        voice = (0.5 * np.sin(2*np.pi*500*t) * np.exp(-t) +
                 0.3 * np.sin(2*np.pi*1000*t) * np.exp(-t/2))
        noise_low = 0.2 * np.sin(2*np.pi*50*t)  # НЧ-гул
        noise_high = 0.1 * (2*np.random.random(len(t)) - 1)  # ВЧ-шум
        audio = voice + noise_low + noise_high
        audio = audio / np.max(np.abs(audio))

    t = np.arange(len(audio)) / fs
    N = len(audio)

    # === Частотная сетка для аудио (ПРАВИЛЬНАЯ) ===
    freqs = fftshift(fftfreq(N, 1/fs))  # Частоты в Гц для реального fs
    AUDIO = fftshift(fft(audio))

    # === 1. Исходный сигнал (фрагмент) ===
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_len = min(5000, len(audio))
    ax.plot(t[:plot_len], audio[:plot_len], 'b-', linewidth=0.8)
    ax.set_xlabel('Время, с', fontsize=14)
    ax.set_ylabel('Амплитуда', fontsize=14)
    ax.set_title('Исходный аудиосигнал (фрагмент 0.1 с)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_audio_orig.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # === 2. Спектр исходного (логарифмический) ===
    fig, ax = plt.subplots(figsize=(16, 8))
    # Только положительная часть спектра
    pos_mask = freqs >= 0
    ax.plot(freqs[pos_mask], 20*np.log10(np.abs(AUDIO[pos_mask]) + 1e-10), 'r-', linewidth=1)
    ax.axvspan(300, 3400, alpha=0.2, color='green', label='Диапазон голоса 300-3400 Гц')
    ax.set_xlabel('Частота, Гц', fontsize=14)
    ax.set_ylabel('Спектр, дБ', fontsize=14)
    ax.set_title('Спектр исходного аудиосигнала', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5000])  # ← ВАЖНО: правильный диапазон для аудио!
    ax.set_ylim([-80, 20])
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_spectrum_orig.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    # === 3. Фильтрация для всех диапазонов ===
    print("\n[2] Фильтрация звука: сравнение диапазонов")

    for low, high, name in AudioParams.voice_ranges:
        print(f"  {name}: {low}-{high} Гц")

        # Применяем полосовой фильтр
        audio_filt, mask, _, AUDIO_filt = apply_freq_filter(
            audio, freqs, lambda f: bandpass_mask(f, low, high))

        # 3.1 Временная область
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(t[:plot_len], audio[:plot_len], 'b-', alpha=0.5, linewidth=0.8, label='Исходный')
        ax.plot(t[:plot_len], audio_filt[:plot_len], 'g-', linewidth=1.2, label='Отфильтрованный')
        ax.set_xlabel('Время, с', fontsize=14)
        ax.set_ylabel('Амплитуда', fontsize=14)
        ax.set_title(f'Результат фильтрации: {name} ({low}-{high} Гц)',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_time_{low}_{high}.png",
                    dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # 3.2 Спектры до и после
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(freqs[pos_mask], 20*np.log10(np.abs(AUDIO[pos_mask]) + 1e-10),
               'r-', alpha=0.6, linewidth=1, label='Исходный')
        ax.plot(freqs[pos_mask], 20*np.log10(np.abs(AUDIO_filt[pos_mask]) + 1e-10),
               'g-', linewidth=1.2, label='Отфильтрованный')
        ax.axvspan(low, high, alpha=0.2, color='green', label=f'Полоса {low}-{high} Гц')
        ax.set_xlabel('Частота, Гц', fontsize=14)
        ax.set_ylabel('Спектр, дБ', fontsize=14)
        ax.set_title(f'Спектры: {name}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 5000])  # ← ПРАВИЛЬНЫЙ ДИАПАЗОН ДЛЯ АУДИО!
        ax.set_ylim([-80, 20])
        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_spec_{low}_{high}.png",
                    dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # 3.3 АЧХ фильтра — ИСПРАВЛЕННАЯ ВЕРСИЯ
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(freqs, mask.astype(float), color='#000000', linewidth=2, label='Коэффициент передачи')
        ax.fill_between(freqs, 0, mask.astype(float), alpha=0.2, color='gray')
        ax.set_xlabel('Частота, Гц', fontsize=14)
        ax.set_ylabel('Коэффициент передачи', fontsize=14)
        ax.set_title(f'АЧХ полосового фильтра ({low}-{high} Гц)',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # ← КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: правильный диапазон частот для аудио
        ax.set_xlim([0, 5000])  # Показываем 0-5000 Гц вместо -30-30 Гц
        ax.set_ylim([-0.1, 1.1])
        ax.set_yticks([0, 1])
        plt.tight_layout()
        plt.savefig(f"{OutputParams.figures_dir}/task2_mask_{low}_{high}.png",
                    dpi=OutputParams.figures_dpi, bbox_inches='tight')
        plt.close()

        # Сохранение аудио
        audio_int = np.int16(audio_filt * 32767)
        wavfile.write(f"{OutputParams.audio_dir}/MUHA_{low}_{high}.wav", fs, audio_int)

    # === 4. Оптимальная маска (отдельный файл) ===
    fig, ax = plt.subplots(figsize=(16, 6))
    optimal_mask = bandpass_mask(freqs, 300, 3400)
    ax.plot(freqs, optimal_mask.astype(float), color='#000000', linewidth=2)
    ax.fill_between(freqs, 0, optimal_mask.astype(float), alpha=0.2, color='green')
    ax.set_xlabel('Частота, Гц', fontsize=14)
    ax.set_ylabel('Коэффициент передачи', fontsize=14)
    ax.set_title('АЧХ оптимального полосового фильтра (300-3400 Гц)',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5000])  # ← ПРАВИЛЬНЫЙ ДИАПАЗОН!
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 1])
    plt.tight_layout()
    plt.savefig(f"{OutputParams.figures_dir}/task2_optimal_mask.png",
                dpi=OutputParams.figures_dpi, bbox_inches='tight')
    plt.close()

    print("\n[Анализ] Оптимальный диапазон 300-3400 Гц соответствует стандарту телефонной связи.")
    print("Подавлены НЧ-гул (<300 Гц) и ВЧ-шипение (>3400 Гц), голос сохранён.")