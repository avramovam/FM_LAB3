"""
Конфигурационный файл с параметрами для всех заданий
"""

import numpy as np

# Параметры для моделирования сигнала
class SignalParams:
    # Параметры прямоугольного импульса
    a = 1.0          # Амплитуда
    t1 = -1.0        # Начало импульса
    t2 = 1.0         # Конец импульса

    # Параметры шумов (базовые значения)
    b = 0.2          # Амплитуда белого шума
    c = 0.5          # Амплитуда синусоидальной помехи
    d = 20.0         # Частота синусоидальной помехи (Гц)

    # Параметры дискретизации
    T_total = 10.0   # Общее время наблюдения (с)
    dt = 0.01        # Шаг дискретизации (с)

    @classmethod
    def get_time_array(cls):
        """Создание временного массива"""
        return np.arange(-cls.T_total/2, cls.T_total/2, cls.dt)

    @classmethod
    def get_freq_array(cls, n_points):
        """Создание частотного массива для FFT"""
        from scipy.fft import fftfreq, fftshift
        freqs = fftshift(fftfreq(n_points, cls.dt))
        return freqs, 1/cls.dt


# Параметры для аудиофильтрации
class AudioParams:
    # Диапазон голоса (Гц)
    voice_low = 300
    voice_high = 3400

    # Альтернативные диапазоны для экспериментов
    voice_ranges = [
        (300, 3400, "Стандартный голос"),
        (200, 3000, "Широкий диапазон"),
        (500, 3000, "Узкий диапазон"),
        (100, 1000, "Только низкие частоты")
    ]


# Параметры для сохранения результатов
class OutputParams:
    save_figures = True
    figures_dpi = 300
    figures_format = 'png'
    results_dir = 'results'
    figures_dir = 'results/figures'
    audio_dir = 'results/audio'


# Параметры отображения
class DisplayParams:
    SHOW_PLOTS = False  # True - показывать графики, False - только сохранять