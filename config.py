"""
Конфигурационный файл с параметрами для всех заданий
"""
import numpy as np


# Параметры моделирования сигнала
class SignalParams:
    # Прямоугольный импульс
    a = 1.0          # амплитуда
    t1 = -1.0        # начало импульса
    t2 = 1.0         # конец импульса

    # Параметры шумов
    b = 0.2          # амплитуда белого шума
    c = 0.5          # амплитуда синусоидальной помехи
    d = 20.0         # частота синусоидальной помехи, Гц

    # Дискретизация
    T_total = 10.0   # время наблюдения, с
    dt = 0.01        # шаг дискретизации, с

    @classmethod
    def get_time_array(cls):
        """Формирует массив времени от -T_total/2 до T_total/2"""
        return np.arange(-cls.T_total/2, cls.T_total/2, cls.dt)

    @classmethod
    def get_freq_array(cls, n_points):
        """Формирует массив частот для БПФ"""
        from scipy.fft import fftfreq, fftshift
        freqs = fftshift(fftfreq(n_points, cls.dt))
        return freqs, 1/cls.dt


# Параметры для обработки аудио
class AudioParams:
    # Диапазон голосового сигнала, Гц
    voice_low = 300
    voice_high = 3400

    # Варианты фильтрации для экспериментов
    voice_ranges = [
        (300, 3400, "Стандартный голос"),
        (200, 3000, "Широкий диапазон"),
        (500, 3000, "Узкий диапазон"),
        (100, 1000, "Только низкие частоты")
    ]


# Параметры сохранения результатов
class OutputParams:
    save_figures = True
    figures_dpi = 300
    figures_format = 'png'
    results_dir = 'results'
    figures_dir = 'results/figures'
    audio_dir = 'results/audio'


# Параметры отображения
class DisplayParams:
    SHOW_PLOTS = False   # True — показывать графики, False — только сохранять