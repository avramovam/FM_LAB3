"""
Главный файл для запуска всех заданий
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')

from utils import ensure_directories
import task_1_3_lpf
import task_1_5_notch
import task_1_7_combined
import task_1_9_hpf
import task_2_audio


def print_header(text):
    """Вывод заголовка"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)


def main():
    """Основная функция"""
    print_header("ЛАБОРАТОРНАЯ РАБОТА №3: ЖЕСТКАЯ ФИЛЬТРАЦИЯ")

    # Создание необходимых директорий
    ensure_directories()

    # Установка seed для воспроизводимости
    np.random.seed(42)

    # Выполнение заданий
    tasks = [
        (task_1_3_lpf.run, "Пункт 1.3: Фильтр нижних частот (ФНЧ)"),
        (task_1_5_notch.run, "Пункт 1.5: Режекторный фильтр"),
        (task_1_7_combined.run, "Пункт 1.7: Комбинированная фильтрация"),
        (task_1_9_hpf.run, "Пункт 1.9: Фильтр верхних частот (ФВЧ)"),
        (task_2_audio.run, "Пункт 2: Фильтрация звука")
    ]

    for task_func, task_name in tasks:
        print_header(task_name)
        try:
            task_func()
        except Exception as e:
            print(f"Ошибка при выполнении {task_name}: {e}")
            import traceback
            traceback.print_exc()

    print_header("РАБОТА ЗАВЕРШЕНА")
    print("Все результаты сохранены в папке 'results'")


if __name__ == "__main__":
    main()