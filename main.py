"""
Главный файл для запуска всех заданий
"""
import numpy as np
from config import OutputParams
from utils import ensure_directories

# Список заданий для выполнения
TASKS = [
    ("1.3 ФНЧ", "task_1_3_lpf"),
    ("1.5 Режектор", "task_1_5_notch"),
    ("1.7 Комбинированный", "task_1_7_combined"),
    ("1.9 ФВЧ", "task_1_9_hpf"),
    ("2 Аудио", "task_2_audio"),
]

def main():
    """Запускает все задания лабораторной работы"""
    np.random.seed(42)          # фиксируем генератор для воспроизводимости
    ensure_directories()        # создаем папки для результатов

    print("=== ЛР3: Жесткая фильтрация ===")

    for name, module_name in TASKS:
        print(f"\n>>> Запуск: {name}")
        module = __import__(module_name, fromlist=['run'])
        module.run()

    print(f"\nГотово. Графики сохранены в: {OutputParams.figures_dir}/")

if __name__ == "__main__":
    main()