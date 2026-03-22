"""Главный файл: запуск только нужных задач"""
import numpy as np
from config import OutputParams
from utils import ensure_dirs

# Задачи для отчета (только они!)
TASKS = [
    ("1.3 ФНЧ", "task_1_3_lpf"),
    ("1.5 Режектор", "task_1_5_notch"),
    ("1.7 Комбинированный", "task_1_7_combined"),
    ("1.9 ФВЧ", "task_1_9_hpf"),
    ("2 Аудио", "task_2_audio"),
]

def main():
    np.random.seed(42)  # Воспроизводимость
    ensure_dirs()
    
    print("=== ЛР3: Жесткая фильтрация (минималистичная версия) ===")
    
    for name, module_name in TASKS:
        print(f"\n>>> Запуск: {name}")
        module = __import__(module_name, fromlist=['run'])
        module.run()
    
    print(f"\n✓ Готово. Графики сохранены в: {OutputParams.figures_dir}/")
    print("✓ Данные для таблиц выведены в консоль — скопируйте в отчет")

if __name__ == "__main__":
    main()