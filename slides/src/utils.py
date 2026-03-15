#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from PIL import Image

def extract_frames(gif_path, output_dir=None, quality=95):
    """
    Извлекает все кадры из GIF и сохраняет их как JPEG.
    
    Параметры:
        gif_path (str): путь к GIF-файлу
        output_dir (str): директория для сохранения кадров (если None, создаётся рядом с GIF)
        quality (int): качество JPEG (1-100)
    """
    # Проверяем существование файла
    if not os.path.isfile(gif_path):
        print(f"Ошибка: файл '{gif_path}' не найден.")
        return

    # Открываем GIF
    try:
        gif = Image.open(gif_path)
    except Exception as e:
        print(f"Не удалось открыть GIF: {e}")
        return

    # Определяем выходную директорию
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(gif_path))[0]
        output_dir = os.path.join(os.path.dirname(gif_path), base_name + "_frames")
    
    # Создаём директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    try:
        while True:
            # Перемещаемся к текущему кадру
            gif.seek(frame_count)
            frame = gif.convert('RGB')  # Конвертируем в RGB (JPEG не поддерживает прозрачность)
            
            # Формируем имя файла с ведущими нулями (4 цифры)
            frame_filename = f"frame_{frame_count+1:04d}.jpeg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Сохраняем кадр как JPEG
            frame.save(frame_path, 'JPEG', quality=quality)
            print(f"Сохранён кадр {frame_count+1}: {frame_path}")
            
            frame_count += 1
    except EOFError:
        # Достигнут конец GIF
        pass

    print(f"Готово. Всего кадров: {frame_count}. Сохранены в: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Извлечение кадров из GIF в JPEG")
    parser.add_argument("gif", help="Путь к GIF-файлу")
    parser.add_argument("-o", "--output", help="Выходная директория (по умолчанию создаётся рядом с GIF)")
    parser.add_argument("-q", "--quality", type=int, default=95, help="Качество JPEG (1-100, по умолчанию 95)")
    args = parser.parse_args()

    extract_frames(args.gif, args.output, args.quality)