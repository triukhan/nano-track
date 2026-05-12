#!/bin/bash

TRAIN_DIR="/home/danylo/GIT/nano-track/data/train"

echo "Починаємо перейменування файлів у $TRAIN_DIR"

for video_dir in "$TRAIN_DIR"/*/ ; do
    if [ ! -d "$video_dir" ]; then
        continue
    fi

    dir_name=$(basename "$video_dir")
    echo "Обробляємо: $dir_name"

    cd "$video_dir" || continue

    # Знаходимо всі jpg файли і сортуємо їх за назвою
    mapfile -t files < <(ls -v *.jpg 2>/dev/null)

    if [ ${#files[@]} -eq 0 ]; then
        echo "  Пропускаємо (немає jpg файлів)"
        continue
    fi

    echo "  Знайдено ${#files[@]} файлів — перейменовуємо..."

    counter=0
    for file in "${files[@]}"; do
        # Формуємо нову назву з ведучими нулями (frame_00000.jpg)
        new_name=$(printf "frame_%05d.jpg" "$counter")

        if [ "$file" != "$new_name" ]; then
            mv "$file" "$new_name"
            echo "    $file → $new_name"
        fi

        ((counter++))
    done

done

echo "Готово! Усі папки оброблено."