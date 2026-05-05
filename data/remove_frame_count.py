from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

input_file = PROJECT_ROOT / 'data' / 'frames' / 'groundtruth.txt'
temp_file = input_file.with_name("groundtruth_cleaned.txt")

pattern = re.compile(r'^\d')

with open(input_file, "r") as f_in, open(temp_file, "w") as f_out:
    for line in f_in:
        line = line.strip()

        if not line:
            continue

        if not pattern.match(line):
            continue

        cleaned = line.rsplit(" ", 1)[0]
        f_out.write(cleaned + "\n")

# замінюємо старий файл новим
temp_file.replace(input_file)