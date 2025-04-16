import random
import json

input_path = 'FashionGen/caption.jsonl'
output_path = 'FashionGen/test_FashionGen.jsonl'
num_samples = 32528

# Đọc tất cả các dòng từ file
with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lấy ngẫu nhiên 10k dòng
random_lines = random.sample(lines, min(num_samples, len(lines)))

# Ghi các dòng đã chọn ra file mới
with open(output_path, 'w', encoding='utf-8') as f:
    f.writelines(random_lines)

print(f"Đã ghi {len(random_lines)} dòng vào {output_path}")
