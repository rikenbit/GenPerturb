import sys
import re

input_file  = "reference/jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
output_file = "reference/jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme_modified.txt"

with open(input_file, 'r') as file:
    text = file.read()


modified_text = re.sub(r"(MOTIF\sMA\d{4}\.\d)\s([A-Za-z0-9]+)", r"\1_\2", text)

with open(output_file, 'w') as file:
    file.write(modified_text)
