import json

with open(r'c:\Users\vanhi\Desktop\HCMUTE_TMDT\HKII_Nam_3\Bao_Mat_TMDT\Fake_reviews\notebooks\01_EDA_Preprocessing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(r'c:\Users\vanhi\Desktop\HCMUTE_TMDT\HKII_Nam_3\Bao_Mat_TMDT\Fake_reviews\extract_nb_utf8.txt', 'w', encoding='utf-8') as fout:
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = ''.join(cell.get('source', []))
        fout.write(f"Cell {i+1} [{cell_type}]:\n")
        fout.write(source + "\n")
        fout.write("-" * 40 + "\n")
