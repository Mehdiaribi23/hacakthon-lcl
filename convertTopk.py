import json

# Chemin vers le fichier notebook
notebook_path = 'Identifier_un_client.ipynb'
output_script = 'client.py'

# Ouvrir le fichier notebook
with open(notebook_path, 'r') as f:
    notebook_data = json.load(f)

# Extraire les cellules de type "code"
code_cells = [cell for cell in notebook_data['cells'] if cell['cell_type'] == 'code']

# Écrire les cellules de code dans un fichier .py
with open(output_script, 'w') as f:
    for cell in code_cells:
        f.write(''.join(cell['source']) + '\n\n')

print(f"Le code a été extrait dans {output_script}")
