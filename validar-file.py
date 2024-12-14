import json

input_file = "/Users/daniel.gomes/Documents/Projetos/Python/data/trn.json"
output_file = "/Users/daniel.gomes/Documents/Projetos/Python/data/trn01.json"

# Reestruturar os objetos JSON em uma lista
valid_data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)  # Tentar carregar cada linha como JSON
            valid_data.append({"title": obj["title"], "content": obj["content"]})
        except json.JSONDecodeError as e:
            print(f"Linha inválida ignorada: {line.strip()} - {e}")

# Salvar o JSON reestruturado
with open(output_file, "w", encoding="utf-8") as f:
  f.write("[")
  for obj in valid_data:
        f.write(json.dumps(obj, separators=(',', ':'), ensure_ascii=False) + ",\n")
  f.write("]")
  
print("Arquivo JSON validado salvo com sucesso.")
