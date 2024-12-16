import json

input_file = "/Users/daniel.gomes/Documents/Projetos/Python/data/trn.json"
output_files = [f"/Users/daniel.gomes/Documents/Projetos/Python/data/parte{i+1}.json" for i in range(17)]  # Gerar nomes para 8 arquivos


# Ler o arquivo JSON validado
with open(input_file, "r", encoding="utf-8") as f:
    data = []
    for line in f:
        try:
            obj = json.loads(line)  # Tentar carregar cada linha como JSON
            data.append({"title": obj["title"], "content": obj["content"]})
        except json.JSONDecodeError as e:
            print(f"Linha inválida ignorada: {line.strip()} - {e}")

# Iterar sobre cada objeto e modificar
for i, obj in enumerate(data):
    # Exemplo de modificação: adicionar uma nova chave
    data[i] = {"title": obj["title"], "content": obj["content"]}   # Exemplo de novo valor baseado no índice

# Verificar se é uma lista
if isinstance(data, list):
    total_items = len(data)
    split_size = total_items // 17  # Dividir o total por 8 partes

    # Criar os chunks (partes)
    chunks = [data[i * split_size:(i + 1) * split_size] for i in range(17)]

    # Certificar que o restante seja incluído na última parte
    if len(data) % 17 != 0:
        chunks[-1].extend(data[17 * split_size:])

    # Salvar cada parte, um objeto por linha
    for i, chunk in enumerate(chunks):
        with open(output_files[i], "w", encoding="utf-8") as out_file:
            out_file.write("[")
            for obj in chunk:
                out_file.write(json.dumps(obj, ensure_ascii=False) + ",\n")
            out_file.write("]")
    
    print(f"Divisão concluída com sucesso. Gerados {len(output_files)} arquivos.")
else:
    print("O JSON não é uma lista. Verifique a estrutura do arquivo.")
