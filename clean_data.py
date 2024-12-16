# Limpeza dos dados
def clean_data(data):
    # Remover duplicados (baseado no título)
    data = [d for i, d in enumerate(data) if d['title'] not in [item['title'] for item in data[:i]]]
    # Remover títulos vazios ou nulos
    data = [d for d in data if d['title']]
    # Remover títulos com caracteres especiais indesejados
    for d in data:
        d['title'] = re.sub(r'[^a-zA-Z0-9\s]', '', d['title'])
    # Remover espaços em branco extras no começo e no final dos títulos
    for d in data:
        d['title'] = d['title'].strip()
    # Converter os títulos para minúsculas
    for d in data:
        d['title'] = d['title'].lower()
    return data
