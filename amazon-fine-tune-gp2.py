import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

# 1. Preparar o Dataset
class AmazonTitlesDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        # Sanitizar e carregar os dados
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao carregar o arquivo JSON: {e}")

        self.data = [item for item in data if 'title' in item and 'content' in item]  # Filtrar entradas inválidas
        if len(self.data) == 0:
            raise ValueError("O dataset está vazio ou malformado.")
            
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = f"What is the description of the product '{item['title']}'?"
        context = item['content']
        prompt = f"Question: {question}\nAnswer: {context}"

        # Codificação com checagem de erros
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignorar padding na função de perda
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
        }

# 2. Carregar o Tokenizer e o Modelo Base
def load_model_and_tokenizer(model_name="EleutherAI/gpt-neo-1.3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Garantir token de padding válido
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))  # Ajusta os embeddings
    return tokenizer, model

# 3. Fine-Tuning do Modelo
def fine_tune_model(model, tokenizer, dataset, output_dir, batch_size=1, epochs=3):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        fp16=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Verificar problemas no modelo antes do treinamento
    for param in model.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError("O modelo contém valores NaN ou Inf antes do treinamento.")
    
    model.to(torch.device("cpu"))
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# 4. Geração de Respostas
def generate_answer(model, tokenizer, question):
    # Sanitizar entrada
    if not question or not isinstance(question, str) or question.strip() == "":
        raise ValueError("Pergunta inválida. Por favor, insira uma pergunta válida.")
    
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(
        prompt, 
        padding=True, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Filtrar saída para remover o prompt da resposta
    answer = answer.split("Answer:")[-1].strip()
    return answer

# 5. Interação com o Usuário
def interact_with_model(model, tokenizer):
    print("Modelo pronto! Digite sua pergunta ('sair' para encerrar).")
    while True:
        question = input("Pergunta: ")
        if question.lower() == "sair":
            break
        try:
            response = generate_answer(model, tokenizer, question)
            print(f"Resposta gerada: {response}\n")
        except Exception as e:
            print(f"Erro: {e}\n")

# Fluxo Principal
def main():
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    dataset_path = "/Users/daniel.gomes/Documents/Projetos/Python/data/part9.json"
    output_dir = "./llama_finetuned"
    
    # 1. Preparar Dataset e Modelo
    print("Carregando modelo e tokenizer...")
    tokenizer, model = load_model_and_tokenizer("EleutherAI/gpt-neo-1.3B")
    
    print("Preparando dataset...")
    dataset = AmazonTitlesDataset(dataset_path, tokenizer)

    # 2. Executar Fine-Tuning
    print("Iniciando Fine-Tuning...")
    fine_tune_model(model, tokenizer, dataset, output_dir)

    # 3. Carregar Modelo Fine-Tuned
    print("Carregando modelo treinado...")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(output_dir)

    # 4. Interagir com o Usuário
    interact_with_model(model, tokenizer)

if __name__ == "__main__":
    main()
