import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

# 1. Preparar o Dataset
class AmazonTitlesDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=256):
        with open(filepath, 'r') as file:
            data = json.load(file)
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = f"What is the product description for: {item['title']}?"
        context = item['content']
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"

        encoding = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        labels = encoding['input_ids'].clone()
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
        }

# 2. Carregar o Tokenizer e o Modelo Base
def load_model_and_tokenizer(model_name="EleutherAI/gpt-neo-2.7B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Garantir que temos o token de padding
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))  # Redimensionar embeddings se necessário
    return tokenizer, model

# 3. Fine-Tuning do Modelo
def fine_tune_model(model, tokenizer, dataset, output_dir, batch_size=2, epochs=3):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        eval_strategy="no",  # Alteração para eval_strategy
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,  # Utilizando tokenizer conforme recomendação
    )
    device = torch.device("cpu")  # Alterar para CPU
    model.to(device)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# 4. Geração de Respostas
def generate_answer(model, tokenizer, question, context):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(inputs['input_ids'], max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 5. Interação com o Usuário
def interact_with_model(model, tokenizer):
    print("Modelo pronto! Digite sua pergunta ('sair' para encerrar).")
    while True:
        question = input("Pergunta: ")
        if question.lower() == "sair":
            break
        context = input("Contexto (descrição do produto): ")
        response = generate_answer(model, tokenizer, question, context)
        print(f"Resposta gerada: {response}\n")

# Fluxo Principal
def main():
    torch.cuda.empty_cache()
    dataset_path = "/Users/daniel.gomes/Documents/Projetos/Python/data/part9.json"
    output_dir = "./llama_finetuned"
    
    # 1. Preparar Dataset
    tokenizer, model = load_model_and_tokenizer("EleutherAI/gpt-neo-2.7B")
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