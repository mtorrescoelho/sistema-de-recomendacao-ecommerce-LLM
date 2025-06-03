import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

MODEL_NAME = "neuralmind/bert-base-portuguese-cased" 

# 1. Carregar e unir os dados
def carregar_dados():
    df_feedbacks = pd.read_csv("dados/feedbacks.csv")
    df_intervalo = pd.read_csv("outputs_p/intervalo_medio_entre_compras.csv")
    df_valor_medio = pd.read_csv("outputs_p/valor_medio_por_compra.csv")

    df = df_feedbacks.merge(df_intervalo, on="ID_Utilizador", how="left")
    df = df.merge(df_valor_medio, on="ID_Utilizador", how="left")
    return df

# 2. Preparar dados
def preparar_dados(df):
    df = df[['Feedback_Texto', 'Avaliacao', 'intervalo_medio_dias', 'valor_medio_por_compra']].dropna()
    df['label'] = df['Avaliacao'].apply(lambda x: 1 if x >= 4 else 0)
    return df

# 3. Carregar modelo
def carregar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    return tokenizer, model

# 4. Tokenização
def tokenizar_textos(df, tokenizer):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda e: tokenizer(e['Feedback_Texto'], truncation=True, padding='max_length'), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset

# 5. Treinar modelo
def treinar_modelo(dataset, model):
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

    args = TrainingArguments(
        output_dir="./resultados",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_strategy="no",
        logging_steps=10,
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return trainer, val_dataset

# 6. Avaliação
def avaliar(trainer, val_dataset):
    preds = trainer.predict(val_dataset)
    pred_labels = np.argmax(preds.predictions, axis=1)
    true_labels = preds.label_ids

    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print(f"Precisão: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
    return precision, recall, f1

# 7. Pipeline
def pipeline_modelo():
    df = carregar_dados()
    df = preparar_dados(df)
    tokenizer, model = carregar_modelo()
    dataset = tokenizar_textos(df, tokenizer)
    trainer, val_dataset = treinar_modelo(dataset, model)
    return avaliar(trainer, val_dataset)

# 8. Execução
if __name__ == "__main__":
    pipeline_modelo()
