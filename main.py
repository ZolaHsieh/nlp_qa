import torch
from model import Model
from datasets import load_metric
from datasets import load_dataset
from transformers import pipeline
from metrics import compute_metrics
from data_preprocess import QATokenizer


model_checkpoint = "distilbert-base-cased" # "bert-base-cased"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_length = 384
stride = 128
model_name = 'my_qa_model'

if __name__ == '__main__':

    ## load dataset
    print("=== Load data ===")
    raw_datasets = load_dataset("squad")

    ## tokenizer & data preprocess
    print("=== Tokenizer & Data Preprocess ===")
    tokenizer = QATokenizer(checkpoint = model_checkpoint,
                            max_length = max_length,
                            stride = stride)


    train_dataset = raw_datasets['train'].map(tokenizer.tokenize_fn_train,
                                          batched=True,
                                          remove_columns=raw_datasets['train'].column_names)
    print(f"Train: rawdata: {len(raw_datasets['train'])}, processed data: {len(train_dataset)}")
    
    val_dataset = raw_datasets['validation'].map(tokenizer.tokenize_fn_val,
                                                 batched=True,
                                                 remove_columns=raw_datasets['validation'].column_names)
    print(f"Validation: rawdata: {len(raw_datasets['validation'])}, processed data: {len(val_dataset)}")


    ## model training
    print("=== Model Training ===")
    model = Model(model_checkpoint)
    trainer = model.get_trainer('distilbert-finetuned-squad',
                                train_dataset,
                                val_dataset,
                                tokenizer.tokenizer)
    trainer.train()

    ## model eval & mertrics preparation
    print("=== Model Eval ===")
    metric = load_metric("squad")
    
    trainer_output = trainer.predict(val_dataset)
    predictions, _, _ = trainer_output
    start_logits, end_logits = predictions

    compute_metrics(metric,
            start_logits,
            end_logits,
            val_dataset, # processed
            raw_datasets["validation"])

    ## save model
    print("=== Save Model ===")
    trainer.save_model(model_name)


    ## reload and inference
    print("=== Reload Model and Inference ===")
    qa_model = pipeline(task = 'question-answering',
              model=model_name,
              device=0)
    
    context = "Today I went to the store to purchase a carton of milk."
    question = "What did I buy?"
    output = qa_model(context=context, question=question)

    ## print result
    print(f"context: {context}")
    print(f"question: {question}")
    print(f"output:{output}")
