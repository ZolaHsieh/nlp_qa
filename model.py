from transformers import TrainingArguments, Trainer, AutoModelForQuestionAnswering

class Model():
    def __init__(self, checkpoint):
        self.model =  AutoModelForQuestionAnswering.from_pretrained(checkpoint)

    def get_trainer(self,
                    output_dir,
                    train_dataset,
                    val_dataset,
                    tokenizer,
                    epochs = 3):

        #設定參數
        args = TrainingArguments(output_dir = output_dir, #'distilbert-finetuned-squad',
                                evaluation_strategy = 'no',
                                save_strategy = 'epoch',
                                learning_rate = 2e-5,
                                num_train_epochs = epochs,
                                weight_decay = 0.01,
                                fp16 = True)
        # setup trainer
        trainer = Trainer(model = self.model,
                        args = args,
                        train_dataset = train_dataset,
                        eval_dataset = val_dataset,
                        tokenizer = tokenizer)
        return trainer
