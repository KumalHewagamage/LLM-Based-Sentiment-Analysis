from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, load_metric
from sklearn.metrics import classification_report
import numpy as np
import torch
from torch import nn

def get_predictions(data_test, classifier):
    """Get predictions from the classifier."""
    data_test['roberta_sentiment'] = data_test['text'].apply(lambda x: classifier(x))
    data_test['predicted'] = data_test["roberta_sentiment"].apply(lambda row: 1 if row[0]["label"] == 'POSITIVE' else 0)
    return data_test

def back_translation(input_text, translation_en_to_de, tokenizer, model_de_to_en):
    """Perform back translation for data augmentation."""
    review_en_to_de = translation_en_to_de(input_text)
    text_en_to_de = review_en_to_de[0]['translation_text']
    input_ids = tokenizer(text_en_to_de, return_tensors="pt", add_special_tokens=False, max_length=512, truncation=True).input_ids
    output_ids = model_de_to_en.generate(input_ids)[0]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def create_data_samples(data, translation_en_to_de, tokenizer, model_de_to_en):
    """Create additional samples using back translation."""
    count_labels = data["label"].value_counts()
    n = count_labels[1] - count_labels[0]
    data_temp = data[data.label == 0].sample(n=n, replace=True, random_state=1)
    data_temp['text'] = data_temp["text"].apply(lambda x: back_translation(x, translation_en_to_de, tokenizer, model_de_to_en))
    data_sampled = pd.concat([data_temp, data], ignore_index=True)
    return data_sampled.sample(frac=1, random_state=0)  # Shuffle the data

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    
    return {"accuracy": accuracy, "f1": f1}

class CustomTrainer(Trainer):
    """Custom trainer with weighted loss."""
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, reduction='mean')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
