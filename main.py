from data_processing import load_and_clean_data, visualize_class_distribution
from model_utils import get_predictions, create_data_samples, compute_metrics, CustomTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, TrainingArguments
import numpy as np
import torch

# Load and clean data
file_path = "data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
df = load_and_clean_data(file_path)
visualize_class_distribution(df)

# Split data
train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6 * len(df)), int(.8 * len(df))])

# Initialize translation models for data augmentation
translation_en_to_de = pipeline("translation_en_to_de", model='t5-base')
tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")
model_de_to_en = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")

# Data augmentation
train = create_data_samples(train, translation_en_to_de, tokenizer, model_de_to_en)
visualize_class_distribution(train)

# Prepare data for training
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
train = Dataset.from_pandas(train).remove_columns(['__index_level_0__'])
validate = Dataset.from_pandas(validate).remove_columns(['__index_level_0__'])
test = Dataset.from_pandas(test).remove_columns(['__index_level_0__'])

# Tokenization and data collation
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english", num_labels=2)

# Training arguments
training_args = TrainingArguments(
   output_dir="finetuning-sentiment-roberta-large-english",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   save_strategy="epoch",
   weight_decay=0.01
)

# Training
trainer = CustomTrainer(
   model=model,
   args=training_args,
   train_dataset=train,
   eval_dataset=validate,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model("sentiment-roberta-large-english-fine-tuned")
