import json

import torch
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from trl import SFTTrainer

# Directory where your PDF files are stored
DATA_PATH = "data/TESTPAPERS"
outputs_folder = "results"

loader = DirectoryLoader(DATA_PATH,
                         glob='*.pdf',
                         loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
texts = text_splitter.split_documents(documents)

texts = text_splitter.split_documents(documents)

texts_serializable = [
    {
        'page_content': doc.page_content,
        'metadata': doc.metadata,
        # Add other attributes as needed
    }
    for doc in texts
]
with open('dataset.json', 'w') as json_file:
    json.dump(texts_serializable, json_file)
    
dataset_name = "dataset.json"

dataset = load_dataset('text', data_files=dataset_name, split='train')

def format_instruction(sample):
    parsed_text = json.loads(sample['text'])
    instruction = parsed_text['instruction']
    response = parsed_text['response']
    return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{instruction}

### Response:
{response}
"""

# Define model without quantization
use_flash_attention = False
model_id = "NousResearch/Llama-2-7b-hf"  # non-gated

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_cache=False,
    # use_flash_attention=use_flash_attention,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
assert model.embed_tokens.weight.device == device

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Training arguments
args = TrainingArguments(
    output_dir=outputs_folder,
    num_train_epochs=3,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True,
)

# Trainer setup
max_seq_length = 2048
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)

# Fine-tuning and save the trained model
trainer.train()
trainer.save_model(outputs_folder)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
db = FAISS.from_documents(texts, embeddings)
db.save_local(DB_FAISS_PATH)
