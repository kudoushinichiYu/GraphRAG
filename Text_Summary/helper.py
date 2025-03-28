import os
import torch
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GenerationConfig
from transformers import pipeline


def clean_text(text):
    """
    This function removes unnecessary whitespace from the input text.

    :param text: The input string to be cleaned
    :return: The cleaned string
    """

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_summary_bart(text, max_len=150):
    """
    Function to summarize humanitarian text using a summarization model (BART in this case)
    :param text: text to be summarized
    :param max_len: maximum length of the summary (in tokens)
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')

    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=1024)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], max_length=max_len, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode summary and return it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def text_summary_T5(text, max_len=150):
    # Load the tokenizer and summarization model
    tokenizer = AutoTokenizer.from_pretrained("t5-small")  # You can use "t5-base" or "facebook/bart-large-cnn" for larger models.
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    # Preprocess the input for the model
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=max_len, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def text_summary_pegasus(text, max_len=150):
    # Load Pegasus tokenizer and model
    model_name = "google/pegasus-xsum"  # You can use other Pegasus models like "google/pegasus-cnn_dailymail"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocess the text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(
    inputs["input_ids"],
    max_length=max_len,
    min_length=30,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
    )

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def text_summary_deepseek(text, max_len=150):
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    messages = [
        {"role": "user", 
         "content": "Please help me summarize the article and keep key time and place in your summary:"+text}
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=max_len)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return result

def text_summary_llama(text, max_len = 150):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipe1 = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a professional research assistant tasked with summarizing national events in a \
         formal, structured, and objective manner. Your summaries should resemble a research report, with clear sections \
         and a focus on key details."},
        {"role": "user", "content": "Please summarize the following article in the style of a research report. Ensure the summary includes the following elements:\n"
                   "1. **Background**: Briefly describe the context or significance of the event.\n"
                   "2. **Key Events**: Highlight the main events, including specific times and locations.\n"
                   "3. **Impact or Consequences**: Discuss the potential or actual impact of the event.\n"
                   "4. **Conclusion**: Provide a concise conclusion summarizing the event's importance.\n"
                   "Maintain a formal and professional tone throughout. Here is the article: " + text},
    ]

    outputs = pipe1(
        messages,
        max_new_tokens=max_len,
    )

    result = outputs[0]["generated_text"][-1]
    return result.get('content')

def text_summary_deepseekllama(text, max_len = 150):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    pipe1 = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a professional chatbot who writes reports about national events."},
        {"role": "user", "content": "Please help me summarize the article and keep key time and place in your summary' : "+text},
    ]

    outputs = pipe1(
        messages,
        max_new_tokens=max_len,
    )

    result = outputs[0]["generated_text"][-1]
    return result.get('content')

# FAISS-based text retrieval
class TextRetriever:
    def __init__(self, folder_path, index_path="faiss_index_cosine.bin", filenames_path="filenames.npy"):
        self.folder_path = folder_path
        self.index_path = index_path
        self.filenames_path = filenames_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Check if index exists, otherwise build it
        if not os.path.exists(index_path) or not os.path.exists(filenames_path):
            print("Index not found. Building vector database...")
            self.build_vector_base()

        # Load FAISS index and filenames
        self.index = faiss.read_index(self.index_path)
        self.filenames = np.load(self.filenames_path)

    def build_vector_base(self):
        """Reads text files, encodes them, and builds a FAISS index."""
        documents = []
        filenames = []

        # Read text files from the folder
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read().strip()
                    documents.append(text)
                    filenames.append(filename)

        # Convert documents to embeddings
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize for cosine similarity

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index.add(embeddings)

        # Save index and filenames
        faiss.write_index(index, self.index_path)
        np.save(self.filenames_path, np.array(filenames))
        print("Vector database built successfully!")

    def search(self, query, top_k=5):
        """Retrieve the top_k most relevant text documents for the query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)  # Normalize

        # FAISS Search
        scores, indices = self.index.search(query_embedding, top_k)
        print(indices)
        retrieved_docs = []
        for idx in indices[0]:
            file_path = os.path.join(self.folder_path, self.filenames[idx])
            print(self.filenames[idx])
            with open(file_path, "r", encoding="utf-8") as file:
                retrieved_docs.append(file.read().strip())
        return retrieved_docs




# def text_summary_bert(text, max_len=150):
#     """
#     Summarizes text by ranking sentences based on their importance using a BERT model.
#
#     :param text: str, input text to be summarized
#     :param max_len: int, maximum length of the summary in tokens
#     :return: str, summarized text
#     """
#     # Load the tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#     model = AutoModelForMaskedLM.from_pretrained(
#         'bert-base-uncased',
#         ignore_mismatched_sizes=True,
#         trust_remote_code=True
#     )
#
#     # Split text into sentences and clean them
#     sentences = [s.strip() for s in text.split('.') if s.strip()]
#     if not sentences:
#         return "Input text is empty or invalid."
#
#     sentence_scores = []
#
#     # Rank sentences based on their importance using BERT
#     for sentence in sentences:
#         inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         # Compute the average logit values as the sentence importance score
#         score = outputs.logits.softmax(dim=-1).mean().item()
#         sentence_scores.append((sentence, score))
#
#     # Sort sentences by importance in descending order
#     sentence_scores.sort(key=lambda x: x[1], reverse=True)
#
#     # Construct the summary with top-ranked sentences
#     summary_sentences = []
#     current_len = 0
#     for sentence, _ in sentence_scores:
#         token_count = len(tokenizer.tokenize(sentence))
#         if current_len + token_count <= max_len:
#             summary_sentences.append(sentence)
#             current_len += token_count
#         else:
#             break
#
#     # Combine selected sentences into a cohesive summary
#     summary = ' '.join(summary_sentences)
#     return summary
#