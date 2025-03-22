import torch
import re
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
        {"role": "system", "content": "You are a professional chatbot who writes reports about national events."},
        {"role": "user", "content": "Please help me summarize the article and keep key time and place in your summary' : "+text},
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