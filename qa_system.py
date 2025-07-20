import pyodbc
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# --- 1. Database Connection and Data Extraction ---
def get_data_from_db():
    # Replace with your actual database connection details
    server = 'your_server_name'
    database = 'your_database_name'
    username = 'your_username'
    password = 'your_password'
    table_name = 'your_table_name'

    try:
        # Establish the connection
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password}'
        )
        cursor = conn.cursor()

        # Fetch data from the specified table
        print(f"Fetching data from table: {table_name}")
        cursor.execute(f"SELECT * FROM {table_name}")

        # Get column names
        columns = [column[0] for column in cursor.description]

        # Fetch all rows
        rows = cursor.fetchall()

        # Close the connection
        conn.close()

        print(f"Successfully fetched {len(rows)} rows.")
        return columns, rows

    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"Database connection error: {sqlstate}")
        print(ex)
        return None, None

# --- 2. Data Formatting ---
def format_data(columns, rows):
    formatted_texts = []
    for row in rows:
        text = []
        for i, value in enumerate(row):
            text.append(f"{columns[i]}: {value}")
        formatted_texts.append(", ".join(text))
    return formatted_texts

# --- 3. Generate and Store Embeddings ---
def create_faiss_index(texts, model_name='all-MiniLM-L6-v2'):
    print("Loading sentence transformer model...")
    model = SentenceTransformer(model_name)

    print("Generating embeddings for the data...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings, dtype=np.float32))

    print("FAISS index created successfully.")
    return index, model

if __name__ == '__main__':
    # Example of fetching and formatting data
    cols, db_rows = get_data_from_db()
    if db_rows:
        formatted_data = format_data(cols, db_rows)
        print("\n--- Formatted Data Example ---")
        for i in range(min(5, len(formatted_data))):
            print(formatted_data[i])

        # --- Create FAISS index ---
        faiss_index, sentence_model = create_faiss_index(formatted_data)

# --- 4. Load Local LLM ---
def load_llm(model_name="microsoft/Phi-3-mini-4k-instruct"):
    print(f"Loading local LLM: {model_name}")

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    # Set pad_token_id to eos_token_id if it's not set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1, # Use device index for pipeline
    )

    print("LLM loaded successfully.")
    return llm_pipeline

# --- 5. Question Answering ---
def answer_question(question, index, sentence_model, llm_pipeline, formatted_texts, top_k=3):
    print(f"\nAnswering question: {question}")

    # 1. Generate embedding for the question
    question_embedding = sentence_model.encode([question])

    # 2. Perform similarity search in FAISS
    distances, indices = index.search(np.array(question_embedding, dtype=np.float32), top_k)

    # 3. Retrieve the most relevant documents
    context_docs = [formatted_texts[i] for i in indices[0]]
    context = "\n".join(context_docs)

    print("--- Retrieved Context ---")
    print(context)

    # 4. Build the prompt for the LLM
    prompt = f"""
    You are a helpful AI assistant. Use the following context to answer the question.
    If the answer is not in the context, say that you don't know.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    # 5. Generate the answer using the LLM
    print("\n--- Generating Answer ---")
    response = llm_pipeline(
        prompt,
        max_new_tokens=150,
        num_return_sequences=1,
        truncation=True,
        eos_token_id=llm_pipeline.tokenizer.eos_token_id
    )

    answer = response[0]['generated_text'].split("Answer:")[-1].strip()
    return answer

if __name__ == '__main__':
    # Example of fetching and formatting data
    cols, db_rows = get_data_from_db()

    if db_rows:
        formatted_data = format_data(cols, db_rows)
        print("\n--- Formatted Data Example ---")
        for i in range(min(5, len(formatted_data))):
            print(formatted_data[i])

        # --- Create FAISS index ---
        faiss_index, sentence_model = create_faiss_index(formatted_data)

        # --- Load LLM ---
        llm = load_llm()

        # --- Example Question ---
        user_question = "Your question here" # Replace with a question relevant to your data
        final_answer = answer_question(user_question, faiss_index, sentence_model, llm, formatted_data)
        print("\n--- Final Answer ---")
        print(final_answer)
