# medical-db
This is an attempt to use LLM to load personal medical records into a local DB, then query them in a questions/answer mode.

Uses LangChain, OpenAI and Faiss vector DB.

**Currently, it is at 50% success rate, and that is better than other LLMs I have tried.**

## Setup
1. Install needed packages `pip install -r requirements.txt`.
1. To run this you'll need an OpenAI API Key, which you can obtain from [here](https://platform.openai.com/account/api-keys).
1. Put the API key in an environment variable named `OPENAI_API_KEY`.
1. Run `python app.py`

## Usage
The class `MedicalDB` has two methods:
  - `load_pdfs()` for creating a local DB with the [embeddings](https://platform.openai.com/docs/guides/embeddings).
  - `interact_with_db` to write questions and expect concise, accurate responses.

Comment out and Uncomment the methods based on your goals.
