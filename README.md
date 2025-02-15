# Web-Scraping-Chatbot
# AI-Powered Web Scraping Chatbot

This project is a chatbot that interacts with a given website URL and answers user queries using the ChatGPT API. The chatbot scrapes the website, processes the content, and provides relevant responses based on user inputs.

## Features
- Extracts and processes website content
- Uses ChatGPT API for intelligent responses
- Implements a Retrieval-Augmented Generation (RAG) approach
- Provides a simple UI via Streamlit
- Allows users to download fetched webpage content

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed and set up a virtual environment.

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-chatbot-webscraper.git
   cd ai-chatbot-webscraper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add your API key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the chatbot:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter the URL of the website you want to extract information from.
2. Ask a question about the website's content.
3. The chatbot will fetch the content, process it, and provide an answer.
4. Optionally, download the extracted content as a text file.

## Technologies Used
- Python
- OpenAI ChatGPT API
- Streamlit
- LangChain
- BeautifulSoup (for web scraping)

## Folder Structure
```
├── app.py           # Main application script
├── README.md        # Project documentation
├── requirements.txt # Dependencies list
├── .env             # Environment variables file (not committed)
```

## License
This project is open-source and available under the MIT License.

