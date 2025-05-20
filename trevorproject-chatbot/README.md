# Trevor Project Chatbot

A text-based chatbot that conducts conversations and analyzes responses for mental health support.

## Features

- Interactive text-based conversation
- Sentiment analysis of responses
- Urgency score calculation
- Data storage in Supabase
- Real-time response analysis

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

3. Set up Supabase:
   - Create a new project
   - Create a table called `chat_sessions` with the following columns:
     - name (text)
     - phone_number (text)
     - responses (jsonb)
     - timestamp (timestamp)
     - sentiment_analysis (jsonb)
     - urgency_score (integer)

## Usage

Run the chatbot:
```bash
node index.js
```

The chatbot will:
1. Ask for your name
2. Ask for your phone number
3. Ask about your feelings
4. Ask about suicidal thoughts
5. Ask if you need urgent help
6. Analyze your responses
7. Store the data in Supabase
8. Display a summary of the conversation

## Data Analysis

The chatbot performs two types of analysis:

1. **Sentiment Analysis**
   - Analyzes emotional content
   - Identifies key words
   - Provides a summary
   - Rates intensity

2. **Urgency Score**
   - Scores from 1-10
   - Based on response content
   - Considers multiple factors
   - Helps prioritize support needs

## Contributing

Feel free to submit issues and enhancement requests. 