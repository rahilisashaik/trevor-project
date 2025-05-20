import OpenAI from 'openai';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import readlineSync from 'readline-sync';

dotenv.config();

// Initialize OpenAI and Supabase clients
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_KEY
);

// Questions to ask
const QUESTIONS = [
    "What is your name?",
    "What is your phone number?",
    "How are you feeling?",
    "Have you thought about committing suicide lately?",
    "Do you need urgent help?"
];

// Store conversation data
let conversationData = {
    name: '',
    phone: '',
    responses: [],
    timestamp: new Date().toISOString()
};

// Function to analyze sentiment using OpenAI
async function analyzeSentiment(text) {
    try {
        const completion = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: "You are analyzing text for emotional content and sentiment. Return a JSON object with the following structure: { sentiment: 'positive/negative/neutral', intensity: 1-10, keywords: ['word1', 'word2'], summary: 'brief summary' }"
                },
                {
                    role: "user",
                    content: text
                }
            ],
            temperature: 0
        });

        return JSON.parse(completion.choices[0].message.content);
    } catch (error) {
        console.error('Error analyzing sentiment:', error);
        return null;
    }
}

// Function to calculate urgency score
async function calculateUrgencyScore(responses) {
    try {
        const completion = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: `You are analyzing suicide hotline chat data to determine an urgency score from 1-10.
                    Rules for scoring:
                    - Score of 10: Immediate intervention needed, clear immediate suicide risk
                    - Score of 8-9: High risk, in desperate need of help/support, has specific suicide plans
                    - Score of 5-7: Moderate risk, struggling a lot or considered suicide
                    - Score of 3-4: Lower risk, struggling but no immediate danger
                    - Score of 1-2: Minimal risk, seeking support but relatively stable
                    
                    Return only an integer between 1-10.`
                },
                {
                    role: "user",
                    content: `Analyze these responses and return an urgency score:
                    ${responses.join('\n')}`
                }
            ],
            temperature: 0
        });

        return parseInt(completion.choices[0].message.content);
    } catch (error) {
        console.error('Error calculating urgency score:', error);
        return 5; // Default score on error
    }
}

// Function to generate conversation summary using the same model as AIbotPhone
async function generateSummary(transcript, name) {
    try {
        const completion = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: "You are summarizing conversation transcripts in a way that focuses on the caller's responses and emotional state. Do not mention the AI or the questions asked - instead describe what the caller expressed or revealed in response to questions about their feelings, thoughts of suicide, and need for help. Use natural, flowing language that centers the caller's experience."
                },
                {
                    role: "user",
                    content: `Summarize the following conversation transcript in 3-4 sentences, focusing only on what the caller expressed or revealed about their feelings, thoughts of suicide, and need for help. Use language like 'The caller expressed/felt/revealed...' and avoid mentioning the AI or questions asked.\n\n${transcript}`
                }
            ],
            temperature: 0
        });
        return completion.choices[0].message.content;
    } catch (error) {
        console.error('Error generating summary:', error);
        return '';
    }
}

// Main conversation function
async function startConversation() {
    console.log('Welcome to the Trevor Project Chatbot');
    console.log('-----------------------------------');

    try {
        // The questions to ask
        const questions = [
            "How are you feeling?",
            "Have you thought about committing suicide lately?",
            "Do you need urgent help?"
        ];

        // Collect responses
        let transcriptLines = [];
        let name = '';
        let phone = '';
        let responses = [];

        // Ask for name
        const nameResponse = readlineSync.question("What is your name?\n");
        name = nameResponse;
        responses.push(nameResponse);
        transcriptLines.push(`User: ${nameResponse}`);

        // Ask for phone number
        const phoneResponse = readlineSync.question("What is your phone number?\n");
        phone = phoneResponse;
        responses.push(phoneResponse);
        transcriptLines.push(`User: ${phoneResponse}`);

        // Ask the main questions
        for (const q of questions) {
            transcriptLines.push(`AI: ${q}`);
            const userResponse = readlineSync.question(`${q}\n`);
            responses.push(userResponse);
            transcriptLines.push(`User: ${userResponse}`);
        }

        // Format the transcript as a single string
        const formattedTranscript = transcriptLines.join('\n');

        // Generate the summary using the AIbotPhone model
        const summary = await generateSummary(formattedTranscript, name);

        // Calculate urgency score
        const urgencyScore = await calculateUrgencyScore(responses);
        conversationData.urgencyScore = urgencyScore;

        // Prepare conversationData
        conversationData.name = name;
        conversationData.phone = phone;
        conversationData.responses = responses;
        conversationData.timestamp = new Date().toISOString();
        conversationData.transcript = formattedTranscript;
        conversationData.summary = summary;

        // Store data in Supabase (only callers and calls)
        try {
            await upsertCaller(conversationData);
            await insertCall(conversationData);
        } catch (storageError) {
            console.error('Failed to store data in Supabase:', storageError.message);
            console.log('Data will be saved locally instead...');
            // Save to local file as backup
            const fs = await import('fs');
            const backupData = {
                ...conversationData,
                error: storageError.message,
                backup_timestamp: new Date().toISOString()
            };
            fs.writeFileSync(
                `backup_${Date.now()}.json`,
                JSON.stringify(backupData, null, 2)
            );
        }

        // Display summary
        console.log('\nConversation Transcript:');
        console.log('------------------------');
        console.log(formattedTranscript);
        console.log('\nConversation Summary:');
        console.log('-------------------');
        console.log(summary);
        console.log(`Urgency Score: ${conversationData.urgencyScore}/10`);
        console.log('\nThank you for your responses. Help is available 24/7.');
    } catch (error) {
        console.error('An error occurred during the conversation:', error);
    }
}

// Update upsertCaller and insertCall to use transcript and summary
// Upsert into callers
async function upsertCaller(data) {
    const now = new Date().toISOString();
    // Check if caller exists
    const { data: existingCaller, error: selectError } = await supabase
        .from('callers')
        .select('*')
        .eq('phone_number', data.phone)
        .single();

    if (selectError && selectError.code !== 'PGRST116') {
        console.error('Error checking caller:', selectError);
        return;
    }

    if (existingCaller) {
        // Update last_call_timestamp and aggregated transcript/summary
        await supabase
            .from('callers')
            .update({
                last_call_timestamp: now,
                aggregated_transcript: (existingCaller.aggregated_transcript || '') + '\n' + data.transcript,
                aggregated_summary: (existingCaller.aggregated_summary || '') + '\n' + data.summary
            })
            .eq('phone_number', data.phone);
    } else {
        // Insert new caller
        await supabase
            .from('callers')
            .insert([{
                phone_number: data.phone,
                name: data.name,
                aggregated_transcript: data.transcript,
                aggregated_summary: data.summary,
                last_call_timestamp: now
            }]);
    }
}

// Insert into calls
async function insertCall(data) {
    await supabase
        .from('calls')
        .insert([{
            phone_number: data.phone,
            call_timestamp: data.timestamp,
            transcript: data.transcript,
            summary: data.summary,
            urgency_score: data.urgencyScore,
            source: 'Chatbot',
        }]);
}

// Start the conversation
startConversation().catch(console.error); 