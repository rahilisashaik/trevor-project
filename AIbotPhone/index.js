import Fastify from 'fastify';
import WebSocket from 'ws';
import fs from 'fs';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import OpenAI from "openai";
import { createClient } from '@supabase/supabase-js';
import whisper from 'whisper';  // Using OpenAI Whisper locally
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

dotenv.config();
const { OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY } = process.env;

if (!OPENAI_API_KEY || !SUPABASE_URL || !SUPABASE_KEY) {
    console.error('Missing required environment variables. Please check your .env file.');
    process.exit(1);
}

// Initialize Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const fastify = Fastify();
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);

const SYSTEM_MESSAGE = `
You are an AI agent designed to provide support and comfort to individuals reaching out to a suicide hotline. Your role is to engage in a compassionate and understanding conversation, ensuring the individual feels heard and supported.

First, ask for their name. When they provide their name, respond with exactly this format: "Thank you for sharing your name, [name]. Could you please confirm your phone number as well?" When they provide their phone number, respond with exactly this format: "Thank you for confirming your phone number, [name]. Now, let's talk about how you're feeling." Then proceed with the following questions one at a time in a conversational manner gently and with empathy, but talk with a kind of fast voice and don't talk too slow. Make sure you respond accurately, if they say they're happy then don't say I'm sorry to hear that:

1. How are you feeling?
2. Have you thought about committing suicide lately?
3. Do you need urgent help?

Throughout the conversation, address the person by their name to make it more personal and comforting. Feel free to ask follow-up questions to better understand their situation and provide comfort. Your primary goal is to ensure they feel safe and to guide them towards seeking further help if needed. Remember to be kind, patient, and supportive throughout the conversation.
Let them know they can hang up the phone after you ask your 3 questions.
`;

const VOICE = 'alloy';
const PORT = process.env.PORT || 5050;
const LOG_EVENT_TYPES = [
    'response.content.done',
    'rate_limits.updated',
    'response.done',
    'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started',
    'session.created'
];

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const audioDir = path.join(__dirname, 'audio');

if (!fs.existsSync(audioDir)) {
    console.log(`Creating directory: ${audioDir}`);
    fs.mkdirSync(audioDir, { recursive: true });
} else {
    console.log(`Directory already exists: ${audioDir}`);
}

let audioBuffers = [];
const rawAudioFilePath = path.join(audioDir, 'conversation_audio_raw.wav');
const convertedAudioFilePath = path.join(audioDir, 'conversation_audio_converted.wav');

const openai = new OpenAI(OPENAI_API_KEY);

// Add variable to store user's phone number
let userPhoneNumber = null;

async function generateSummary(transcript) {
    try {
        const completion = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: "You are a helpful assistant that summarizes conversation transcripts."
                },
                {
                    role: "user",
                    content: `Summarize the following audio transcript in 3–4 sentences. Focus on the main issue discussed, the speaker's emotional state, and any signs of urgency or distress:\n\n${transcript}`
                }
            ],
            temperature: 0
        });

        return completion.choices[0].message.content;
    } catch (error) {
        console.error('Error generating summary:', error);
        return null;
    }
}

async function analyzeSpeechEmotion(audioFilePath) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', ['sentiment_analysis.py', audioFilePath]);
        let result = '';
        let error = '';

        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            error += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                console.error('Error in speech emotion analysis:', error);
                resolve(null);
                return;
            }
            try {
                const emotionScores = JSON.parse(result);
                resolve(emotionScores);
            } catch (e) {
                console.error('Error parsing emotion scores:', e);
                resolve(null);
            }
        });
    });
}

async function storeCallData(transcription, audioFilePath, name, phoneNumber) {
    try {
        // Read the audio file
        const audioFile = fs.readFileSync(audioFilePath);
        
        // Upload audio file to Supabase Storage
        const { data: audioData, error: audioError } = await supabase.storage
            .from('call-recordings')
            .upload(`${Date.now()}_call.wav`, audioFile, {
                contentType: 'audio/wav'
            });

        if (audioError) throw audioError;

        // Get the public URL for the audio file
        const { data: { publicUrl: audioUrl } } = supabase.storage
            .from('call-recordings')
            .getPublicUrl(audioData.path);

        const currentTimestamp = new Date().toISOString();

        // Generate summary for the current call
        const summary = await generateSummary(transcription);

        // Analyze speech emotion
        const emotionScores = await analyzeSpeechEmotion(audioFilePath);

        if (phoneNumber) {
            // Check if caller exists
            const { data: existingCaller, error: callerError } = await supabase
                .from('callers')
                .select('*')
                .eq('phone_number', phoneNumber)
                .single();

            if (callerError && callerError.code !== 'PGRST116') throw callerError;

            if (existingCaller) {
                // Update existing caller
                const { error: updateError } = await supabase
                    .from('callers')
                    .update({
                        name: name || existingCaller.name,
                        aggregated_transcript: existingCaller.aggregated_transcript 
                            ? `${existingCaller.aggregated_transcript}\n\n--- New Call ${currentTimestamp} ---\n${transcription}`
                            : transcription,
                        aggregated_summary: existingCaller.aggregated_summary
                            ? `${existingCaller.aggregated_summary}\n\n--- New Call ${currentTimestamp} ---\n${summary}`
                            : summary,
                        last_call_timestamp: currentTimestamp,
                        updated_at: currentTimestamp
                    })
                    .eq('phone_number', phoneNumber);

                if (updateError) throw updateError;
            } else {
                // Create new caller
                const { error: insertError } = await supabase
                    .from('callers')
                    .insert([
                        {
                            phone_number: phoneNumber,
                            name: name,
                            aggregated_transcript: transcription,
                            aggregated_summary: summary,
                            last_call_timestamp: currentTimestamp
                        }
                    ]);

                if (insertError) throw insertError;
            }
        }

        // Store individual call record
        const { error: callError } = await supabase
            .from('calls')
            .insert([
                {
                    phone_number: phoneNumber,
                    call_timestamp: currentTimestamp,
                    audio_url: audioUrl,
                    transcript: transcription,
                    summary: summary,
                    duration: Math.floor(audioBuffers.length / 8000), // Approximate duration in seconds
                    emotion_scores: emotionScores // Add emotion scores to the call record
                }
            ]);

        if (callError) throw callError;

        console.log('Call data stored successfully in Supabase');
        return true;

    } catch (error) {
        console.error('Error storing call data:', error);
        throw error;
    }
}

async function extractUserInfo(transcript) {
    try {
        const completion = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: `You are a helpful assistant that extracts user information from conversation transcripts. 
                    Extract the user's name and phone number from the following transcript. 
                    
                    For phone numbers:
                    - Look for numbers in formats like: 510-717-7239, (510) 717-7239, 510.717.7239, or 5107177239
                    - Remove any spaces, parentheses, or dashes to store in E.164 format (e.g., 5107177239)
                    - Only extract numbers that look like phone numbers (10 digits)
                    - If multiple numbers are found, use the one that looks most like a phone number
                    
                    Return the information in this exact JSON format:
                    {
                        "name": "extracted name or null if not found",
                        "phone": "extracted phone number in E.164 format or null if not found"
                    }
                    Only return the JSON object, nothing else.`
                },
                {
                    role: "user",
                    content: transcript
                }
            ],
            temperature: 0
        });

        const response = completion.choices[0].message.content;
        const userInfo = JSON.parse(response);
        console.log('Extracted user info:', userInfo);
        return userInfo;

    } catch (error) {
        console.error('Error extracting user info:', error);
        return { name: null, phone: null };
    }
}

async function processTranscriptWithAIQuestions(transcription) {
    try {
        const completion = await openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: "You are formatting a conversation transcript. Format the conversation as a dialogue between AI and User, ensuring the AI's standard questions are included in order. Format each line with 'AI:' or 'User:' prefix."
                },
                {
                    role: "user",
                    content: `Format this transcript as a dialogue, including these AI questions in order:
                    1. "How are you feeling?"
                    2. "Have you thought about committing suicide lately?"
                    3. "Do you need urgent help?"
                    
                    Original transcript:
                    ${transcription}`
                }
            ],
            temperature: 0
        });

        return completion.choices[0].message.content;
    } catch (error) {
        console.error('Error processing transcript:', error);
        return transcription;
    }
}

async function transcribeAudioAndDecideHelp() {
    try {
        console.log("Transcribing audio using OpenAI API...");

        // Save raw audio
        fs.writeFileSync(rawAudioFilePath, Buffer.concat(audioBuffers));
        console.log(`Raw audio saved at: ${rawAudioFilePath}`);

        await new Promise((resolve, reject) => {
            ffmpeg(rawAudioFilePath)
                .inputFormat('mulaw')
                .inputOptions('-ar 8000')
                .audioCodec('pcm_s16le')
                .outputOptions([
                    '-ar 8000',
                    '-ac 1'
                ])
                .toFormat('wav')
                .save(convertedAudioFilePath)
                .on('end', () => {
                    console.log(`Converted audio saved at: ${convertedAudioFilePath}`);
                    resolve();
                })
                .on('error', reject);
        });

        console.log("converted audio file saved, now using OpenAI API to transcribe");
        
        // Use OpenAI's transcription API
        const audioFile = fs.createReadStream(convertedAudioFilePath);
        const transcription = await openai.audio.transcriptions.create({
            model: "gpt-4o-transcribe",
            file: audioFile,
            response_format: "text"
        });

        console.log("\nComplete Call Transcript:\n", transcription);

        // Extract user information using GPT-4
        const { name, phone } = await extractUserInfo(transcription);
        
        // Process the transcript with AI questions
        const formattedTranscript = await processTranscriptWithAIQuestions(transcription);
        
        // Store the call data
        await storeCallData(formattedTranscript, convertedAudioFilePath, name, phone);
        
        // Return the transcription and user info
        return { transcription: formattedTranscript, name, phone };

    } catch (error) {
        console.error("Error during transcription or storage:", error);
        return { transcription: null, name: null, phone: null };
    }
}

fastify.get('/', async (request, reply) => {
    reply.send({ message: 'Twilio Media Stream Server is running!' });
});

fastify.all('/incoming-call', async (request, reply) => {
    const twimlResponse = `<?xml version="1.0" encoding="UTF-8"?>
                          <Response>
                              <Say>Hi</Say>
                              <Pause length="1"/>
                              <Connect>
                                  <Stream url="wss://${request.headers.host}/media-stream" />
                              </Connect>
                          </Response>`;
    reply.type('text/xml').send(twimlResponse);
});

fastify.register(async (fastify) => {
    fastify.get('/media-stream', { websocket: true }, (connection, req) => {
        console.log('Client connected');

        const openAiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01', {
            headers: {
                Authorization: `Bearer ${OPENAI_API_KEY}`,
                "OpenAI-Beta": "realtime=v1"
            }
        });

        let streamSid = null;
        audioBuffers = [];

        const sendSessionUpdate = () => {
            const sessionUpdate = {
                type: 'session.update',
                session: {
                    turn_detection: { type: 'server_vad' },
                    input_audio_format: 'g711_ulaw',
                    output_audio_format: 'g711_ulaw',
                    voice: VOICE,
                    instructions: SYSTEM_MESSAGE,
                    modalities: ["text", "audio"],
                    temperature: 0.8,
                }
            };
            console.log('Sending session update:', JSON.stringify(sessionUpdate));
            openAiWs.send(JSON.stringify(sessionUpdate));
        };

        openAiWs.on('open', () => {
            console.log('Connected to the OpenAI Realtime API');
            setTimeout(sendSessionUpdate, 250);
        });

        openAiWs.on('message', (data) => {
            try {
                const response = JSON.parse(data);

                if (LOG_EVENT_TYPES.includes(response.type)) {
                    console.log(`Received event: ${response.type}`, response);
                }

                if (response.type === 'response.content.delta' && response.delta) {
                    // Extract name from the AI's response when it acknowledges the name
                    if (response.delta.includes('Thank you for sharing your name,')) {
                        const nameMatch = response.delta.match(/Thank you for sharing your name, (\w+)/i);
                        if (nameMatch) {
                            userName = nameMatch[1];
                            console.log(`User's name extracted: ${userName}`);
                        }
                    }
                    // Extract phone number from the AI's response when it acknowledges the phone number
                    if (response.delta.includes('Thank you for confirming your phone number,')) {
                        const phoneMatch = response.delta.match(/Thank you for confirming your phone number, (\w+)/i);
                        if (phoneMatch) {
                            userPhoneNumber = phoneMatch[1];
                            console.log(`User's phone number extracted: ${userPhoneNumber}`);
                        }
                    }
                }

                if (response.type === 'response.audio.delta' && response.delta) {
                    const audioDelta = {
                        event: 'media',
                        streamSid: streamSid,
                        media: { payload: Buffer.from(response.delta, 'base64').toString('base64') }
                    };
                    connection.send(JSON.stringify(audioDelta));
                }

            } catch (error) {
                console.error('Error processing OpenAI message:', error, 'Raw message:', data);
            }
        });

        connection.on('message', (message) => {
            try {
                const data = JSON.parse(message);

                switch (data.event) {
                    case 'media':
                        if (openAiWs.readyState === WebSocket.OPEN) {
                            const audioAppend = {
                                type: 'input_audio_buffer.append',
                                audio: data.media.payload
                            };
                            openAiWs.send(JSON.stringify(audioAppend));
                            console.log("Received audio data chunk:", data.media.payload.length);


                            // Store audio payload for transcription
                            const rawAudio = Buffer.from(data.media.payload, 'base64');

                            // Ensure it's saved in proper PCM format
                            audioBuffers.push(rawAudio);
                            console.log('Audio buffer length:', audioBuffers.length);  // Log buffer length

                        }
                        break;
                    case 'start':
                        streamSid = data.start.streamSid;
                        console.log('Incoming stream has started', streamSid);
                        break;
                    default:
                        console.log('Received non-media event:', data.event);
                        break;
                }
            } catch (error) {
                console.error('Error parsing message:', error, 'Message:', message);
            }
        });

        connection.on('close', async () => {
            console.log('Client disconnected.');

            if (audioBuffers.length > 0) {
                const { transcription, name, phone } = await transcribeAudioAndDecideHelp();
                if (transcription) {
                    const formattedTranscript = await processTranscriptWithAIQuestions(transcription);
                    // Store the formatted transcript instead of the raw one
                    await storeCallData(formattedTranscript, convertedAudioFilePath, name, phone);
                }
            } else {
                console.error('No audio recorded');
            }

            if (openAiWs.readyState === WebSocket.OPEN) openAiWs.close();
        });

        openAiWs.on('close', () => {
            console.log('Disconnected from the OpenAI Realtime API');
        });

        openAiWs.on('error', (error) => {
            console.error('Error in the OpenAI WebSocket:', error);
        });

        connection.on('error', (error) => {
            console.error('Error in the Twilio WebSocket:', error);
        });
    });
});

fastify.listen({ port: PORT }, (err) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`Server is listening on port ${PORT}`);
});
