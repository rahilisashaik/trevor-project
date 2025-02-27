import Fastify from 'fastify';
import WebSocket from 'ws';
import fs from 'fs';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import OpenAI from "openai";
import whisper from 'whisper';  // Using OpenAI Whisper locally
import { fileURLToPath } from 'url';

dotenv.config();
const { OPENAI_API_KEY } = process.env;
if (!OPENAI_API_KEY) {
    console.error('Missing OpenAI API key. Please set it in the .env file.');
    process.exit(1);
}

const fastify = Fastify();
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);

const SYSTEM_MESSAGE = `
You are an AI agent designed to provide support and comfort to individuals reaching out to a suicide hotline. Your role is to engage in a compassionate and understanding conversation, ensuring the individual feels heard and supported.

Please ask the following questions one at a time in a conversational manner gently and with empathy, but talk with a kind of fast voice and don't talk too slow. Make sure you respond accurately, if they say they're happy then don't say I'm sorry to hear that:

1. How are you feeling?
2. Have you thought about committing suicide lately?
3. Do you need urgent help?

Feel free to ask follow-up questions to better understand their situation and provide comfort. Your primary goal is to ensure they feel safe and to guide them towards seeking further help if needed. Remember to be kind, patient, and supportive throughout the conversation.
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

// Create directories if they don't exist


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const audioDir = path.join(__dirname, 'audio');
if (!fs.existsSync(audioDir)) {
    fs.mkdirSync(audioDir, { recursive: true });
}

let audioBuffers = [];
const rawAudioFilePath = path.join(audioDir, 'conversation_audio_raw.wav');
const convertedAudioFilePath = path.join(audioDir, 'conversation_audio_converted.wav');

const openai = new OpenAI(OPENAI_API_KEY);

async function decideHelp(conversationTranscript) {
    try {
        console.log("\nAnalyzing transcript for suicide status...\n");
        console.log("conversationTranscript:")
        console.log(conversationTranscript);

        const completion = await openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
                { role: "system", content: SYSTEM_MESSAGE },
                { role: "user", content: `Here is the transcript of the call:\n\n"${conversationTranscript}"\n\nDoes this person urgently need care? Response simply by saying this person is a SCORE out of ten. Replace SCORE with however dire the circumstances are.` }
            ],
            temperature: 0
        });

        const gptResponse = completion.choices[0].message;
        console.log("status:")
        console.log(gptResponse);

    } catch (error) {
        console.error("Error determining help status:", error);
    }
}

async function transcribeAudioAndDecideHelp() {
    try {
        console.log("Transcribing audio using Whisper...");

        // Save raw audio
        fs.writeFileSync(rawAudioFilePath, Buffer.concat(audioBuffers));
        console.log(`Raw audio saved at: ${rawAudioFilePath}`);

        await new Promise((resolve, reject) => {
            ffmpeg(rawAudioFilePath)
                .outputOptions('-ar 16000') // Set sample rate for Whisper (16kHz)
                .toFormat('wav')
                .save(convertedAudioFilePath)
                .on('end', () => {
                    console.log(`Converted audio saved at: ${convertedAudioFilePath}`);
                    resolve();
                })
                .on('error', reject);
        });

        const result = await whisper.transcribe({
            file: convertedAudioFilePath,
            model: 'base', // Change to 'medium' or 'large' for better accuracy
            language: 'en'
        });

        console.log("\nWhisper Transcript:\n", result.text.trim());
        decideHelp(result.text.trim());

    } catch (error) {
        console.error("Error during transcription or help analysis:", error);
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

                            // Store audio payload for transcription
                            audioBuffers.push(Buffer.from(data.media.payload, 'base64'));
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
                await transcribeAudioAndDecideHelp();
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
