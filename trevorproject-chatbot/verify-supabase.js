import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config();

const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_KEY
);

async function verifySupabaseSetup() {
    try {
        console.log('Verifying Supabase connection...');
        
        // Test connection
        const { data: testData, error: testError } = await supabase
            .from('chat_sessions')
            .select('count')
            .limit(1);

        if (testError) {
            console.error('Error connecting to Supabase:', testError);
            return;
        }

        console.log('Successfully connected to Supabase!');

        // Verify table structure
        const { data: tableInfo, error: tableError } = await supabase
            .rpc('get_table_info', { table_name: 'chat_sessions' });

        if (tableError) {
            console.error('Error getting table info:', tableError);
            return;
        }

        console.log('\nTable Structure:');
        console.log('----------------');
        console.log(JSON.stringify(tableInfo, null, 2));

        // Test insert
        const testRecord = {
            name: 'test_user',
            phone_number: '1234567890',
            responses: ['test response'],
            timestamp: new Date().toISOString(),
            sentiment_analysis: { sentiment: 'neutral', intensity: 5 },
            urgency_score: 1
        };

        console.log('\nTesting data insertion...');
        const { data: insertData, error: insertError } = await supabase
            .from('chat_sessions')
            .insert([testRecord])
            .select();

        if (insertError) {
            console.error('Error inserting test data:', insertError);
            return;
        }

        console.log('Successfully inserted test data!');
        console.log('Test record:', insertData);

        // Clean up test data
        const { error: deleteError } = await supabase
            .from('chat_sessions')
            .delete()
            .eq('name', 'test_user');

        if (deleteError) {
            console.error('Error cleaning up test data:', deleteError);
            return;
        }

        console.log('\nVerification complete! Supabase is properly configured.');
    } catch (error) {
        console.error('Unexpected error during verification:', error);
    }
}

verifySupabaseSetup(); 