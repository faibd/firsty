from flask import Flask, request, jsonify, Response
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
import tempfile
import base64

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# HTML frontend embedded as a string
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Agent</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
        button { padding: 10px 20px; margin: 10px; }
        #status { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Voice Agent</h1>
    <button id="startRecording">Start Recording</button>
    <button id="stopRecording" disabled>Stop Recording</button>
    <div id="status"></div>
    <audio id="audioPlayback" controls></audio>
    <script>
        let mediaRecorder;
        let audioChunks = [];

        const startButton = document.getElementById('startRecording');
        const stopButton = document.getElementById('stopRecording');
        const status = document.getElementById('status');
        const audioPlayback = document.getElementById('audioPlayback');

        startButton.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = e => {
                    audioChunks.push(e.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('audio', audioBlob, 'recording.wav');

                    status.textContent = 'Processing...';
                    const response = await fetch('/voice-agent', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();

                    if (result.error) {
                        status.textContent = 'Error: ' + result.error;
                        return;
                    }

                    status.textContent = 'Response: ' + result.text_response;
                    const audioData = atob(result.audio_base64);
                    const audioArray = new Uint8Array(audioData.length);
                    for (let i = 0; i < audioData.length; i++) {
                        audioArray[i] = audioData.charCodeAt(i);
                    }
                    const audioBlobResponse = new Blob([audioArray], { type: 'audio/mp3' });
                    audioPlayback.src = URL.createObjectURL(audioBlobResponse);
                    audioPlayback.play();
                };

                mediaRecorder.start();
                startButton.disabled = true;
                stopButton.disabled = false;
                status.textContent = 'Recording...';
            } catch (err) {
                status.textContent = 'Error accessing microphone: ' + err;
            }
        });

        stopButton.addEventListener('click', () => {
            mediaRecorder.stop();
            startButton.disabled = false;
            stopButton.disabled = true;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return Response(HTML_CONTENT, mimetype='text/html')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        return client.audio.transcriptions.create(model="whisper-1", file=f)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_chat_response(user_input):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "user", "content": user_input}
        ]
    )

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_speech(text):
    return client.audio.speech.create(model="tts-1", voice="alloy", input=text)

@app.route('/voice-agent', methods=['POST'])
def voice_agent():
    try:
        # Check if audio file is in the request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Transcribe audio using OpenAI Whisper
        transcription = transcribe_audio(temp_audio_path)
        user_input = transcription.text

        # Generate response using OpenAI's chat model
        chat_response = generate_chat_response(user_input)
        text_response = chat_response.choices[0].message.content

        # Convert response to speech
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_speech:
            speech_response = generate_speech(text_response)
            speech_response.stream_to_file(temp_speech.name)
            temp_speech_path = temp_speech.name

        # Read audio file and encode to base64
        with open(temp_speech_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Clean up temporary files
        os.remove(temp_audio_path)
        os.remove(temp_speech_path)

        return jsonify({
            "text_response": text_response,
            "audio_base64": audio_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
