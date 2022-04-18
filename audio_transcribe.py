#export GOOGLE_APPLICATION_CREDENTIALS="capstone-342405-142e85c380f0.json"

from os.path import exists

def audio_transcribe(gcs_uri, output_file):
    #Asynchronously transcribes the audio file specified by the google cloud service uri
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri = gcs_uri)
    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz = 32000,
        language_code = "en-US",
        audio_channel_count = 1,
    )

    operation = client.long_running_recognize(config = config, audio = audio)

    print("Waiting for operation to complete...") #Maybe delete
    response = operation.result(timeout=10000)
    if exists(output_file):
        os.remove(output_file)
    f = open(output_file, "a")
    # Get the transcripts for each audio segment
    for result in response.results:
        if result.alternatives[0].confidence > 0.5:
            f.write(result.alternatives[0].transcript + "\n")
    f.close()
audio_file = 'gs://audio-bucket-22/ch1.flac'
output_file = f'ch1_11apr.txt'
audio_transcribe(audio_file, output_file)
