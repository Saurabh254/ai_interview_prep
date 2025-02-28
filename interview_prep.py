import os
import json
import time
import wave
import tempfile
import threading
import pyaudio
import numpy as np
import ollama
from datetime import datetime
import speech_recognition as sr
import pyttsx3


class AIInterviewer:
    def __init__(
        self, model="llama3:latest", num_questions=5, interview_topic="general"
    ):
        self.model = model
        self.num_questions = num_questions
        self.interview_topic = interview_topic
        self.recognizer = sr.Recognizer()
        self.answers = []
        self.questions = []
        self.ratings = []

        # Audio settings - simplified approach
        self.audio = pyaudio.PyAudio()
        self.speaker = pyttsx3.init()
        self.stream = None
        self.frames = []
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16

    def generate_questions(self):
        """Generate interview questions using Ollama"""
        print(
            f"Generating {self.num_questions} questions for {self.interview_topic} interview..."
        )

        prompt = f"""
        You are an expert interviewer. Generate {self.num_questions} challenging but fair interview questions
        for a {self.interview_topic} interview. The questions should assess both technical knowledge and
        problem-solving abilities. Return ONLY the questions as a JSON array.
        """

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional interviewer. Provide only the requested output format.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        try:
            # Try to extract JSON from response
            content = response["message"]["content"]
            # Look for JSON in the content
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                questions = json.loads(json_str)
            else:
                # Fallback if no proper JSON format is found
                questions = [q.strip() for q in content.split("\n") if q.strip()]
                questions = questions[: self.num_questions]  # Limit to requested number
        except Exception as e:
            print(f"Error parsing questions: {e}")
            # Fallback questions if JSON parsing fails
            questions = [
                f"Tell me about your experience with {self.interview_topic}.",
                f"What challenges have you faced in {self.interview_topic} and how did you overcome them?",
                f"Describe a project where you used {self.interview_topic} skills.",
                f"How do you stay updated with the latest developments in {self.interview_topic}?",
                f"Where do you see the field of {self.interview_topic} heading in the next 5 years?",
            ][: self.num_questions]

        self.questions = questions
        return questions

    def start_recording(self):
        """Start recording audio using PyAudio - more reliable approach"""
        self.frames = []

        # Check available input devices
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount")

        # List available input devices
        print("\nAvailable input devices:")
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get("maxInputChannels") > 0:
                print(f"Device {i}: {device_info.get('name')}")

        # Ask user to select input device
        try:
            device_index = int(input("Select input device index (default: 0): ") or "0")
        except ValueError:
            device_index = 0

        # Open stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )

        print("\n🎤 Recording started! Speak now...")
        self.stream.start_stream()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio recording"""
        self.frames.append(in_data)
        return (None, pyaudio.paContinue)

    def stop_recording(self):
        """Stop recording and save the audio"""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            print("Recording stopped")

            if not self.frames:
                print("No audio data recorded")
                return None

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name

            # Save audio to WAV file
            with wave.open(temp_filename, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(self.frames))

            return temp_filename
        return None

    def transcribe_audio(self, audio_file):
        """Transcribe audio using multiple speech recognition engines for better accuracy with Indian accents"""
        if not audio_file or not os.path.exists(audio_file):
            return "No audio recorded or file not found."

        try:
            # First try with Google's speech recognition (good for Indian accents)
            with sr.AudioFile(audio_file) as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio_data = self.recognizer.record(source)

            try:
                text = self.recognizer.recognize_google(audio_data)
                print("Transcription successful with Google Speech Recognition")
                return text
            except sr.UnknownValueError:
                print(
                    "Google Speech Recognition could not understand audio, trying alternative..."
                )
            except sr.RequestError:
                print(
                    "Google Speech Recognition service unavailable, trying alternative..."
                )

            # Fallback to Sphinx (offline) if Google fails
            try:
                text = self.recognizer.recognize_sphinx(audio_data)
                print("Transcription successful with Sphinx (offline)")
                return text
            except sr.UnknownValueError:
                return "Sorry, I couldn't understand the audio."
            except sr.RequestError:
                return "Speech recognition systems are unavailable."

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return f"Error transcribing audio: {e}"
        finally:
            # Clean up the temporary file
            try:
                os.unlink(audio_file)
            except:
                pass

    def rate_answer(self, question, answer):
        """Rate the interview answer using Ollama"""
        prompt = f"""
        As an expert interviewer, rate the following answer to the question:

        Question: "{question}"

        Answer: "{answer}"

        Provide a score from 1 to 10 and brief feedback on the answer's quality.
        Format your response as a JSON with two fields: 'score' (number) and 'feedback' (string).
        """

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert interviewer evaluating responses. Return only the requested JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        try:
            # Try to extract JSON from response
            content = response["message"]["content"]

            # Look for JSON object in the content
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                rating = json.loads(json_str)
            else:
                # Fallback if no proper JSON format is found
                rating = {
                    "score": 5,  # Default middle score
                    "feedback": "The answer was adequate but the rating system couldn't parse detailed feedback.",
                }
        except Exception as e:
            print(f"Error parsing rating: {e}")
            rating = {"score": 5, "feedback": "Error generating detailed feedback."}

        return rating

    def save_interview_results(self):
        """Save interview results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_results_{timestamp}.json"

        results = {
            "interview_topic": self.interview_topic,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "questions": [],
            "overall_score": (
                round(sum(r["score"] for r in self.ratings) / len(self.ratings), 1)
                if self.ratings
                else 0
            ),
        }

        for i, (question, answer, rating) in enumerate(
            zip(self.questions, self.answers, self.ratings)
        ):
            results["questions"].append(
                {
                    "number": i + 1,
                    "question": question,
                    "answer": answer,
                    "score": rating["score"],
                    "feedback": rating["feedback"],
                }
            )

        with open(filename, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Interview results saved to {filename}")
        return filename

    def check_audio_levels(self, duration=3):
        """Check microphone levels to ensure it's working properly"""
        print(f"\nTesting microphone for {duration} seconds. Please speak...")

        # Check available input devices
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount")

        # List available input devices
        print("Available input devices:")
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get("maxInputChannels") > 0:
                print(f"Device {i}: {device_info.get('name')}")

        # Ask user to select input device
        try:
            device_index = int(input("Select input device index (default: 0): ") or "0")
        except ValueError:
            device_index = 0

        # Test recording
        test_frames = []

        def callback(in_data, frame_count, time_info, status):
            test_frames.append(in_data)
            return (None, pyaudio.paContinue)

        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=callback,
        )

        stream.start_stream()

        # Show a simple level meter
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                if test_frames:
                    # Convert the most recent frame to a numpy array
                    data = np.frombuffer(test_frames[-1], dtype=np.int16)
                    # Calculate volume
                    volume_norm = np.linalg.norm(data) / 32767
                    # Print a simple level meter
                    meter = "#" * int(volume_norm * 50)
                    print(
                        f"\rMicrophone Level: {meter.ljust(50)} {int(volume_norm * 100)}%",
                        end="",
                    )
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            print("\n")
            stream.stop_stream()
            stream.close()

        # Check if any audio was recorded
        if not test_frames:
            print("⚠️ No audio detected! Please check your microphone settings.")
            print("Try selecting a different input device.")
            return False
        else:
            # Write a test file to verify
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                test_filename = temp_file.name

            with wave.open(test_filename, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(test_frames))

            # Test transcription
            try:
                with sr.AudioFile(test_filename) as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    print(f'Microphone test successful! Transcription: "{text}"')
                    os.unlink(test_filename)
                    return True
            except Exception as e:
                print(f"⚠️ Transcription test failed: {e}")
                print(
                    "You can continue, but the speech recognition might not work correctly."
                )
                os.unlink(test_filename)
                return False

    def conduct_interview(self):
        """Conduct the full interview process"""
        print("\n" + "=" * 50)
        print("🎙️ AI Interview Assistant".center(50))
        print("=" * 50)
        print(f"Topic: {self.interview_topic}")
        print(f"Number of questions: {self.num_questions}")
        print("Press Ctrl+C at any time to end the interview")
        print("=" * 50)

        # Test audio setup
        print("\nLet's test your microphone before starting...")
        mic_working = self.check_audio_levels()

        if not mic_working:
            use_text_input = (
                input(
                    "\nWould you like to continue with text input only? (y/n): "
                ).lower()
                == "y"
            )
            if not use_text_input:
                print(
                    "Exiting interview. Please check your microphone settings and try again."
                )
                return
        else:
            use_text_input = False

        # Generate questions
        self.generate_questions()

        try:
            for i, question in enumerate(self.questions):
                print(f"\n🎤 Question {i+1}/{len(self.questions)}:")
                print(f"> {question}")
                self.speaker.say(question)
                self.speaker.runAndWait()

                if use_text_input:
                    # Text input mode
                    answer = input("\nYour answer: ")
                else:
                    # Voice input mode
                    input("\nPress Enter when ready to answer...")

                    # Record answer
                    self.start_recording()
                    input("Recording... Press Enter when finished answering.")
                    audio_file = self.stop_recording()

                    # Transcribe answer
                    print("\nProcessing your answer...")
                    answer = self.transcribe_audio(audio_file)
                    print(f'I heard: "{answer}"')

                    # Confirm or edit transcription
                    confirm = input("Is this transcription correct? (y/n): ").lower()
                    if confirm != "y":
                        answer = input("Please type your answer manually: ")

                self.answers.append(answer)

                # Rate the answer
                print("\nRating your answer...")
                rating = self.rate_answer(question, answer)
                self.ratings.append(rating)

                print(f"Score: {rating['score']}/10")
                print(f"Feedback: {rating['feedback']}")
                print("\n" + "-" * 50)

            # Save results
            results_file = self.save_interview_results()

            # Show summary
            print("\n" + "=" * 50)
            print("📊 Interview Summary".center(50))
            print("=" * 50)
            overall_score = (
                round(sum(r["score"] for r in self.ratings) / len(self.ratings), 1)
                if self.ratings
                else 0
            )
            print(f"Overall Score: {overall_score}/10")
            print(f"Detailed results saved to: {results_file}")
            print("=" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\nInterview ended early.")
            if self.answers:
                self.save_interview_results()

        except Exception as e:
            print(f"Error during interview: {e}")

        finally:
            # Clean up
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass

            try:
                self.audio.terminate()
            except:
                pass


if __name__ == "__main__":
    # Check if Ollama is available
    try:
        ollama.list()
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please make sure Ollama is installed and running.")
        exit(1)

    # Get interview parameters
    print("Welcome to the AI Interview Assistant!")

    # Check if llama3 is available
    models = ollama.list()
    available_models = [model["model"] for model in models.get("models", [])]

    default_model = "llama3:latest"
    if default_model not in available_models:
        print(f"Warning: {default_model} not found in available models.")
        print("Available models:", ", ".join(available_models))
        model = input(
            f"Enter model name (default: {available_models[0] if available_models else 'llama3'}): "
        )
        if not model and available_models:
            model = available_models[0]
        elif not model:
            model = "llama3.2"
    else:
        model = default_model

    topic = input("Enter interview topic (default: general): ") or "general"

    try:
        num_questions = int(input("Number of questions (default: 5): ") or "5")
    except ValueError:
        num_questions = 5

    # Create and run the interviewer
    interviewer = AIInterviewer(
        model=model, num_questions=num_questions, interview_topic=topic
    )
    interviewer.conduct_interview()
