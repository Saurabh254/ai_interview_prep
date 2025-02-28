# AI Interviewer

A command-line tool that simulates job interviews using Ollama/Llama 3.2 to help candidates practice and prepare for job interviews through their terminal.

## Overview

AI Interviewer is a Python script that provides an interactive interview experience directly in your terminal. Using local LLM models through Ollama (specifically Llama 3.2), this tool enables realistic interview practice without requiring an internet connection or sending your data to external services.

## Features

- **Terminal-based Interface**: Clean, simple interaction through your command line
- **Local LLM Integration**: Powered by Ollama running Llama 3.2 locally on your machine
- **Customizable Interviews**: Configure interview scenarios for different job roles
- **Offline Capability**: No internet required once model is downloaded
- **Interview Transcripts**: Save your practice sessions for later review
- **Minimal Resource Usage**: Optimized to run efficiently on standard hardware

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) installed with Llama 3.2 model
- 8GB+ RAM recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/saurabh254/ai_interview_prep.git

# Navigate to the project directory
cd ai_interview_prep

# Install dependencies
uv sync

# Ensure Ollama is installed and Llama 3.2 model is pulled
ollama pull llama3.2
```

## Usage

```bash
# Basic usage
uv python interview_prep.py

```


## How It Works

1. The script initializes a connection to the local Ollama instance
2. It loads a series of interview questions based on the selected job role
3. Questions are presented one by one in the terminal
4. User responses are captured and analyzed by the LLM
5. The AI interviewer responds with follow-up questions or moves to the next topic
6. At the end, a brief summary of the interview is provided
