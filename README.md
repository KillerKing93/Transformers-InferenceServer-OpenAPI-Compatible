# NodeJS LLamaCPP Inference Server (OpenAPI Compatible)

This project provides a NodeJS-based inference server for the Qwen3 model, compatible with the OpenAI API standard. It utilizes the `llama.cpp` library for efficient model inference.

## Features

*   OpenAI API compatible endpoint for model inference.
*   Powered by `llama.cpp` for high-performance inference.
*   Supports the Qwen3 model.

## Prerequisites

*   Node.js
*   `node-llama-cpp`

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/KillerKing93/NodeJS-LLamaCPP-InferenceServer-OpenAPI-Compatible.git
    ```
2.  Install the dependencies:
    ```bash
    npm install
    ```

## Usage

1.  Start the server:
    ```bash
    npm start
    ```
2.  The server will be running at `http://localhost:3000`.

## API Endpoints

The server exposes the following OpenAI API compatible endpoint:

*   `POST /v1/chat/completions`

For detailed information about the request and response format, please refer to the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/chat/create).
