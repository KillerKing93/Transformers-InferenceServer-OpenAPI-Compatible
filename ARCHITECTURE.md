# Architecture

This document outlines the architecture of the NodeJS LLamaCPP Inference Server.

## Components

*   **`index.js`**: The main entry point of the application. It initializes the Express server, loads the model, and defines the API endpoints.
*   **Express Server**: A web server framework for NodeJS that handles incoming HTTP requests.
*   **`node-llama-cpp`**: A NodeJS binding for the `llama.cpp` library. It provides the core functionality for loading and running the Qwen3 model.
*   **Qwen3 Model**: The large language model used for inference.

## Workflow

1.  The server starts and loads the Qwen3 model into memory using `node-llama-cpp`.
2.  A client sends a POST request to the `/v1/chat/completions` endpoint with a prompt.
3.  The Express server receives the request and passes it to the inference handler.
4.  The inference handler uses `node-llama-cpp` to run the model with the given prompt.
5.  The model generates a response.
6.  The server sends the response back to the client in the OpenAI API compatible format.
