#!/usr/bin/env python3
"""Mock embedding server for testing v10r integration."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BGE-M3 produces 1024-dimensional vectors
EMBEDDING_DIM = 1024

def generate_mock_embedding(text):
    """Generate a deterministic mock embedding based on text."""
    # Create a simple deterministic embedding
    base_value = sum(ord(c) for c in text) / (len(text) + 1)
    embedding = []
    for i in range(EMBEDDING_DIM):
        value = (base_value + i * 0.001) % 2.0 - 1.0  # Keep values in [-1, 1]
        embedding.append(round(value, 6))
    return embedding

class MockEmbeddingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock embedding API."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "service": "mock-infinity"}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404, "Not found")
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/v1/embeddings':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                
                # Handle both single string and array of strings
                input_texts = request_data.get('input', [])
                if isinstance(input_texts, str):
                    input_texts = [input_texts]
                
                # Generate embeddings for each input
                embeddings_data = []
                for idx, text in enumerate(input_texts):
                    embedding = generate_mock_embedding(text)
                    embeddings_data.append({
                        "object": "embedding",
                        "index": idx,
                        "embedding": embedding
                    })
                
                # Create response
                response = {
                    "object": "list",
                    "data": embeddings_data,
                    "model": request_data.get('model', 'text-embedding-ada-002'),
                    "usage": {
                        "prompt_tokens": len(input_texts) * 5,
                        "total_tokens": len(input_texts) * 5
                    }
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
                logger.info(f"Generated {len(embeddings_data)} embeddings")
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                self.send_error(400, str(e))
        else:
            self.send_error(404, "Not found")
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr."""
        logger.info(f"{self.client_address[0]} - {format % args}")

def run_server(port=80):
    """Run the mock embedding server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, MockEmbeddingHandler)
    logger.info(f"Mock embedding server listening on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server() 