import os
import json
import logging
from http.server import SimpleHTTPRequestHandler, HTTPServer
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("CoPilotWebServer")

# Add current directory to path and import agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main

# Instantiate DecisionAgent
logger.info("Initializing multi-agent flight operations copilot backend...")
agent = main.OpsCopilotAgent()
logger.info("Backend components successfully initialized.")

class CoPilotRequestHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Serve all static files from the 'web' folder
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
        path_suffix = path.lstrip("/")
        # Default to index.html if empty
        if not path_suffix:
            path_suffix = "index.html"
        return os.path.join(base_dir, path_suffix)

    def do_POST(self):
        if self.path == "/api/query":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(post_data)
                query = data.get("query", "")
            except Exception:
                query = post_data
            
            logger.info(f"Received query: {query}")
            
            try:
                # Classify intent
                intent = agent.classify_intent(query)
                agent_name = "DecisionAgent"
                if intent == "WEATHER":
                    agent_name = "WeatherAgent"
                elif intent == "RAG":
                    agent_name = "RAGAgent"
                elif intent == "CALCULATOR":
                    agent_name = "PerformanceAgent"
                
                # Execute agent
                response = agent.run(query)
                
                out_data = {
                    "status": "success",
                    "intent": intent,
                    "agent": agent_name,
                    "response": response
                }
            except Exception as e:
                logger.exception("Error executing agent query")
                out_data = {
                    "status": "error",
                    "message": str(e)
                }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(json.dumps(out_data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        # Handle pre-flight requests
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, CoPilotRequestHandler)
    logger.info(f"Flight Operations GUI Server running at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        httpd.server_close()

if __name__ == "__main__":
    port_arg = 8000
    if len(sys.argv) > 1:
        try:
            port_arg = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port_arg)
