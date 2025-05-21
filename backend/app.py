import logging
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from modules.chat_module import ChatHandler, conversational_prompt
from modules.rag_model import DocumentRAG, upload_file, process_rag_query
from modules.web_crawler import EphemeralRAG
from modules import image_scanner
from tempfile import TemporaryDirectory
import asyncio
from PIL import Image
import uuid  # Import uuid for unique filenames

# MODIFIED IMPORTS
import subprocess
import signal
import threading

# Import live_video.py as a module (No longer needed here)
# from modules import live_video # Removed unnecessary import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, static_folder='../frontend')
CORS(app)  # Enable CORS for all routes

# Initialize the chat handler
chat_handler = ChatHandler()

# Initialize the RAG model
rag_model = DocumentRAG()

# Initialize the Web Crawler
web_crawler = None

# Initialize mode tracker
current_mode = 'chat'

# NEW GLOBAL VARIABLES FOR LIVE CHAT
live_video_process = None
live_video_lock = threading.Lock()

# NEW GLOBAL VARIABLES FOR VOICE CHAT
voice_chat_process = None
voice_chat_lock = threading.Lock()

# Serve Frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.isfile(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Add this new route to serve detected images
@app.route('/detected_images/<filename>')
def serve_detected_image(filename):
    return send_from_directory(image_scanner.output_folder, filename)

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handles general chat requests."""
    global current_mode
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logging.warning("Invalid request: No message received")
            return jsonify({"error": "Invalid request: No message provided"}), 400

        user_message = data['message']
        response = chat_handler.handle_conversation(conversational_prompt, user_message)

        if response is None:
             logging.error("Chat response generation failed")
             return jsonify({"error": "Failed to generate a response"}), 500
        return jsonify({"response": response, "success":True})

    except Exception as e:
        logging.error(f"Unexpected error during chat handling: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/upload', methods=['POST'])
def handle_upload():
    """Handles file uploads for RAG."""
    global current_mode
    try:
        if 'file' not in request.files:
             logging.warning("Invalid request: No file received")
             return jsonify({"error": "No file part", "success":False}), 400

        file = request.files['file']

        if file.filename == '':
           logging.warning("Invalid request: No file selected")
           return jsonify({'error': 'No selected file', "success":False}), 400

        # Create a TemporaryDirectory instance
        temp_dir = TemporaryDirectory()
        upload_folder = temp_dir.name if not rag_model.persist_data else "data/uploads"
        file_path, filename = upload_file(file, upload_folder)

        if not file_path:
            temp_dir.cleanup()
            return jsonify({"error": "File saving failed", "success":False}), 500

        content = rag_model.load_document(file_path)

        if not content:
            temp_dir.cleanup()
            return jsonify({"error": "Document loading failed", "success":False}), 500

        documents = rag_model.process_content(content, filename)

        if not documents:
             temp_dir.cleanup()
             return jsonify({"error": "Document processing failed", "success":False}), 500

        rag_model.create_vector_store(documents)
        temp_dir.cleanup()
        return jsonify({'filename': filename, "success":True}), 200

    except Exception as e:
         logging.error(f"Error during file upload: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/rag_query', methods=['POST'])
def handle_rag_query():
    """Handles RAG queries."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            logging.warning("Invalid request: No question received")
            return jsonify({"error": "Invalid request: No question provided", "success":False}), 400

        question = data['question']
        response = process_rag_query(question, rag_model)

        if response is None:
            logging.error("Error getting the answer from RAG model")
            return jsonify({"error": "Failed to generate a response", "success":False}), 500

        return jsonify({'response':response['result'], 'document_name': response['source_documents'][0].metadata['source'], "success":True}), 200

    except Exception as e:
         logging.error(f"Error during rag query: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/cleanup_rag', methods=['POST'])
def handle_cleanup_rag():
    """Handles the cleanup of the rag model."""
    try:
        rag_model.cleanup()
        return jsonify({'message': 'RAG mode cleanup successfully', "success":True}), 200
    except Exception as e:
         logging.error(f"Error during rag cleanup: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/crawl_website', methods=['POST'])
async def handle_crawl_website():
    """Handles website crawling requests."""
    global current_mode
    global web_crawler
    try:
         data = request.get_json()
         if not data or 'url' not in data:
            logging.warning("Invalid request: No URL received")
            return jsonify({"error": "Invalid request: No URL provided", "success":False}), 400

         url = data['url']
         if web_crawler:
             web_crawler.cleanup()
         web_crawler = EphemeralRAG()

         content = await web_crawler.crawl_website(url)

         if not content:
            return jsonify({"error": "Website crawling failed", "success":False}), 500

         documents = web_crawler.process_content(content, url)
         web_crawler.create_vector_store(documents)

         return jsonify({'message': 'Website crawled successfully', "success":True}), 200

    except Exception as e:
         logging.error(f"Error during web crawling: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/web_query', methods=['POST'])
def handle_web_query():
    """Handles web crawl queries."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            logging.warning("Invalid request: No question received")
            return jsonify({"error": "Invalid request: No question provided", "success":False}), 400

        question = data['question']
        if web_crawler is None:
            logging.error("Web Crawler is None")
            return jsonify({"error": "Web Crawler not initialized", "success":False}), 500

        response = web_crawler.setup_qa_chain().invoke({"query":question})

        if response is None:
            logging.error("Error getting the answer from Web Crawler model")
            return jsonify({"error": "Failed to generate a response", "success":False}), 500

        return jsonify({'response':response['result'], 'document_name': response['source_documents'][0].metadata['source'], "success":True}), 200

    except Exception as e:
         logging.error(f"Error during web query: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/cleanup_web', methods=['POST'])
def handle_cleanup_web():
    """Handles the cleanup of the web crawler."""
    try:
         if web_crawler:
             web_crawler.cleanup()
         return jsonify({'message': 'Web crawler mode cleanup successfully', "success":True}), 200
    except Exception as e:
         logging.error(f"Error during web cleanup: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/upload_image', methods=['POST'])
def handle_upload_image():
     """Handles image uploads for image processing."""
     global current_mode
     try:
          if 'image' not in request.files:
               logging.warning("Invalid request: No image received")
               return jsonify({"error": "No image part", "success":False}), 400
          image_file = request.files['image']
          if image_file.filename == '':
               logging.warning("Invalid request: No image selected")
               return jsonify({'error': 'No selected image', "success":False}), 400

          # Create temporary directory
          temp_dir = TemporaryDirectory()
          # Generate a unique filename
          filename = os.path.join(temp_dir.name, f"{uuid.uuid4()}{os.path.splitext(image_file.filename)[1]}")
          # Save the uploaded image to the temporary directory
          image_file.save(filename)

          # Process with saved file path
          with Image.open(filename) as image: # Open image using the saved path
              caption_result = image_scanner.get_image_caption(image, filename) # Pass image and server-side file path

          # Store temporary directory for cleanup - pass temp_dir to image_scanner
          image_scanner.set_temp_dir(temp_dir) # Pass temp_dir instance for cleanup

          if not caption_result['success']:
               logging.error(f"Error during image caption: {caption_result['error']}")
               return jsonify({'error': 'Error generating image caption', "success":False}), 500

          current_mode = 'image_chat'
          return jsonify({'response': "I am looking at the image you sent me . I have to say it is looking interesting" , 'caption': caption_result['response'], "success":True}), 200

     except Exception as e:
          logging.error(f"Error during image upload: {e}")
          return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/image_query', methods=['POST'])
def handle_image_query():
     """Handles image processing queries."""
     try:
          # Directly get question from request.form - expecting FormData
          question = request.form.get('question')
          if not question:
               logging.warning("Invalid request: No question received in FormData")
               return jsonify({"error": "Invalid request: No question provided", "success":False}), 400

          if "detect" in question.lower():
            object_to_detect = question.split("detect")[1].strip()
            detection_result = image_scanner.detect_objects(object_to_detect)
            if not detection_result["success"]:
                logging.error(f"Error during object detection: {detection_result['error']}")
                return jsonify({"error": detection_result['response'], "success": False})
            return jsonify(detection_result)

          elif "point" in question.lower():
            query = question.split("point")[1].strip()
            point_result = image_scanner.point_at_object(query)
            if not point_result["success"]:
                logging.error(f"Error during object pointing: {point_result['error']}")
                return jsonify({"error": point_result['response'], "success": False})
            return jsonify(point_result)

          else:
             answer_result = image_scanner.answer_image_question(question)
             if not answer_result["success"]:
                logging.error(f"Error during image question answering: {answer_result['error']}")
                return jsonify({"error": answer_result["error"], "success":False}), 500
             return jsonify(answer_result)

     except Exception as e:
          logging.error(f"Error during image processing: {e}")
          return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/cleanup_image', methods=['POST'])
def handle_cleanup_image():
    """Handles the cleanup of the image resources."""
    global current_mode
    try:
        cleanup_result = image_scanner.cleanup_image_resources()
        if not cleanup_result["success"]:
            logging.error(f"Error during image cleanup: {cleanup_result['error']}")
            return jsonify({"error": "Error during image cleanup", "success":False}), 500
        current_mode = 'chat'
        return jsonify(cleanup_result), 200
    except Exception as e:
         logging.error(f"Error during image cleanup: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

# NEW LIVE CHAT ENDPOINTS
@app.route('/start_live_chat', methods=['POST'])
def start_live_chat():
    """Starts the live chat process."""
    global live_video_process
    global voice_chat_process

    with live_video_lock:  # Use live_video_lock for live video
        if live_video_process is not None:
            logging.warning("Live chat already running")
            return jsonify({'error': 'Live chat already running', "success":False}), 400

        if voice_chat_process is not None:
            with voice_chat_lock: # Use voice_chat_lock before accessing voice_chat_process
                try:
                     os.kill(voice_chat_process.pid, signal.SIGTERM)
                     voice_chat_process.wait()
                     voice_chat_process = None
                     logging.info("Voice chat stopped successfully")
                except Exception as e:
                     logging.error(f"Failed to stop voice chat: {e}")

        try:
            live_video_process = subprocess.Popen(['python', 'modules/live_video.py'])
            logging.info("Live chat started successfully")
            return jsonify({'message': 'Live chat started', "success":True}), 200
        except Exception as e:
            logging.error(f"Failed to start live chat: {e}")
            return jsonify({'error': f'Failed to start live chat: {e}', "success":False}), 500

@app.route('/stop_live_chat', methods=['POST'])
def stop_live_chat():
    """Stops the live chat process."""
    global live_video_process

    with live_video_lock: # Use live_video_lock for live video
        if live_video_process is None:
            logging.warning("Live chat is not running")
            return jsonify({'error': 'Live chat not running', "success":False}), 400

        try:
            os.kill(live_video_process.pid, signal.SIGTERM)
            live_video_process.wait()
            live_video_process = None
            logging.info("Live chat stopped successfully")
            return jsonify({'message': 'Live chat stopped', "success":True}), 200
        except Exception as e:
             logging.error(f"Failed to stop live chat: {e}")
             return jsonify({'error': f'Failed to stop live chat: {e}', "success":False}), 500

# NEW VOICE CHAT ENDPOINTS
@app.route('/start_voice_chat', methods=['POST'])
def start_voice_chat():
    """Starts the voice chat process."""
    global voice_chat_process
    global live_video_process

    with voice_chat_lock:  # Use voice_chat_lock for voice chat
        if voice_chat_process is not None:
            logging.warning("Voice chat already running")
            return jsonify({'error': 'Voice chat already running', "success": False}), 400

        if live_video_process is not None:
             with live_video_lock: # Use live_video_lock before accessing live_video_process
                try:
                     os.kill(live_video_process.pid, signal.SIGTERM)
                     live_video_process.wait()
                     live_video_process = None
                     logging.info("Live video stopped successfully")
                except Exception as e:
                     logging.error(f"Failed to stop live video: {e}")

        try:
            voice_chat_process = subprocess.Popen(['python', 'modules/voice_chat.py'])
            logging.info("Voice chat started successfully")
            return jsonify({'message': 'Voice chat started', "success": True}), 200
        except Exception as e:
            logging.error(f"Failed to start voice chat: {e}")
            return jsonify({'error': f'Failed to start voice chat: {e}', "success": False}), 500

@app.route('/stop_voice_chat', methods=['POST'])
def stop_voice_chat():
    """Stops the voice chat process."""
    global voice_chat_process

    with voice_chat_lock: # Use voice_chat_lock for voice chat
        if voice_chat_process is None:
            logging.warning("Voice chat is not running")
            return jsonify({'error': 'Voice chat not running', "success": False}), 400

        try:
            os.kill(voice_chat_process.pid, signal.SIGTERM)
            voice_chat_process.wait()
            voice_chat_process = None
            logging.info("Voice chat stopped successfully")
            return jsonify({'message': 'Voice chat stopped', "success": True}), 200
        except Exception as e:
            logging.error(f"Failed to stop voice chat: {e}")
            return jsonify({'error': f'Failed to stop voice chat: {e}', "success": False}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
