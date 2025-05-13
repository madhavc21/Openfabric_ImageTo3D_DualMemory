import gradio as gr
import os
import logging
from typing import List, Dict, Optional, Tuple

try:
    from core.stub import Stub
    from llm_client import OllamaLLMClient
    from memory_manager import MemoryManager
    from pipeline_logic import AIProcessingPipeline
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}. Please ensure all custom modules are in the correct path.")

    Stub = None
    OllamaLLMClient = None
    MemoryManager = None
    AIProcessingPipeline = None

# Basic Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    STT_AVAILABLE = True
    stt_whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    logger.info("Faster-Whisper STT model (base) loaded on CPU.")
except ImportError:
    STT_AVAILABLE = False
    WhisperModel = None
    stt_whisper_model = None
    logger.warning("faster-whisper not installed. Voice input will be disabled.")
except Exception as e_stt: 
    STT_AVAILABLE = False
    WhisperModel = None
    stt_whisper_model = None
    logger.error(f"Failed to load Faster-Whisper STT model: {e_stt}", exc_info=True)

# Configurations for Gradio App
TEXT_TO_IMAGE_APP_ID = "c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network"
IMAGE_TO_3D_APP_ID = "f0b5f319156c4819b9827000b17e511a.node3.openfabric.network"

# IMPORTANT: Use a capable model for intent parsing & JSON.
# Examples: "llama3:8b-instruct", "deepseek-llm:7b-chat", "deepseek-coder:6.7b-instruct"
OLLAMA_MODEL_NAME = "gemma3:1b" # Best small scale model after testing
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_REQUEST_TIMEOUT = 180

# Paths for Gradio app's own memory and outputs
APP_DIR_GRADIO = os.path.dirname(os.path.abspath(__file__))
GRADIO_DATASTORE_DIR = os.path.join(APP_DIR_GRADIO, "gradio_datastore")
DB_PATH_GRADIO = os.path.join(GRADIO_DATASTORE_DIR, "gradio_memory.sqlite")
FAISS_INDEX_PATH_GRADIO = os.path.join(GRADIO_DATASTORE_DIR, "gradio_creations.faiss")
PIPELINE_OUTPUT_BASE_DIR_GRADIO = os.path.join(GRADIO_DATASTORE_DIR, "gradio_generated_assets")

os.makedirs(GRADIO_DATASTORE_DIR, exist_ok=True) 

pipeline_instance = None
if Stub and OllamaLLMClient and MemoryManager and AIProcessingPipeline:
    try:
        logger.info("Initializing Gradio App Core Components...")
        gradio_stub = Stub(app_ids=[TEXT_TO_IMAGE_APP_ID, IMAGE_TO_3D_APP_ID])
        gradio_llm_client = OllamaLLMClient(
            model_name=OLLAMA_MODEL_NAME,
            ollama_base_url=OLLAMA_BASE_URL,
            request_timeout=LLM_REQUEST_TIMEOUT
        )
        gradio_memory_manager = MemoryManager(
            db_path=DB_PATH_GRADIO,
            faiss_index_path=FAISS_INDEX_PATH_GRADIO
        )
        pipeline_instance = AIProcessingPipeline(
            stub=gradio_stub,
            llm_client=gradio_llm_client,
            memory_manager=gradio_memory_manager,
            text_to_image_app_id=TEXT_TO_IMAGE_APP_ID,
            image_to_3d_app_id=IMAGE_TO_3D_APP_ID,
            output_base_dir=PIPELINE_OUTPUT_BASE_DIR_GRADIO
        )
        logger.info("Gradio App Core Components Initialized Successfully.")
    except Exception as e_init:
        logger.error(f"FATAL: Error initializing Gradio App Core Components: {e_init}", exc_info=True)
else:
    logger.error("One or more core modules (Stub, LLMClient, MemoryManager, AIProcessingPipeline) not imported. Pipeline will not work.")



def transcribe_audio_input(audio_filepath: Optional[str]) -> str:
    if not STT_AVAILABLE or not stt_whisper_model:
        return "Speech-to-text model not available."
    if not audio_filepath:
        return "" 

    logger.info(f"Attempting to transcribe audio file: {audio_filepath}")
    try:
        segments, info = stt_whisper_model.transcribe(audio_filepath, beam_size=5)
        transcribed_text = "".join(segment.text for segment in segments).strip()
        logger.info(f"Transcription: '{transcribed_text}' (Language: {info.language}, Probability: {info.language_probability:.2f})")
        return transcribed_text
    except Exception as e:
        logger.error(f"Error during audio transcription: {e}", exc_info=True)
        return f"[Error transcribing: {str(e)}]"

def respond_to_chat(
    user_message: str, 
    chat_display_history: List[Tuple[Optional[str], Optional[str]]], 
    llm_conversation_history_state: List[Dict[str,str]] 
    ) -> Tuple[str, List[Tuple[Optional[str], Optional[str]]], List[Dict[str,str]], Optional[str], Optional[str]]:
    """
    Handles user input, calls the AI pipeline, and prepares outputs for Gradio.
    """
    if not pipeline_instance:
        error_msg = "ERROR: AI Pipeline is not initialized. Please check server logs."
        chat_display_history.append((user_message, error_msg))
        return "", chat_display_history, llm_conversation_history_state, None, None

    chat_display_history.append((user_message, None)) 
    

    current_llm_conversation_history = llm_conversation_history_state or []
    current_llm_conversation_history.append({"role": "user", "content": user_message})

    gradio_user_id = "gradio_default_user" 
    
    history_for_pipeline = current_llm_conversation_history[-7:-1] if len(current_llm_conversation_history) > 1 else None
    
    logger.info(f"Calling pipeline with user_message: '{user_message}' (Pipeline will fetch its own STM context)")

    pipeline_result = pipeline_instance.run(
        user_prompt=user_message, 
        user_id=gradio_user_id
    )

    bot_message = "Sorry, something went wrong."
    image_output_path = None
    model_output_path = None

    if pipeline_result:
        bot_message = pipeline_result.get("message_to_user", "Processed.")
        if pipeline_result.get("error"):
            bot_message += f"\nError: {pipeline_result.get('error')}"
        
        image_output_path = pipeline_result.get("image_path")
        model_output_path = pipeline_result.get("model_path")

        if not pipeline_result.get("error") and pipeline_result.get("enhanced_prompt"):
            logger.info(f"Final prompt used for generation: {pipeline_result.get('enhanced_prompt')}")
           
        current_llm_conversation_history.append({"role": "assistant", "content": bot_message})
    else:
        bot_message = "Pipeline did not return a result."
        current_llm_conversation_history.append({"role": "assistant", "content": bot_message})


    if chat_display_history and chat_display_history[-1][1] is None: 
        chat_display_history[-1] = (chat_display_history[-1][0], bot_message)
    else: 
        chat_display_history.append((None, bot_message))

    max_llm_history_turns = 10 
    if len(current_llm_conversation_history) > max_llm_history_turns:
        current_llm_conversation_history = current_llm_conversation_history[-max_llm_history_turns:]

    return "", chat_display_history, current_llm_conversation_history, image_output_path, model_output_path


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as demo:
    gr.Markdown("# 🚀 AI Creative Partner Chat")
    gr.Markdown("Chat with the AI to generate and modify images and 3D models. Try things like 'Create a red car', then 'make it blue', or 'Remember the car I made? Add wings to it.'")

    if not pipeline_instance:
        gr.Warning("AI Pipeline failed to initialize. Core functionality will be unavailable. Please check server logs.")

    llm_history_state = gr.State([]) 


    with gr.Row():
        with gr.Column(scale=2): # Chat column
            chatbot_display = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False)
            with gr.Accordion("🎤 Voice Input (Optional)", open=False, visible=STT_AVAILABLE): 
                mic_input = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", 
                    label="Record your prompt here",
                    show_label=False
                )
                transcribe_button = gr.Button("🗣️ Transcribe to Textbox Below", visible= STT_AVAILABLE)
            
            user_msg_textbox = gr.Textbox(
                label="Your Message (Type or use Voice Input):", 
                placeholder="e.g., A majestic dragon...",
                lines=3, 
                show_label=False
            )
            send_button = gr.Button("Send / Generate", variant="primary")
            clear_button = gr.ClearButton([user_msg_textbox, chatbot_display, llm_history_state]) # Add more components to clear

        with gr.Column(scale=1): 
            gr.Markdown("### Generated Image")
            output_image_display = gr.Image(label="Image Output", height=300, interactive=False)
            gr.Markdown("### Generated 3D Model")
            output_model_display = gr.Model3D(
                label="3D Model Output", 
                height=300, 
                interactive=True, 
                clear_color=[0.9, 0.9, 0.92, 1.0]
            )

    send_event_args = {
        "fn": respond_to_chat,
        "inputs": [user_msg_textbox, chatbot_display, llm_history_state],
        "outputs": [user_msg_textbox, chatbot_display, llm_history_state, output_image_display, output_model_display],
    }
    send_button.click(**send_event_args)
    user_msg_textbox.submit(**send_event_args)
    if STT_AVAILABLE:
        transcribe_button.click(
            fn=transcribe_audio_input,
            inputs=[mic_input],
            outputs=[user_msg_textbox] 
        )
    gr.Examples(
        examples=[
            ["A serene landscape with a crystal clear lake and distant mountains"],
            ["Generate a futuristic robot assistant"],
        ],
        inputs=[user_msg_textbox],
        label="Example Prompts"
    )

if __name__ == "__main__":
    if pipeline_instance:
        logger.info("Attempting to clear old short-term memory for 'gradio_default_user' before UI launch...")
        # Archive anything older than, say, 5 seconds to effectively clear most past session data
        pipeline_instance.memory_manager.archive_short_term_log(
            user_id="gradio_default_user", 
            older_than_seconds=5 
        )
        logger.info("Short-term memory pre-archive attempt complete.")
    if not pipeline_instance:
        print("WARNING: AI Pipeline (pipeline_instance) is None. The Gradio app might not function correctly.")
        print("This usually means an error occurred during the initialization of Stub, LLMClient, MemoryManager, or AIProcessingPipeline.")
        print("Please check the logs above this message for details on the initialization error.")
    
    demo.queue()
    demo.launch() 