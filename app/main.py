import logging
import os
from typing import Dict, List, Optional

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.stub import Stub

from llm_client import OllamaLLMClient        
from memory_manager import MemoryManager      
from pipeline_logic import AIProcessingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


configurations: Dict[str, ConfigClass] = dict()

# LLM Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Ensure this model name is correct and pulled in your Ollama
OLLAMA_MODEL_NAME = "gemma3:1b" 
LLM_REQUEST_TIMEOUT = 120 

# MemoryManager Configuration
# Get the directory of the current main.py script
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "datastore", "production_main_memory.sqlite")

# AIProcessingPipeline Output Configuration
PIPELINE_OUTPUT_BASE_DIR = os.path.join(APP_DIR, "datastore", "production_main_assets")

try:
    llm_client_instance = OllamaLLMClient(
        model_name=OLLAMA_MODEL_NAME,
        ollama_base_url=OLLAMA_BASE_URL,
        request_timeout=LLM_REQUEST_TIMEOUT
    )
    memory_manager_instance = MemoryManager(db_path=DB_PATH)
    logger.info("Global LLM client and Memory Manager initialized for main.py.")
except Exception as e_init:
    logger.error(f"Fatal error initializing global components: {e_init}", exc_info=True)
    llm_client_instance = None
    memory_manager_instance = None  


def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application (not used in this implementation).
    """
    global configurations 

    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf

def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """

    logger.info("--- Openfabric App Execution Started ---")
    
    request: InputClass = model.request
    response: OutputClass = model.response

    user_prompt_from_input: Optional[str] = getattr(request, 'prompt', None)
    if user_prompt_from_input:
        response.message = f"Processing prompt: {user_prompt_from_input}" # Initial message
    else:
        response.message = "ERROR: No prompt provided in the request."
        logger.warning("Request received with no prompt.")
        return

    logger.info(f"Received prompt: '{user_prompt_from_input}'")

    if not llm_client_instance or not memory_manager_instance:
        error_msg = "FATAL: Core components (LLM Client or Memory Manager) failed to initialize during app startup. Cannot process request."
        logger.error(error_msg)
        response.message = error_msg 
        return

    user_id_for_config_and_pipeline = 'super-user' 
    user_config: Optional[ConfigClass] = configurations.get(user_id_for_config_and_pipeline, None)

    if not user_config:
        error_msg = f"ERROR: Configuration for '{user_id_for_config_and_pipeline}' not found. Cannot retrieve App IDs."
        logger.error(error_msg)
        response.message = error_msg 
        return
    
    logger.info(f"Using configuration for '{user_id_for_config_and_pipeline}': {user_config}")

    app_ids_hostnames: List[str] = user_config.app_ids if user_config.app_ids else []

    if not app_ids_hostnames or len(app_ids_hostnames) < 2:
        error_msg = "ERROR: Required service App IDs (hostnames) are missing or insufficient in configuration."
        logger.error(f"{error_msg} Provided: {app_ids_hostnames}")
        response.message = error_msg 
        return

    of_stub = Stub(app_ids_hostnames) 
    logger.info("core.stub.Stub initialized successfully within execute() using configured App IDs.")

    text_to_image_app_id_from_config = app_ids_hostnames[0]
    image_to_3d_app_id_from_config = app_ids_hostnames[1]
    logger.info(f"Text-to-Image App ID from config: {text_to_image_app_id_from_config}")
    logger.info(f"Image-to-3D App ID from config: {image_to_3d_app_id_from_config}")

    try:
        pipeline = AIProcessingPipeline(
            stub=of_stub,
            llm_client=llm_client_instance,
            memory_manager=memory_manager_instance,
            text_to_image_app_id=text_to_image_app_id_from_config,
            image_to_3d_app_id=image_to_3d_app_id_from_config,
            output_base_dir=PIPELINE_OUTPUT_BASE_DIR
        )
        logger.info("AIProcessingPipeline instance created within execute().")

        pipeline_result = pipeline.run(user_prompt=user_prompt_from_input, user_id=user_id_for_config_and_pipeline)

        if pipeline_result.get("error"):
            response.message = (
                f"Pipeline Error: {pipeline_result['error']}\n"
                f"Original Prompt: {pipeline_result.get('original_prompt', 'N/A')}"
            )
            logger.error(f"Pipeline execution failed: {pipeline_result['error']}")
        else:
            response.message = (
                f"Success! Creation ID: {pipeline_result.get('creation_id', 'N/A')}\n"
                f"Original Prompt: {pipeline_result.get('original_prompt', 'N/A')}\n"
                f"Enhanced Prompt: {pipeline_result.get('enhanced_prompt', 'N/A')}\n"
                f"Image saved to: {pipeline_result.get('image_path', 'N/A')}\n"
                f"3D Model saved to: {pipeline_result.get('model_path', 'N/A')}\n"
                f"Video saved to: {pipeline_result.get('video_path', 'N/A')}\n"
                f"Tags: {', '.join(pipeline_result.get('tags', []))}"
            )
            logger.info("Pipeline execution successful.")

    except Exception as e:
        error_msg = f"An unexpected error occurred during pipeline processing: {e}"
        logger.error(error_msg, exc_info=True)
        response.message = f"FATAL ERROR: {error_msg}" # Overwrite

    logger.info(f"Final response message being sent (first 200 chars): {response.message[:200]}...")
    logger.info("--- Openfabric App Execution Finished ---")