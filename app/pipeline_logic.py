import os
import uuid
import base64
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import time

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    logging.warning("dateparser library not found. Date parsing for memory recall will be limited. "
                    "Install with: pip install dateparser")
    dateparser = None
    DATEPARSER_AVAILABLE = False


from core.stub import Stub
from llm_client import OllamaLLMClient
from memory_manager import MemoryManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIProcessingPipeline:
    def __init__(self,
                 stub: Stub,
                 llm_client: OllamaLLMClient,
                 memory_manager: MemoryManager,
                 text_to_image_app_id: str,
                 image_to_3d_app_id: str,
                 output_base_dir: str = "app/datastore/generated_assets",
                 short_term_memory_window_seconds: int = 3600, # Default 1 hour
                 archive_short_term_older_than_seconds: int = 3600 * 24 # Default 1 day
                 ):
        
        self.stub = stub
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.text_to_image_app_id = text_to_image_app_id
        self.image_to_3d_app_id = image_to_3d_app_id
        self.output_base_dir = output_base_dir
        self.short_term_memory_window_seconds = short_term_memory_window_seconds
        self.archive_short_term_older_than_seconds = archive_short_term_older_than_seconds

        self.images_dir = os.path.join(self.output_base_dir, "images")
        self.models_dir = os.path.join(self.output_base_dir, "models")
        self.videos_dir = os.path.join(self.output_base_dir, "videos")

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        
        logger.info(f"AIProcessingPipeline initialized. Outputs in '{self.output_base_dir}'. "
                    f"STM window: {self.short_term_memory_window_seconds}s, Archive STM older than: {self.archive_short_term_older_than_seconds}s")

    def _generate_unique_filename(self, directory: str, extension: str) -> str:
        filename = f"{uuid.uuid4().hex}.{extension}"
        return os.path.join(directory, filename)

    def _extract_tags_from_prompt(self, prompt: str, max_tags: int = 10) -> List[str]:
        if not prompt: return []
        words = set(w.lower().strip(",.!?") for w in prompt.split() if len(w) > 3)
        common_words = {"a", "the", "and", "of", "for", "with", "in", "on", "at", "is", "it", "this", "that", "to", "by", "from", "make", "me", "create", "generate", "design"}
        tags = [word for word in words if word not in common_words]
        return list(tags)[:max_tags] 

    def _generate_image_and_3d(self, generation_prompt: str, user_id: str) -> Dict[str, Any]:
        image_gen_result = {"image_path": None, "model_path": None, "video_path": None, "error": None}
        image_bytes = None
        current_stage = "Text-to-Image Generation (helper)"
        try:
            t2i_payload = {"prompt": generation_prompt}
            t2i_response = self.stub.call(self.text_to_image_app_id, t2i_payload, user_id)
            if not t2i_response: raise ValueError("T2I app returned empty response.")
            image_bytes = t2i_response.get('result')
            if not image_bytes or not isinstance(image_bytes, bytes):
                raise ValueError(f"T2I app: 'result' not valid image bytes. Got: {type(image_bytes)}")
            image_gen_result["image_path"] = self._generate_unique_filename(self.images_dir, "png")
            with open(image_gen_result["image_path"], 'wb') as f: f.write(image_bytes)
            logger.info(f"Image saved: {image_gen_result['image_path']}")
        except Exception as e:
            logger.error(f"Error in {current_stage}: {e}", exc_info=True)
            image_gen_result["error"] = f"Error in {current_stage}: {str(e)}"
            return image_gen_result
        current_stage = "Image-to-3D Generation (helper)"
        try:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            i23d_payload = {"input_image": image_base64}
            i23d_response = self.stub.call(self.image_to_3d_app_id, i23d_payload, user_id)
            if not i23d_response: raise ValueError("I23D app returned empty response.")
            model_data = i23d_response.get('generated_object')
            if not model_data or not isinstance(model_data, bytes):
                raise ValueError(f"I23D app: 'generated_object' not valid model bytes. Got: {type(model_data)}")
            image_gen_result["model_path"] = self._generate_unique_filename(self.models_dir, "glb")
            with open(image_gen_result["model_path"], 'wb') as f: f.write(model_data)
            logger.info(f"Model saved: {image_gen_result['model_path']}")
            video_data = i23d_response.get('video_object')
            if video_data:
                video_bytes_final = None
                if isinstance(video_data, str):
                    try: video_bytes_final = base64.b64decode(video_data)
                    except Exception: logger.warning("Failed to decode video_object from base64.")
                elif isinstance(video_data, bytes): video_bytes_final = video_data
                if video_bytes_final:
                    image_gen_result["video_path"] = self._generate_unique_filename(self.videos_dir, "mp4")
                    with open(image_gen_result["video_path"], 'wb') as f: f.write(video_bytes_final)
                    logger.info(f"Video saved: {image_gen_result['video_path']}")
        except Exception as e:
            logger.error(f"Error in {current_stage}: {e}", exc_info=True)
            image_gen_result["error"] = (image_gen_result["error"] + "; " if image_gen_result["error"] else "") + f"Error in {current_stage}: {str(e)}"
        return image_gen_result

    def run(self, user_prompt: str, user_id: str = 'default_user') -> Dict[str, Any]:
        
        result = {
            "creation_id": None, "original_prompt": user_prompt, "enhanced_prompt": None,
            "image_path": None, "model_path": None, "video_path": None,
            "tags": [], "error": None, "message_to_user": f"Processing your request: '{user_prompt[:50]}...'"
        }
        current_stage = "Initial Memory Operations"
        logger.info(f"--- Pipeline Run Start for User: {user_id}, Prompt: '{user_prompt}' ---")
        try:
            self.memory_manager.archive_short_term_log(user_id, self.archive_short_term_older_than_seconds)

            history_for_llm = self.memory_manager.get_recent_short_term_interactions(
                user_id, self.short_term_memory_window_seconds, limit=6 
            )
            if history_for_llm:
                 logger.info(f"HISTORY BEING SENT TO LLM (last item of actual history): {history_for_llm[-1]}")
            else:
                 logger.info("No prior history being sent to LLM for this turn.")
            logger.info(f"Retrieved {len(history_for_llm)} interactions to use as context for LLM.")


            current_utc_time_iso = datetime.now(timezone.utc).isoformat() 

            self.memory_manager.log_short_term_interaction(
                user_id, "USER_INPUT", {"text": user_prompt, "timestamp_iso": current_utc_time_iso} 
            )

            current_stage = "LLM Intent Parsing"
            logger.info(f"Stage: {current_stage}")
            llm_interpretation = self.llm_client.interpret_user_request(user_prompt, conversation_history=history_for_llm)
            
            current_utc_time_iso = datetime.now(timezone.utc).isoformat() 
            self.memory_manager.log_short_term_interaction(
                user_id, "ASSISTANT_INTERPRETATION", {"interpretation": llm_interpretation, "timestamp_iso": current_utc_time_iso}
            )

            if not llm_interpretation or "intent" not in llm_interpretation:
                logger.error(f"LLM interpretation failed or malformed: {llm_interpretation}")
                result["error"] = "Failed to understand request due to LLM interpretation error."
                result["message_to_user"] = "I'm sorry, I had trouble understanding that. Could you try rephrasing?"
                return result 
            
            logger.info(f"LLM Interpretation: {json.dumps(llm_interpretation, indent=2)}")
            intent = llm_interpretation.get("intent", "UNKNOWN")
            final_generation_prompt = None
            recalled_item_info_for_msg = "" 

            if intent == "NEW_CREATION":
                current_stage = "New Creation Prompting"
                subject_desc = llm_interpretation.get("subject_description")
                if subject_desc:
                    final_generation_prompt = self.llm_client.creatively_enhance_subject(subject_desc) 
                    result["message_to_user"] = f"Okay, creating something new based on: '{subject_desc[:70]}...'"
                else: 
                     logger.warning("LLM NEW_CREATION intent but no subject_description. Using original prompt for enhancement.")
                     final_generation_prompt = self.llm_client.creatively_enhance_subject(user_prompt) 
                     result["message_to_user"] = f"Okay, creating something new based on your input: '{user_prompt[:70]}...'"

            elif intent == "SHORT_TERM_MODIFICATION":
                current_stage = "Short-Term Modification Prompting"

                base_subject_for_mod = llm_interpretation.get("subject_description") 
                modification = llm_interpretation.get("modification_instruction")
                
                if base_subject_for_mod and modification:
                    logger.info(f"Attempting short-term mod. LLM's base subject: '{base_subject_for_mod}', Mod: '{modification}'")
                    final_generation_prompt = self.llm_client.combine_past_prompt_with_modification(base_subject_for_mod, modification) # LLM Call 2 (Modification)
                    result["message_to_user"] = f"Okay, trying to modify the recent idea ('{base_subject_for_mod[:50]}...') with: '{modification[:50]}...'"
                else:
                    logger.warning("SHORT_TERM_MODIFICATION: Not enough info from LLM (subject or modification missing). Fallback to new creation.")
                    result["message_to_user"] = "I wasn't sure what to modify from recent context, so I'll try creating something new based on your last message."
                    final_generation_prompt = self.llm_client.creatively_enhance_subject(user_prompt) # CORRECTED METHOD NAME

            elif intent == "LONG_TERM_RECALL_MODIFICATION":
                current_stage = "Long-Term Recall & Modification Prompting"
                keywords_value = llm_interpretation.get("memory_search_keywords")
                date_ref_str_from_llm = llm_interpretation.get("memory_date_reference_original")
                modification = llm_interpretation.get("modification_instruction")

                search_keywords_list = []
                if isinstance(keywords_value, str): search_keywords_list = [k.strip() for k in keywords_value.split(',') if k.strip()]
                elif isinstance(keywords_value, list): search_keywords_list = [str(k).strip() for k in keywords_value if k and str(k).strip()]
                
                search_query_for_semantic = " ".join(search_keywords_list) if search_keywords_list else user_prompt
                
                parsed_date_start, parsed_date_end = None, None
                date_string_to_parse_for_recall = None
                if isinstance(date_ref_str_from_llm, str): date_string_to_parse_for_recall = date_ref_str_from_llm
                elif isinstance(date_ref_str_from_llm, list) and len(date_ref_str_from_llm) > 0:
                    date_string_to_parse_for_recall = str(date_ref_str_from_llm[0])
                
                if date_string_to_parse_for_recall and DATEPARSER_AVAILABLE:
                    parsed_date = dateparser.parse(date_string_to_parse_for_recall, settings={'PREFER_DATES_FROM': 'past'})
                    if parsed_date:
                        logger.info(f"Parsed date '{date_string_to_parse_for_recall}' to: {parsed_date.date()}")
                        parsed_date_start = datetime.combine(parsed_date.date(), datetime.min.time())
                        parsed_date_end = datetime.combine(parsed_date.date(), datetime.max.time())
                
                found_creations = []
                if self.memory_manager.embedding_model:
                    found_creations = self.memory_manager.search_creations_semantic(
                        query_text=search_query_for_semantic, user_id=user_id, top_k=5)
                    if parsed_date_start and found_creations: 
                        dated_results = [fc for fc in found_creations if parsed_date_start <= datetime.fromisoformat(fc['created_at'].replace(' ', 'T')) <= parsed_date_end] # Handle potential space in timestamp
                        logger.info(f"Semantic search found {len(found_creations)}, after date filter: {len(dated_results)}")
                        found_creations = dated_results

                if not found_creations and search_keywords_list: 
                    found_creations = self.memory_manager.search_creations_keyword_date(
                        search_term=search_keywords_list[0] if search_keywords_list else None,
                        user_id=user_id, date_start=parsed_date_start, date_end=parsed_date_end, limit=1)

                if found_creations and modification:
                    recalled_item = found_creations[0]
                    recalled_item_info_for_msg = f"(Recalled ID: {recalled_item['id']} - '{recalled_item['original_prompt'][:30]}...')"
                    logger.info(f"Found recalled item {recalled_item_info_for_msg}")
                    base_prompt_for_mod = recalled_item['enhanced_prompt'] or recalled_item['original_prompt']
                    final_generation_prompt = self.llm_client.combine_past_prompt_with_modification(base_prompt_for_mod, modification) # LLM Call 2 (Modification)
                    result["message_to_user"] = f"Okay, using your past creation {recalled_item_info_for_msg}. Modifying with: '{modification[:50]}...'"
                else:
                    logger.warning("LONG_TERM_RECALL_MODIFICATION: No relevant past creation found or modification missing. Fallback.")
                    result["message_to_user"] = "I couldn't find a matching past creation for modification, or the modification was unclear. I'll try creating something new based on your request."
                    final_generation_prompt = self.llm_client.creatively_enhance_subject(user_prompt)


            else:
                logger.info(f"Unhandled intent '{intent}' or insufficient data. Attempting default enhancement.")
                subject_for_enhancement = llm_interpretation.get("subject_description", user_prompt) 
                result["message_to_user"] = f"I'll try to create something based on: '{subject_for_enhancement[:70]}...'"
                final_generation_prompt = self.llm_client.creatively_enhance_subject(subject_for_enhancement)

            if final_generation_prompt:
                result["enhanced_prompt"] = final_generation_prompt
                generation_outputs = self._generate_image_and_3d(final_generation_prompt, user_id)
                
                result["image_path"] = generation_outputs["image_path"]
                result["model_path"] = generation_outputs["model_path"]
                result["video_path"] = generation_outputs["video_path"]
                if generation_outputs["error"]:
                    result["error"] = (result["error"] + "; " if result["error"] else "") + generation_outputs["error"]
                    result["message_to_user"] += f" However, there was an issue during asset generation: {generation_outputs['error']}"

                if result["image_path"] and not generation_outputs["error"]:
                    current_stage = "Long-Term Memory Save"
                    logger.info(f"Stage: {current_stage}")
                    tags = self._extract_tags_from_prompt(final_generation_prompt)
                    result["tags"] = tags
                    creation_id = self.memory_manager.save_creation_summary(
                        user_id=user_id, original_prompt=user_prompt, 
                        enhanced_prompt=final_generation_prompt, 
                        image_path=result["image_path"], model_path=result["model_path"],
                        video_path=result["video_path"], tags=tags
                    )
                    result["creation_id"] = creation_id
                    if creation_id:
                        logger.info(f"Creation details saved to long-term memory with ID: {creation_id}")
                        result["message_to_user"] += f" New creation saved with ID: {creation_id}."
                        current_utc_time_iso = datetime.now(timezone.utc).isoformat()
                        generation_summary_for_stm = {
                            "description": final_generation_prompt[:100] + "...", 
                            "creation_id": creation_id,
                            "image_path": result["image_path"] is not None,
                            "model_path": result["model_path"] is not None,
                            "timestamp_iso": current_utc_time_iso 
                        }
                        self.memory_manager.log_short_term_interaction(user_id, "ASSISTANT_GENERATION_SUMMARY", generation_summary_for_stm)
                    else:
                        logger.error("Failed to save creation details to long-term memory.")
            elif not result["error"]: 
                if not result["message_to_user"]:
                    result["message_to_user"] = "I couldn't determine what to create. Please try rephrasing your request."
                logger.info(result["message_to_user"])


        except Exception as e:
            logger.error(f"Critical error in AIProcessingPipeline stage '{current_stage}': {e}", exc_info=True)
            result["error"] = f"Critical error in stage '{current_stage}': {str(e)}"
            result["message_to_user"] = f"An unexpected critical error occurred: {str(e)}"
        
        logger.info(f"--- Pipeline Run End. Result Message: {result['message_to_user']} ---")
        return result


# Test Section 
if __name__ == '__main__':

    print("--- AIProcessingPipeline Full Logic Test ---")

    MOCK_TEXT_TO_IMAGE_APP_ID = "c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network"
    MOCK_IMAGE_TO_3D_APP_ID = "f0b5f319156c4819b9827000b17e511a.node3.openfabric.network"
    TEST_MODEL_NAME = "gemma3:1b" 
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    TEST_DB_DIR = os.path.join(current_script_dir, "test_pipeline_stm_data")
    os.makedirs(TEST_DB_DIR, exist_ok=True)
    TEST_SQLITE_PATH = os.path.join(TEST_DB_DIR, "test_pipeline_stm_creations.db")
    TEST_FAISS_PATH = os.path.join(TEST_DB_DIR, "test_pipeline_stm_creations.faiss")
    if os.path.exists(TEST_SQLITE_PATH): os.remove(TEST_SQLITE_PATH)
    if os.path.exists(TEST_FAISS_PATH): os.remove(TEST_FAISS_PATH)
    if os.path.exists(TEST_FAISS_PATH + ".ids.json"): os.remove(TEST_FAISS_PATH + ".ids.json")

    test_output_dir = os.path.join(current_script_dir, "test_pipeline_stm_generated_assets")

    try:
        stub_instance = Stub(app_ids=[MOCK_TEXT_TO_IMAGE_APP_ID, MOCK_IMAGE_TO_3D_APP_ID])
        llm_instance = OllamaLLMClient(model_name=TEST_MODEL_NAME)
        memory_instance = MemoryManager(db_path=TEST_SQLITE_PATH, faiss_index_path=TEST_FAISS_PATH)
        
        pipeline = AIProcessingPipeline(
            stub=stub_instance, llm_client=llm_instance, memory_manager=memory_instance,
            text_to_image_app_id=MOCK_TEXT_TO_IMAGE_APP_ID, image_to_3d_app_id=MOCK_IMAGE_TO_3D_APP_ID,
            output_base_dir=test_output_dir,
            short_term_memory_window_seconds=60, # 1 minute for testing
            archive_short_term_older_than_seconds=120 # 2 minutes for testing
        )
        print("Pipeline initialized for STM test.")

        # Test Flow
        print("\n--- Test 1: New Creation ---")
        res1 = pipeline.run("A majestic lion king", user_id="user_stm_test")
        print(f"Result 1: {json.dumps(res1, indent=2)}")
        if res1.get("creation_id"):
            print(f"Sleeping for a bit to create time difference for STM...")
            time.sleep(5) # Wait 5 seconds

        print("\n--- Test 2: Short-term modification (implicit 'it') ---")
        # The LLM needs to get "lion king" context from the history MemoryManager provides
        res2 = pipeline.run("make it wear a crown", user_id="user_stm_test")
        print(f"Result 2: {json.dumps(res2, indent=2)}")
        if res2.get("creation_id"):
            print(f"Sleeping for a bit longer...")
            time.sleep(65) # Sleep longer than STM window (60s) + a bit

        print("\n--- Test 3: Another new creation (old STM for lion should be mostly gone for LLM context) ---")
        res3 = pipeline.run("a futuristic motorcycle", user_id="user_stm_test")
        print(f"Result 3: {json.dumps(res3, indent=2)}")
        
        print("\n--- Test 4: Long-term recall of the lion ---")
        # Assuming lion was creation_id 1
        res4 = pipeline.run("Remember the lion king I made? Add a golden scepter to its paw.", user_id="user_stm_test")
        print(f"Result 4: {json.dumps(res4, indent=2)}")

        memory_instance.close()
        print("\n--- AIProcessingPipeline STM Test Completed ---")

    except ImportError as ie:
        print(f"ImportError during test: {ie}. Ensure all custom modules and dependencies like 'dateparser' are installed and accessible.")
    except Exception as e:
        logger.error("Pipeline direct STM test failed", exc_info=True)
        print(f"An error occurred during the pipeline STM test: {e}")