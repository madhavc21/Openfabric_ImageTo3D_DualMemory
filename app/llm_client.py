import requests
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaLLMClient:
    def __init__(self,
                 model_name: str,
                 ollama_base_url: str = "http://localhost:11434",
                 request_timeout: int = 180): 
        if not model_name:
            raise ValueError("Ollama model name must be provided.")
            
        self.model_name = model_name
        self.base_url = ollama_base_url.rstrip('/')
        self.chat_api_url = f"{self.base_url}/api/chat"
        self.timeout = request_timeout

        logger.info(f"OllamaLLMClient initialized for model '{self.model_name}' at '{self.base_url}'")
        self._check_ollama_connection()

    def _check_ollama_connection(self):
        try:
            response = requests.get(self.base_url, timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama at {self.base_url}. Ensure Ollama is running.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Error communicating with Ollama: {e}")
        except Exception as e: 
            logger.error(f"An unexpected error occurred during Ollama connection check: {e}")

    def _make_ollama_request(self, 
                             messages: List[Dict[str, str]], 
                             output_format: Optional[str] = None
                             ) -> Optional[Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": { "temperature": 0.3 } 
        }
        if output_format == "json":
            payload["format"] = "json"

        logger.info(f"Sending request to Ollama. Output format: {output_format or 'text'}. Model: {self.model_name}")
        if messages: 
            log_msg_content = messages[-1]['content']
            logger.debug(f"Last message content (first 150 chars): {log_msg_content[:150]}...")
        
        try:
            response = requests.post(self.chat_api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_data = response.json()
            
            if 'message' not in response_data or 'content' not in response_data['message']:
                logger.error(f"Ollama response missing 'message.content'. Response: {response_data}"); return None

            llm_content_str = response_data['message']['content']
            logger.debug(f"Raw LLM content (first 300): {llm_content_str[:300]}...")

            if output_format == "json":
                try: 
                    parsed_json_content = json.loads(llm_content_str)
                    logger.info("Successfully parsed JSON content from LLM.")
                    return parsed_json_content
                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse JSON string from LLM content: {je}. Content was: '{llm_content_str}'"); return None
            else: 
                return llm_content_str.strip()

        except requests.exceptions.Timeout:
            logger.error(f"Request to Ollama timed out after {self.timeout} seconds."); return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during Ollama API request: {e}"); return None
        except json.JSONDecodeError as e: 
            logger.error(f"Failed to decode overall JSON response: {e}. Response text: {response.text}"); return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Ollama request: {e}", exc_info=True); return None

    def interpret_user_request(self, 
                               user_input: str, 
                               conversation_history: Optional[List[Dict[str, str]]] = None
                               ) -> Optional[Dict[str, Any]]:
        if not user_input:
            logger.warning("User input for interpretation is empty.")
            return None

        today_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d (%A)")
        system_prompt = f"""Your ONLY task is to analyze the user's request and current conversation context, then respond with a structured JSON object. Today's date is {today_date_str}.
Strictly adhere to the JSON format below. All fields must be present; use null if a field is not applicable. Do NOT include any text outside the JSON block.

Possible Intents:
- "NEW_CREATION": User wants to create something new.
- "SHORT_TERM_MODIFICATION": User wants to modify the immediately preceding creation. The subject is implied from recent context.
- "LONG_TERM_RECALL_MODIFICATION": User wants to recall a specific past creation and then modify it.
- "LONG_TERM_RECALL_ONLY": User wants to find/see a specific past creation without modification.
- "GENERAL_QUERY": A general question or statement not directly related to creation/recall.
- "UNKNOWN": Intent cannot be determined.

JSON Output Structure:
{{
  "intent": "STRING (One of the Possible Intents listed above)",
  "subject_description": "STRING | null (For NEW_CREATION: the core subject as stated by user. For SHORT_TERM_MODIFICATION: your inference of what 'it' refers to based on history, or the subject if explicitly stated. For LONG_TERM_RECALL intents: the primary subject/keywords for recall.)",
  "modification_instruction": "STRING | null (The specific change requested, e.g., 'add wings', 'make it blue'. Null if not a modification intent.)",
  "memory_search_keywords": ["STRING", ...] | null (ONLY for LONG_TERM_RECALL intents: 1-3 concise keywords (nouns, key adjectives) like ["dragon", "castle", "robot", "car", "red"]. Null for other intents. MUST be a JSON list of strings or null.)",
  "memory_date_reference_original": "STRING | null (ONLY for LONG_TERM_RECALL intents: the exact date phrase used by the user, e.g., 'last tuesday', 'yesterday'. Null for other intents.)"
}}

Guidelines & Examples:

1. User Input: "A majestic griffin soaring."
   Your JSON Response:
   {{
     "intent": "NEW_CREATION",
     "subject_description": "A majestic griffin soaring",
     "modification_instruction": null,
     "memory_search_keywords": null,
     "memory_date_reference_original": null
   }}

2. (Previous assistant action: Generated a "blue sphere")
   User Input: "Make it green and larger."
   Your JSON Response:
   {{
     "intent": "SHORT_TERM_MODIFICATION",
     "subject_description": "blue sphere", // What 'it' refers to, based on history
     "modification_instruction": "make it green and larger",
     "memory_search_keywords": null,
     "memory_date_reference_original": null
   }}

3. User Input: "Remember that robot I created last month? Can you make a version of it with laser eyes?"
   Your JSON Response:
   {{
     "intent": "LONG_TERM_RECALL_MODIFICATION",
     "subject_description": null, // Keywords will find it
     "modification_instruction": "make a version of it with laser eyes",
     "memory_search_keywords": ["robot"],
     "memory_date_reference_original": "last month"
   }}

4. User Input: "Show me the castle pictures from the beginning of February."
   Your JSON Response:
   {{
     "intent": "LONG_TERM_RECALL_ONLY",
     "subject_description": null, // Keywords will find it
     "modification_instruction": null,
     "memory_search_keywords": ["castle", "pictures"],
     "memory_date_reference_original": "the beginning of February"
   }}

5. User Input: "What time is it?"
   Your JSON Response:
   {{
     "intent": "GENERAL_QUERY",
     "subject_description": "What time is it?",
     "modification_instruction": null,
     "memory_search_keywords": null,
     "memory_date_reference_original": null
   }}
   
Conversation History (if provided, most recent turns):
{json.dumps(conversation_history[-4:] if conversation_history else [], indent=2)}

Current User Input (process this one):
"""        
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_input})
        
        logger.info(f"Requesting interpretation for: '{user_input}' (History embedded in system prompt)")
        structured_response = self._make_ollama_request(messages, output_format="json") 
        
        default_response_structure = {
            "intent": "UNKNOWN", "subject_description": user_input, 
            "modification_instruction": None, "memory_search_keywords": None, 
            "memory_date_reference_original": None
        }
        if isinstance(structured_response, dict):
            for key in default_response_structure:
                if key not in structured_response:
                    structured_response[key] = default_response_structure.get(key) 
            
            if isinstance(structured_response.get("memory_search_keywords"), str) and structured_response["memory_search_keywords"]:
                structured_response["memory_search_keywords"] = [k.strip() for k in structured_response["memory_search_keywords"].split(',') if k.strip()]
            elif not isinstance(structured_response.get("memory_search_keywords"), list):
                 structured_response["memory_search_keywords"] = None
            return structured_response
        else:
            logger.error(f"LLM interpretation did not return a valid dictionary. Received: {structured_response}")
            return default_response_structure


    def creatively_enhance_subject(self, subject_description: str) -> Optional[str]:
        if not subject_description: 
            logger.warning("Subject description is empty for creative enhancement."); return None
        system_message = (
            "You are an AI assistant specializing in creative prompt engineering for image generation. "
            "Your task is to take a user's subject description and expand it into a rich, descriptive prompt paragraph "
            "that will inspire stunning visuals. Focus on artistic style, mood, lighting, composition, "
            "textures, and specific visual elements. Use vivid language. "
            "Output ONLY the enhanced prompt paragraph. No conversational fluff, no intro phrases like 'Here is an enhanced prompt:'."
        )
        user_request_message = (
            f"Please expand the following subject into a creative and detailed prompt for an image generation model:\n"
            f"Subject: \"{subject_description}\""
        )
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_request_message}]
        logger.info(f"Requesting creative enhancement for subject: '{subject_description}'")
        enhanced_prompt = self._make_ollama_request(messages, output_format=None)
        if enhanced_prompt: 
            logger.info(f"Creative enhancement generated: '{enhanced_prompt[:100]}...'")
        else: 
            logger.warning(f"Creative enhancement failed for '{subject_description}'. Returning original subject.")
            return subject_description 
        return enhanced_prompt

    def combine_past_prompt_with_modification(self, base_prompt: str, modification_instruction: str) -> Optional[str]:
        if not base_prompt or not modification_instruction: 
            logger.warning("Missing base prompt or modification instruction for combination.")
            return base_prompt if base_prompt else None 
            
        system_message = (
            "You are an AI assistant that revises creative image prompts. "
            "Given a base image prompt and a modification instruction, "
            "intelligently integrate the modification into the base prompt to create a new, cohesive prompt "
            "suitable for an image generation model. Preserve the style, mood, and key elements of the base prompt "
            "unless the modification explicitly changes them. "
            "Output ONLY the new modified prompt as a single paragraph. No conversational fluff."
        )
        user_request_message = (
            f"Base Prompt to be modified:\n\"{base_prompt}\"\n\n"
            f"Modification Instruction to apply:\n\"{modification_instruction}\"\n\n"
            f"Generate the new, modified prompt incorporating the instruction into the base prompt:"
        )
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_request_message}]
        logger.info(f"Requesting prompt combination. Base: '{base_prompt[:50]}...', Mod: '{modification_instruction}'")
        new_prompt = self._make_ollama_request(messages, output_format=None)
        if new_prompt: 
            logger.info(f"Generated modified prompt: '{new_prompt[:100]}...'")
        else: 
            logger.warning(f"Prompt combination failed. Attempting simple concatenation as fallback.")
            return f"{base_prompt}, {modification_instruction}" 
        return new_prompt

# Test Section
if __name__ == '__main__':
    TEST_MODEL_NAME = "gemma3:1b"
    print(f"--- Testing OllamaLLMClient with model: {TEST_MODEL_NAME} ---")

    try:
        client = OllamaLLMClient(model_name=TEST_MODEL_NAME)
        
        print("\n--- Test 1: interpret_user_request (NEW_CREATION) ---")
        interp1 = client.interpret_user_request("A cyberpunk cityscape at dusk with flying cars.")
        print(f"Interpretation 1: {json.dumps(interp1, indent=2)}")
        assert interp1 and interp1.get("intent") == "NEW_CREATION"
        assert interp1.get("subject_description", "").strip().lower() == "a cyberpunk cityscape at dusk with flying cars"

        print("\n--- Test 2: interpret_user_request (SHORT_TERM_MODIFICATION) ---")
        history_for_stm = [
            {"role": "user", "content": "Create a red apple"},
            {"role": "assistant", "content": json.dumps({"intent": "NEW_CREATION", "subject_description": "a red apple"})} # Simulating AI's understanding of last turn
        ]
        interp2 = client.interpret_user_request("make it green", conversation_history=history_for_stm)
        print(f"Interpretation 2: {json.dumps(interp2, indent=2)}")
        assert interp2 and interp2.get("intent") == "SHORT_TERM_MODIFICATION"
        assert interp2.get("modification_instruction", "").strip().lower() == "make it green"
        # Check if LLM correctly identified the subject from history
        assert interp2.get("subject_description", "").strip().lower() == "a red apple", \
               f"STM Subject Expected: 'a red apple', Got: '{interp2.get('subject_description')}'"


        print("\n--- Test 3: interpret_user_request (LONG_TERM_RECALL_MODIFICATION) ---")
        interp3 = client.interpret_user_request("Remember the dragon I made last week? Change its color to gold.")
        print(f"Interpretation 3: {json.dumps(interp3, indent=2)}")
        assert interp3 and interp3.get("intent") == "LONG_TERM_RECALL_MODIFICATION"
        assert interp3.get("modification_instruction", "").strip().lower() == "change its color to gold"
        assert isinstance(interp3.get("memory_search_keywords"), list) and "dragon" in interp3.get("memory_search_keywords", [])
        assert interp3.get("memory_date_reference_original", "").strip().lower() == "last week"

        print("\n--- Test 4: creatively_enhance_subject ---")
        enhanced = client.creatively_enhance_subject("a lonely lighthouse on a stormy coast")
        print(f"Enhanced: {enhanced}")
        assert enhanced and "lighthouse" in enhanced and len(enhanced) > 40 

        print("\n--- Test 5: combine_past_prompt_with_modification ---")
        base = "A majestic medieval castle on a hill, sunny day, photorealistic."
        mod = "make it a haunted castle during a thunderstorm at night"
        combined = client.combine_past_prompt_with_modification(base, mod)
        print(f"Combined: {combined}")
        assert combined and "haunted" in combined and "thunderstorm" in combined and "castle" in combined

        print("\n--- ALL LLM CLIENT TESTS PASSED (or check logs) ---")

    except Exception as e:
        logger.error("Error during client test", exc_info=True)
        print(f"Test Error: {e}")