import sqlite3
import json
import logging
from datetime import datetime, timedelta, date, timezone
from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS or SentenceTransformers not installed. Semantic search will be disabled. "
                    "Install with: pip install faiss-cpu sentence-transformers")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class MemoryManager:
    def __init__(self, 
                 db_path: str, 
                 faiss_index_path: Optional[str] = None, 
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_sqlite_tables()
        logger.info(f"MemoryManager initialized with SQLite database at '{self.db_path}'")

        self.faiss_index = None
        self.embedding_model = None
        self.faiss_index_path = faiss_index_path
        self.sqlite_ids_for_faiss = [] 
        self.embedding_dim = 384

        if FAISS_AVAILABLE:
            try:
                logger.info(f"Loading sentence-transformer model: {embedding_model_name}")
                self.embedding_model = SentenceTransformer(embedding_model_name)
                dummy_embedding = self.embedding_model.encode(["test"])[0]
                self.embedding_dim = len(dummy_embedding)
                logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

                if self.faiss_index_path and os.path.exists(self.faiss_index_path):
                    logger.info(f"Loading FAISS index from {self.faiss_index_path}")
                    self.faiss_index = faiss.read_index(self.faiss_index_path)
                    ids_map_path = self.faiss_index_path + ".ids.json"
                    if os.path.exists(ids_map_path):
                        with open(ids_map_path, 'r') as f:
                            self.sqlite_ids_for_faiss = json.load(f)
                        logger.info(f"Loaded {len(self.sqlite_ids_for_faiss)} IDs for FAISS index mapping to 'creations' table.")
                    else: 
                        logger.warning(f"FAISS index loaded, but ID map file '{ids_map_path}' not found. Rebuilding ID map if possible (requires all creations).")
                        self._rebuild_faiss_id_map_from_db() 
                else:
                    logger.info("No existing FAISS index file found or path not provided. Creating new FAISS index.")
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim) 
                
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer or FAISS: {e}. Semantic search will be disabled.", exc_info=True)
                self.embedding_model = None; self.faiss_index = None
        else:
            logger.info("FAISS or SentenceTransformers not available. Semantic search functionality will be disabled.")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;") 
        return conn

    def _create_sqlite_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Long-term "creations" table (summary of successfully generated assets)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS creations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    original_prompt TEXT NOT NULL,
                    enhanced_prompt TEXT,
                    image_path TEXT,
                    model_path TEXT,
                    video_path TEXT,
                    tags TEXT,             -- JSON serialized list of strings
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT          -- JSON serialized dictionary
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_creations_user_created ON creations (user_id, created_at DESC);")
            logger.info("SQLite table 'creations' (long-term) checked/created.")

            # Short-term "interaction_log" table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    interaction_type TEXT NOT NULL, -- e.g., "USER_INPUT", "LLM_INTERPRETATION", "ASSISTANT_GENERATION_SUMMARY"
                    data TEXT NOT NULL            -- JSON blob of interaction details
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interaction_log_user_time ON interaction_log (user_id, timestamp DESC);")
            logger.info("SQLite table 'interaction_log' (short-term) checked/created.")
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error creating SQLite tables: {e}")
            raise
        finally:
            conn.close()
    
    def _row_to_dict(self, row: sqlite3.Row) -> Optional[Dict[str, Any]]: 
        """Converts a sqlite3.Row object to a dictionary, parsing JSON fields."""
        if not row:
            return None
        item = dict(row) 
        
        if 'tags' in item and item['tags'] is not None:
            try: item['tags'] = json.loads(item['tags'])
            except json.JSONDecodeError: logger.warning(f"Could not decode tags JSON for item ID {item.get('id', item.get('log_id'))}: {item['tags']}"); item['tags'] = []
        elif 'tags' in item and item['tags'] is None: item['tags'] = []

        if 'metadata' in item and item['metadata'] is not None:
            try: item['metadata'] = json.loads(item['metadata'])
            except json.JSONDecodeError: logger.warning(f"Could not decode metadata JSON for item ID {item.get('id', item.get('log_id'))}: {item['metadata']}"); item['metadata'] = {}
        elif 'metadata' in item and item['metadata'] is None: item['metadata'] = {}

        if 'data' in item and item['data'] is not None: 
            try: item['data'] = json.loads(item['data'])
            except json.JSONDecodeError: logger.warning(f"Could not decode 'data' JSON for interaction_log ID {item.get('log_id')}: {item['data']}"); item['data'] = {}
        elif 'data' in item and item['data'] is None: item['data'] = {}
            
        return item
    
    def _rebuild_faiss_id_map_from_db(self):
        """Attempts to rebuild sqlite_ids_for_faiss if the index exists but map is lost."""
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            self.sqlite_ids_for_faiss = []
            return

        logger.info("Attempting to rebuild FAISS ID map from all creations in SQLite DB (this assumes FAISS index was built in same order).")
        all_creations = self.get_all_creations_for_faiss_rebuild(limit=self.faiss_index.ntotal) 
        if len(all_creations) == self.faiss_index.ntotal:
            self.sqlite_ids_for_faiss = [c['id'] for c in all_creations]
            logger.info(f"Successfully rebuilt FAISS ID map with {len(self.sqlite_ids_for_faiss)} entries.")
            self._save_faiss_index() 
        else:
            logger.error(f"Mismatch between FAISS ntotal ({self.faiss_index.ntotal}) and creations in DB ({len(all_creations)}). Cannot reliably rebuild ID map. Consider re-indexing all data.")
            self.sqlite_ids_for_faiss = [] 

    def get_all_creations_for_faiss_rebuild(self, limit: int) -> List[Dict[str, Any]]:
        """Special getter for faiss rebuild, ensuring correct order if possible."""
        conn = self._get_connection()
        cursor = conn.cursor()
        items = []
        try:
            
            cursor.execute("SELECT id FROM creations ORDER BY id ASC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            for row in rows: items.append(dict(row)) 
        finally: conn.close()
        return items

    def _save_faiss_index(self):
        if self.faiss_index and self.faiss_index_path and FAISS_AVAILABLE:
            try:
                logger.info(f"Saving FAISS index to {self.faiss_index_path} with {self.faiss_index.ntotal} vectors.")
                os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
                faiss.write_index(self.faiss_index, self.faiss_index_path)
                ids_map_path = self.faiss_index_path + ".ids.json"
                with open(ids_map_path, 'w') as f:
                    json.dump(self.sqlite_ids_for_faiss, f)
                logger.info(f"FAISS index and ID map saved successfully.")
            except Exception as e:
                logger.error(f"Error saving FAISS index or ID map: {e}", exc_info=True)


    def _generate_embeddings(self, text_to_embed: str) -> Optional[np.ndarray]:
        if not self.embedding_model or not text_to_embed: return None
        try:
            embedding = self.embedding_model.encode([text_to_embed])[0]
            return embedding.astype('float32')
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text_to_embed[:50]}...': {e}")
            return None

    # Short-Term Memory Methods 
    def log_short_term_interaction(self, user_id: str, interaction_type: str, data: Dict[str, Any]) -> Optional[int]:
        """Logs an interaction to the short-term log."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            data_json = json.dumps(data)
            cursor.execute(
                "INSERT INTO interaction_log (user_id, interaction_type, data) VALUES (?, ?, ?)",
                (user_id, interaction_type, data_json)
            )
            conn.commit()
            log_id = cursor.lastrowid
            logger.debug(f"Logged short-term interaction for user '{user_id}', type '{interaction_type}', log_id {log_id}")
            return log_id
        except sqlite3.Error as e:
            logger.error(f"Error logging short-term interaction: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

    def get_recent_short_term_interactions(self, user_id: str, time_window_seconds: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves recent interactions for a user within a time window, formatted for LLM history."""
        conn = self._get_connection()
        cursor = conn.cursor()
        history_for_llm = []
        try:
            threshold_time_dt_aware = datetime.now(timezone.utc) - timedelta(seconds=time_window_seconds)
            threshold_time_sqlite_format_str = threshold_time_dt_aware.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"STM Fetch for user='{user_id}': Querying for timestamps (TEXT format) >= '{threshold_time_sqlite_format_str}' (Window: {time_window_seconds}s, Limit: {limit})")
            cursor.execute(
                """
                SELECT interaction_type, data, timestamp FROM interaction_log 
                WHERE user_id = ? AND timestamp >= ? 
                ORDER BY timestamp DESC LIMIT ?
                """, 
                (user_id, threshold_time_sqlite_format_str, limit) 
            )
            rows = cursor.fetchall()
            for row in reversed(rows): 
                interaction = self._row_to_dict(row) 
                role = "user" 
                content = ""
                interaction_data = interaction.get('data', {}) if isinstance(interaction, dict) else {}

                interaction_type = interaction.get('interaction_type')
                if interaction['interaction_type'] == "USER_INPUT":
                    role = "user"
                    content = interaction['data'].get('text', '')
                elif interaction['interaction_type'] == "ASSISTANT_INTERPRETATION":
                    role = "assistant" 
                    interpretation_content = interaction_data.get('interpretation', {})

                    content = json.dumps(interaction['data'].get('interpretation', {})) 
                elif interaction['interaction_type'] == "ASSISTANT_GENERATION_SUMMARY":
                    role = "assistant" 
                    content = f"Generated: {interaction['data'].get('description', 'an item')}. Image: {interaction['data'].get('image_path') is not None}. Model: {interaction['data'].get('model_path') is not None}."
                
                if content:
                    history_for_llm.append({"role": role, "content": content})
            
            logger.debug(f"Retrieved {len(history_for_llm)} recent interactions for user '{user_id}' for LLM context.")
        except sqlite3.Error as e:
            logger.error(f"Error fetching recent short-term interactions: {e}")
        finally:
            conn.close()
        return history_for_llm

    def archive_short_term_log(self, user_id: Optional[str] = None, older_than_seconds: int = 3600 * 24): # Default: 1 day
        """
        Deletes old entries from the interaction_log.
        Assumes that if an item was important enough for long-term recall,
        it was already saved to the 'creations' table by save_creation_summary.
        This method is now primarily for housekeeping the short-term log.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        deleted_count = 0
        try:
            threshold_time = datetime.now() - timedelta(seconds=older_than_seconds)
            
            base_query = "DELETE FROM interaction_log WHERE timestamp < ?"
            params = [threshold_time]

            if user_id:
                base_query += " AND user_id = ?"
                params.append(user_id)
            
            logger.info(f"Archiving (deleting) short-term log entries older than {threshold_time} for user: {user_id or 'ALL'}")
            cursor.execute(base_query, tuple(params))
            conn.commit()
            deleted_count = cursor.rowcount
            logger.info(f"Archived (deleted) {deleted_count} old entries from interaction_log.")
        except sqlite3.Error as e:
            logger.error(f"Error archiving short-term log: {e}")
            conn.rollback()
        finally:
            conn.close()
        return deleted_count

     # Long-Term Memory Methods
    def save_creation_summary(self,
                              user_id: str, original_prompt: str, enhanced_prompt: Optional[str],
                              image_path: Optional[str], model_path: Optional[str],
                              video_path: Optional[str] = None, tags: Optional[List[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        conn = self._get_connection()
        cursor = conn.cursor()
        creation_id = None
        try:
            tags_json = json.dumps(tags) if tags else None
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute(
                "INSERT INTO creations (user_id, original_prompt, enhanced_prompt, image_path, model_path, video_path, tags, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, original_prompt, enhanced_prompt, image_path, model_path, video_path, tags_json, metadata_json)
            )
            conn.commit()
            creation_id = cursor.lastrowid
            logger.info(f"Creation summary saved to 'creations' (long-term) table with ID: {creation_id}")

            if creation_id and self.faiss_index is not None and self.embedding_model is not None:
                text_for_embedding = f"Prompt: {enhanced_prompt or original_prompt}. Tags: {', '.join(tags) if tags else ''}."
                embedding_vector = self._generate_embeddings(text_for_embedding)
                if embedding_vector is not None:
                    self.faiss_index.add(np.array([embedding_vector]))
                    self.sqlite_ids_for_faiss.append(creation_id)
                    logger.info(f"Embedding for creation ID {creation_id} added to FAISS. Total: {self.faiss_index.ntotal}")
                    if self.faiss_index_path: self._save_faiss_index()
                else: logger.warning(f"Could not generate embedding for LT creation ID {creation_id}.")
            return creation_id
        except sqlite3.Error as e:
            logger.error(f"Error saving creation summary to 'creations' table: {e}"); conn.rollback()
            return None
        finally: conn.close()

    def get_creation_by_id(self, creation_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM creations WHERE id = ?", (creation_id,))
            row = cursor.fetchone()
            return self._row_to_dict(row)
        except sqlite3.Error as e: logger.error(f"Error fetching creation by ID {creation_id}: {e}"); return None
        finally: conn.close()
    
    def get_creations_by_ids(self, creation_ids: List[int]) -> List[Dict[str, Any]]:
        if not creation_ids: return []
        conn = self._get_connection()
        cursor = conn.cursor()
        results = []
        try:
            placeholders = ','.join(['?'] * len(creation_ids))
            query = f"SELECT * FROM creations WHERE id IN ({placeholders})"
            cursor.execute(query, tuple(creation_ids))
            rows = cursor.fetchall()
            temp_results = {self._row_to_dict(row)['id']: self._row_to_dict(row) for row in rows if row}
            results = [temp_results.get(cid) for cid in creation_ids if temp_results.get(cid)]
            return results
        except sqlite3.Error as e: logger.error(f"Error fetching creations by IDs: {e}"); return []
        finally: conn.close()

    def search_creations_keyword_date(self, 
                                      search_term: Optional[str] = None, 
                                      user_id: Optional[str] = None,
                                      date_start: Optional[datetime] = None,
                                      date_end: Optional[datetime] = None,
                                      limit: int = 10
                                      ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        results = []
        conditions = []; params = []
        if user_id: conditions.append("user_id = ?"); params.append(user_id)
        if search_term:
            like_term = f"%{search_term}%"
            conditions.append("(original_prompt LIKE ? OR enhanced_prompt LIKE ? OR tags LIKE ?)")
            params.extend([like_term, like_term, like_term])
        if date_start: conditions.append("created_at >= ?"); params.append(date_start)
        if date_end:
            effective_date_end = datetime.combine(date_end, datetime.max.time()) if isinstance(date_end, date) and not isinstance(date_end, datetime) else date_end
            conditions.append("created_at <= ?"); params.append(effective_date_end)
        query = "SELECT * FROM creations"
        if conditions: query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"; params.append(limit)
        try:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            for row in rows: results.append(self._row_to_dict(row))
        except sqlite3.Error as e: logger.error(f"Error in keyword/date search: {e}")
        finally: conn.close()
        return results

    def search_creations_semantic(self, 
                                  query_text: str, 
                                  user_id: Optional[str] = None, 
                                  top_k: int = 5
                                  ) -> List[Dict[str, Any]]:
        if not self.faiss_index or not self.embedding_model or not FAISS_AVAILABLE: return []
        if not query_text: return []
        logger.info(f"Semantic search for: '{query_text[:50]}...' (top_k={top_k})")
        query_embedding = self._generate_embeddings(query_text)
        if query_embedding is None: return []
        try:
            distances, faiss_indices = self.faiss_index.search(np.array([query_embedding]), self.faiss_index.ntotal if self.faiss_index.ntotal < top_k else top_k)
            found_sqlite_ids = []
            for i, faiss_idx in enumerate(faiss_indices[0]):
                if faiss_idx != -1 and faiss_idx < len(self.sqlite_ids_for_faiss):
                    sqlite_id = self.sqlite_ids_for_faiss[faiss_idx]
                    found_sqlite_ids.append(sqlite_id)
            if not found_sqlite_ids: return []
            retrieved_creations = self.get_creations_by_ids(found_sqlite_ids)
            final_results = [item for item in retrieved_creations if item and (user_id is None or item['user_id'] == user_id)][:top_k] 
            return final_results
        except Exception as e: logger.error(f"Error during FAISS search: {e}"); return []
        
    def close(self):
        self._save_faiss_index()
        logger.info("MemoryManager closed (FAISS index saved if applicable).")


# Test Section for Short-Term and Archiving -
if __name__ == '__main__':
    import time 

    TEST_DB_DIR_STM = "test_stm_memory_data"
    os.makedirs(TEST_DB_DIR_STM, exist_ok=True)
    TEST_SQLITE_PATH_STM = os.path.join(TEST_DB_DIR_STM, "test_stm_creations.db")
    TEST_FAISS_PATH_STM = os.path.join(TEST_DB_DIR_STM, "test_stm_creations.faiss") 

    for f_path in [TEST_SQLITE_PATH_STM, TEST_FAISS_PATH_STM, TEST_FAISS_PATH_STM + ".ids.json"]:
        if os.path.exists(f_path): os.remove(f_path)
    print(f"Cleaned up old test files in {TEST_DB_DIR_STM}")

    SHORT_TERM_WINDOW_SECONDS = 15 
    ARCHIVE_OLDER_THAN_SECONDS = 30

    memory = MemoryManager(db_path=TEST_SQLITE_PATH_STM, faiss_index_path=TEST_FAISS_PATH_STM)
    test_user = "stm_user_001"

    print(f"\n--- Testing Short-Term Memory (Window: {SHORT_TERM_WINDOW_SECONDS}s) ---")

    memory.log_short_term_interaction(test_user, "USER_INPUT", {"text": "Hello AI"})
    time.sleep(1)
    memory.log_short_term_interaction(test_user, "ASSISTANT_INTERPRETATION", {"interpretation": {"intent": "GREETING"}})
    time.sleep(1)
    memory.log_short_term_interaction(test_user, "USER_INPUT", {"text": "Create a happy cloud"})
    time.sleep(1)
    gen_summary_data_cloud = {
        "original_prompt": "Create a happy cloud", "enhanced_prompt": "A joyful, fluffy white cloud smiling in a bright blue sky.",
        "image_path": "cloud.png", "model_path": "cloud.glb", "tags": ["cloud", "happy", "sky"]
    }
    memory.log_short_term_interaction(test_user, "ASSISTANT_GENERATION_SUMMARY", gen_summary_data_cloud)
    
    memory.save_creation_summary(
        user_id=test_user, 
        original_prompt=gen_summary_data_cloud["original_prompt"],
        enhanced_prompt=gen_summary_data_cloud["enhanced_prompt"],
        image_path=gen_summary_data_cloud["image_path"],
        model_path=gen_summary_data_cloud["model_path"],
        tags=gen_summary_data_cloud["tags"]
    )


    print(f"\n--- Waiting for {SHORT_TERM_WINDOW_SECONDS + 2} seconds to pass short-term window ---")
    time.sleep(SHORT_TERM_WINDOW_SECONDS + 2)

    memory.log_short_term_interaction(test_user, "USER_INPUT", {"text": "Make the cloud sad"}) # This should be in new window
    
    # Retrieve recent (should only get the "Make the cloud sad" and its preceding assistant actions if any after window)
    recent_history = memory.get_recent_short_term_interactions(test_user, SHORT_TERM_WINDOW_SECONDS, limit=5)
    print(f"\nRecent history (should be short, after window passed for earlier items):")
    for item in recent_history:
        print(f"  {item['role']}: {item['content']}")

    assert len(recent_history) <= 2 * 2

    print(f"\n--- Archiving log entries older than {ARCHIVE_OLDER_THAN_SECONDS}s ---")
    deleted_stm_count = memory.archive_short_term_log(user_id=test_user, older_than_seconds=ARCHIVE_OLDER_THAN_SECONDS)
    print(f"Archived (deleted) {deleted_stm_count} entries from short-term log.")


    # Verify long-term memory for the cloud is still there
    cloud_creations_lt = memory.search_creations_keyword_date(search_term="cloud", user_id=test_user)
    print(f"\nCloud creations in long-term memory: {len(cloud_creations_lt)}")
    assert len(cloud_creations_lt) == 1 
    if cloud_creations_lt:
        print(f"  Found: {cloud_creations_lt[0]['original_prompt']}")

    memory.close()
    print("\n--- Short-Term Memory Test Logic Completed ---")