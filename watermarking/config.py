RANDOM_SEED = 42
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda"
TORCH_DTYPE = "float16"


DATASET_NAME = "databricks/databricks-dolly-15k"
N_PROMPTS = 1000

MAX_NEW_TOKENS = 400
MIN_NEW_TOKENS = 50       
TEMPERATURE = 1.0         
DO_SAMPLE = True          
BATCH_SIZE = 1            
MIN_DELTA_FRACTION = 0.5

CATEGORY_SAMPLES = {
    "open_qa":              200,
    "brainstorming":        200,
    "creative_writing":     200,
    "general_qa":           150,
    "information_extraction": 150,
    "closed_qa":            100,
}

KGW_GAMMA = 0.25          
KGW_DELTA = 2.0         
# KGW_SEEDING_SCHEME = "selfhash"
KGW_SEEDING_SCHEME = "minhash"
KGW_IGNORE_REPEATED_NGRAMS = True

SYNTHID_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 785, 900, 111, 222, 333, 444, 555, 666, 777, 888, 999, 101]
SYNTHID_NGRAM_LEN = 5
SYNTHID_SAMPLING_TABLE_SIZE = 2**16
SYNTHID_CONTEXT_HISTORY_SIZE = 1024

ENTROPY_SCALING = "linear"
ENTROPY_THRESHOLD = 0.3
ENTROPY_EPS = 1e-9

OUTPUT_DIR = "./outputs"
GENERATIONS_FILE = "generations.jsonl"
DETECTIONS_FILE  = "detections.jsonl"