# Config class
# TODO For future reference, make invalid states unrepresentable


class Config:
    dir_model_cache = "./ModelCache"

    qa_collection = "Data/PathologyQuestions.xlsx"
    qa_responses = "Results/ModelResponsesFastChatT5_Revision.xlsx"

    # Model and generation config
    models = {
        "fastchat-t5": {
            "temperature": 0.1,
            "top_p": 0.75,
            "top_k": 40,
            "num_beams": 4,
            "max_new_tokens": 256,
        },
    }

    # Set model here
    current_model = "fastchat-t5"

    # Rest of the gen params
    temperature = models[current_model]["temperature"]
    top_p = models[current_model]["top_p"]
    top_k = models[current_model]["top_k"]
    num_beams = models[current_model]["num_beams"]
    max_new_tokens = models[current_model]["max_new_tokens"]

    # Fragments separately
    num_fragments = 4

    # Instructions to generate embeddings
    note_instruction = "Represent the pathology report:"
    question_instruction = "Represent the pathology question:"
