# Interface with a local model in batches

import time
import pathlib
import textwrap

import fitz
import tqdm
import torch

import pandas as pd
import numpy as np

from sklearn import metrics
from langchain.text_splitter import CharacterTextSplitter
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers import AutoModel, GenerationConfig

from config import Config


class Models:
    @classmethod
    def model_loader_fastchat_t5(cls):
        # The model is derived from flan-t5-xl

        # Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(
            "lmsys/fastchat-t5-3b-v1.0", cache_dir=Config.dir_model_cache
        )

        # Model
        model = T5ForConditionalGeneration.from_pretrained(
            "lmsys/fastchat-t5-3b-v1.0",
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map={"": 0},  # Required for multi-GPU systems
            cache_dir=Config.dir_model_cache,
        )

        # Just in case
        model.eval()

        return model, tokenizer

    @classmethod
    def load_model(cls):
        return cls.model_loader_fastchat_t5()


class PromptGenerator:
    @classmethod
    def alpaca_base(cls, question, fragments):  # Doesn't work all that well
        str_fragments = "\n".join(fragments)
        prompt = textwrap.dedent(
            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {question}

    ### Input:
    {str_fragments}

    ### Response:"""
        )

        return prompt

    @classmethod
    def langchain_base(cls, question, fragments):
        str_fragments = "\n".join(fragments)
        prompt = textwrap.dedent(
            f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{str_fragments}

Question: {question}
Response:"""
        )

        return prompt

    @classmethod
    def generate_prompt(cls, question, fragments):
        return cls.langchain_base(question, fragments)


class BatchedChartReview:
    @classmethod
    def read_report(cls, report_path):
        document = fitz.open(report_path)
        document_path = pathlib.Path(report_path)
        study_date = document_path.stem.split("_")[1]

        report_text = ""
        for page in document:
            report_text += page.get_text()

        return report_text, study_date

    # Instance methods
    def __init__(self):
        # Text splitter for splitting... text.
        # Values for FastChat: chunk_size=500, chunk_overlap=100
        self.text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=600, chunk_overlap=100, length_function=len
        )

        # Load QA list
        self.df_reports = pd.read_excel(
            Config.qa_collection, sheet_name="pathology_reports"
        )
        df_questions = pd.read_excel(Config.qa_collection, sheet_name="questions")
        self.questions = df_questions["question"].tolist()

        # Embedding model - runs on CPU
        self.instruct_embedding_model = AutoModel.from_pretrained(
            "nvidia/NV-Embed-v2",
            trust_remote_code=True,
            cache_dir=Config.dir_model_cache,
        )
        self.instruct_embedding_model.cuda()

    def split_notes(self, note):
        assert isinstance(note, str), "Note must be a string"

        # Split the note into chunks and add order identifiers
        all_reports = []
        subnotes = self.text_splitter.split_text(note)
        for lower_count, sub_report in enumerate(subnotes):
            sub_report = f"Part {lower_count + 1}: {sub_report}"
            all_reports.append(sub_report)

        return all_reports

    def generate_corpus_embeddings(self, notes):
        # Encode notes with instruction and max_length
        embedding_pairs = []
        for note_fragment in notes:
            note_fragment = note_fragment.replace("\n", " ")

            # Use Config.note_instruction for the instruction and the note_fragment as the prompt
            these_embeddings = self.instruct_embedding_model.encode(
                prompts=[note_fragment],  # Prompts is the actual text to encode
                instruction=Config.note_instruction,  # Instruction as per the new method signature
                max_length=32768,  # Max length as specified
            )

            # Move the embeddings to CPU and convert to NumPy
            these_embeddings = these_embeddings.cpu().numpy()

            embedding_pairs.append((note_fragment, these_embeddings))

        # Convert embeddings to a NumPy array
        corpus_embeddings = np.array([embedding for _, embedding in embedding_pairs])

        return embedding_pairs, corpus_embeddings

    def similarity_search(self, embedding_pairs, corpus_embeddings, question):
        # Encode question
        question = question.replace("\n", " ")

        # Concatenate instruction and question as a single string
        question_input = Config.question_instruction + " " + question

        # Encode the question
        question_embeddings = self.instruct_embedding_model.encode(
            prompts=[question_input],  # The prompt with the instruction + question
            max_length=32768,
        )

        # Move the question embeddings to CPU and convert to NumPy (if it's still a tensor)
        question_embeddings = (
            question_embeddings.cpu().numpy()
        )  # Move tensor to CPU and convert to NumPy if needed

        # Ensure corpus_embeddings is a NumPy array (it should already be)
        if not isinstance(corpus_embeddings, np.ndarray):
            corpus_embeddings = (
                corpus_embeddings.squeeze(1).cpu().numpy()
            )  # Convert to NumPy if it's still a tensor

        # Reshape question_embeddings and corpus_embeddings to be 2D
        # If they have 3 dimensions (e.g., [batch_size, seq_length, embedding_dim]),
        # we reduce them to [batch_size, embedding_dim] by taking mean or max across seq_length
        if len(question_embeddings.shape) == 3:
            question_embeddings = question_embeddings.mean(
                axis=1
            )

        if len(corpus_embeddings.shape) == 3:
            corpus_embeddings = corpus_embeddings.mean(
                axis=1
            )  # Similarly, reduce corpus embeddings to 2D

        # Calculate cosine_similarity between question and each note fragment
        similarity_scores = metrics.pairwise.cosine_similarity(
            question_embeddings, corpus_embeddings
        )

        # Get the top most similar fragments
        fragment_idx = similarity_scores.squeeze().argsort()[-Config.num_fragments :][
            ::-1
        ]

        # Get corresponding text
        fragments = [embedding_pairs[fragment][0] for fragment in fragment_idx]

        return fragments

    def question_answer(self, model, tokenizer, report):
        # Generator config
        generation_config = GenerationConfig(
            temperature=Config.temperature,
            top_p=Config.top_p,
            top_k=Config.top_k,
            do_sample=True,
            # num_beams=Config.num_beams,  # Beam search is a little dicey for some models
        )
        # Split the report into chunks and get the embeddings
        report_chunks = self.split_notes(report)
        embedding_pairs, corpus_embeddings = self.generate_corpus_embeddings(
            report_chunks
        )

        all_responses = {
            "report_text": [],
            "question": [],
            "response": [],
            "time_taken": [],
        }
        for question in self.questions:

            # Generate prompt
            prompt = PromptGenerator.generate_prompt(question, report)
            inputs = tokenizer(prompt, return_tensors="pt")

            # If the input is too long, go down the split route
            if (inputs["input_ids"].shape[1] - Config.max_new_tokens) > 2048:

                # Get the most similar chunks
                # NOTE Rename from report_chunks likely not required - just making sure it works
                relevant_chunks = self.similarity_search(
                    embedding_pairs, corpus_embeddings, question
                )

                # Generate prompt
                prompt = PromptGenerator.generate_prompt(question, relevant_chunks)
                inputs = tokenizer(prompt, return_tensors="pt")

            # Move to GPU
            input_ids = inputs["input_ids"].cuda()

            # Time taken for this response
            start_time = time.time()

            # Generation without streaming
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=Config.max_new_tokens,
                )
            response = tokenizer.decode(generation_output.sequences[0])
            response = " ".join(response.split())

            # Print as usual
            print(f'Prompt: {prompt}')
            print(f"Question: {question}")
            print(f"Response: {response}")

            # Print a red line
            print("\033[91m" + "-" * 100 + "\033[0m")

            all_responses["report_text"].append(report)
            all_responses["question"].append(question)
            all_responses["response"].append(response)
            all_responses["time_taken"].append(time.time() - start_time)

        df_responses = pd.DataFrame(all_responses)

        return df_responses

    def hammer_time(self):
        # Create langchain pipeline
        model, tokenizer = Models.load_model()

        # Time to hammer
        start_time = time.time()

        # Iterate through the QA list by subjectid
        responses_across_experiments = []
        for report in tqdm.tqdm(self.df_reports["PATH_REPORT"]):
            df_responses = self.question_answer(model, tokenizer, report)
            responses_across_experiments.append(df_responses)

        # Print time taken
        print(f"Total time taken: {time.time() - start_time} seconds")

        if responses_across_experiments:
            df_responses = pd.concat(responses_across_experiments, ignore_index=True)
            df_responses.to_excel(Config.qa_responses, index=False)
