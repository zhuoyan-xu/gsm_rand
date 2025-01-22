from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from pdb import set_trace as pds

    
class LlamaCompletion:
    def __init__(self, model_name = "meta-llama/Llama-3.2-3B-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Before generation, set the pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.completion_tokens = 0
        self.prompt_tokens = 0

    def completions(self, question="user query",temperature=0.7, max_tokens=1000, n=1, stop=None):

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},  # Default system message
            {"role": "user", "content": question}
        ]       

        # Format messages into a prompt
        prompt = ""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate responses
        with torch.no_grad():
            outputs = []
            for _ in range(n):
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                # Remove the prompt from the response
                response = decoded[len(prompt):].strip()
                outputs.append(response)

            # Track token usage
            self.prompt_tokens += len(inputs.input_ids[0])
            self.completion_tokens += len(output[0]) - len(inputs.input_ids[0])
            # Create a response object similar to OpenAI's format
            return type('Response', (), {
                'outputs': outputs,
                'usage': type('Usage', (), {
                    'completion_tokens': self.completion_tokens,
                    'prompt_tokens': self.prompt_tokens
                })
            })


