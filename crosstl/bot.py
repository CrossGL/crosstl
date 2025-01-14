from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pretrained chatbot model and tokenizer
model_name = "microsoft/DialoGPT-medium"  # imported a medium version of 'DialoGPT'.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name
)  # Creating an Instance of the Model.


def chat_with_bot(user_input, chat_history_ids=None):
    """
    Generates a response from the chatbot based on user input and the conversation history.

    Args:
        user_input (str): The user's input message to the chatbot.
        chat_history_ids (torch.Tensor, optional): A tensor containing the conversation history
                                                   (i.e., previous exchanges between the user and the bot).
                                                   If None, the conversation history is initialized as empty.

    Returns:
        tuple: A tuple containing:
            - response (str): The chatbot's response to the user input.
            - chat_history_ids (torch.Tensor): Updated conversation history after the new input and response.

    The function encodes the user's input, appends it to the conversation history (if available),
    and then generates a response using the model. The response is decoded and returned as a string.
    The conversation history is updated for future interactions.

    Example:
        user_input = "Hello, how are you?"
        response, chat_history = chat_with_bot(user_input)
    """
    # Taking User input and converting it to Tensors:
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt"
    )
    bot_input_ids = (
        new_input_ids
        if chat_history_ids is None
        else torch.cat([chat_history_ids, new_input_ids], dim=-1)
    )

    # Generate simple response:
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
    )  # Limiting the Output Text to atmost 1000 words.
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True
    )
    return response, chat_history_ids


# Example Evaluation:
user_input = "HI! How are you bot?"
response, chat_history_ids = chat_with_bot(user_input)
print("Bot:", response)
