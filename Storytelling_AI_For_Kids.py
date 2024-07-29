import re
from datasets import Dataset, DatasetDict
import torch
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Read the dataset
def read_dataset(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the content by genre
    pattern = r'<GENRE: (.*?)>\n(.*?)\nMoral: (.*?)\n'
    matches = re.findall(pattern, content, re.DOTALL)
    
    data = []
    for match in matches:
        genre, story, moral = match
        data.append({
            'genre': genre.strip(),
            'story': story.strip(),
            'moral': moral.strip()
        })
    
    return data

# Tokenize the dataset
def tokenize_data(data, tokenizer):
    tokenized_data = {'input_ids': [], 'attention_mask': []}
    for entry in data:
        input_text = f"<GENRE: {entry['genre']}>\n{entry['story']}\nMoral: {entry['moral']}\n"
        tokenized_input = tokenizer(input_text, padding='max_length', truncation=True, max_length=512)
        tokenized_data['input_ids'].append(tokenized_input['input_ids'])
        tokenized_data['attention_mask'].append(tokenized_input['attention_mask'])
    
    return tokenized_data

# Load and preprocess the dataset
file_path = 'stories.txt'
data = read_dataset(file_path)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the data
tokenized_data = tokenize_data(data, tokenizer)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_dict(tokenized_data)

# Save the processed dataset
dataset.save_to_disk('processed_stories_dataset')

print("Dataset prepared and saved to 'processed_stories_dataset'")


# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to read the dataset
def read_dataset(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the content by genre
    pattern = r'<GENRE: (.*?)>\n(.*?)\nMoral: (.*?)\n'
    matches = re.findall(pattern, content, re.DOTALL)
    
    data = []
    for match in matches:
        genre, story, moral = match
        data.append({
            'genre': genre.strip(),
            'story': story.strip(),
            'moral': moral.strip()
        })
    
    return data

# Function to generate a story using GPT-3/GPT-4
def generate_story(genre, story, moral):
    prompt = f"<GENRE: {genre}>\n{story}\nMoral: {moral}\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load and preprocess the dataset
file_path = 'stories.txt'
data = read_dataset(file_path)

# Generate stories for each entry in the dataset
for entry in data:
    genre = entry['genre']
    story = entry['story']
    moral = entry['moral']
    
    generated_story = generate_story(genre, story, moral)
    print(f"Genre: {genre}")
    print(f"Generated Story: {generated_story}")
    print(f"Moral: {moral}\n")

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepare the dataset
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

# Fine-tuning parameters
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust this value
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    save_steps=10_000,
    save_total_limit=2,
)


# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Load the dataset
train_dataset = load_dataset('stories.txt', tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model('./fine-tuned-gpt2')
# 1. Create the Streamlit App File

# Function to generate response based on prompt
def generate_response(prompt, model, tokenizer, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs['input_ids'], max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Interactive storytelling with Streamlit
def interactive_storytelling():
    st.title("Interactive Storytelling")

    genre_prompts = {
        "1": ("fairy tale", "a brave knight, a clever princess, or a talking animal"),
        "2": ("adventure", "a thrilling journey, a treasure hunt, or a daring expedition"),
        "3": ("fantasy", "elves, dragons, or wizards"),
        "4": ("sci-fi", "distant planets, encounter aliens, or dive into futuristic technology"),
        "5": ("mystery", "uncover a hidden treasure, solve a crime, or reveal a secret"),
        "6": ("animal tale", "a wise owl, a brave lion, or a mischievous monkey"),
        "7": ("fable", "wisdom, kindness, or perseverance"),
        "8": ("mythology", "Greek, Norse, or Egyptian"),
        "9": ("historical fiction", "ancient civilizations, medieval kingdoms, or the roaring twenties"),
        "10": ("humor", "puns, slapstick comedy, or witty banter"),
        "11": ("friendship", "loyalty, compassion, or teamwork"),
        "12": ("superheroes", "flight, super strength, or invisibility"),
        "13": ("sports", "soccer, basketball, or swimming"),
        "14": ("holidays", "Halloween, Christmas, or New Year's Eve"),
        "15": ("bedtime", "dreamlands, whispering forests, or starlit skies")
    }

    genre = st.selectbox("What story do you want to hear today?", list(genre_prompts.keys()), format_func=lambda x: genre_prompts[x][0].capitalize())

    if genre:
        st.write(f"Wonderful! Do you want a story about {genre_prompts[genre][1]}?")
        specific_choice = st.text_input("Describe your specific choice:")

        if specific_choice:
            prompt = f"<GENRE: {genre_prompts[genre][0]}> Once upon a time, in a magical land, there was a {specific_choice} who"
            story_parts = [prompt]
            story = generate_response(prompt, model, tokenizer)
            st.write(story)

            while True:
                st.write("\nWhat happens next?")
                next_choice = st.radio(
                    "Choose an option:",
                    ["The character encounters a challenge.", "The character makes a new friend.", "The character discovers something amazing.", "Summarize the story and finish."]
                )

                if next_choice == "The character encounters a challenge.":
                    prompt = f"The {specific_choice} faced a great challenge. It was..."
                    story_parts.append(prompt)
                elif next_choice == "The character makes a new friend.":
                    prompt = f"The {specific_choice} made a new friend. This friend was..."
                    story_parts.append(prompt)
                elif next_choice == "The character discovers something amazing.":
                    prompt = f"The {specific_choice} discovered something amazing. It was..."
                    story_parts.append(prompt)
                elif next_choice == "Summarize the story and finish.":
                    st.write("Summarizing the story and finishing it.")
                    story_parts.append(story)
                    summary = " ".join(story_parts)
                    st.write(f"\nHere is the summary of your story:\n\n{summary}")
                    break

                # Generate the next part of the story
                story = generate_response(prompt, model, tokenizer)
                st.write(story)

if __name__ == "__main__":
    interactive_storytelling()


