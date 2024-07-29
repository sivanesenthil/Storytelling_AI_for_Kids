import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import base64
st.set_page_config(
    page_title="Storytelling AI for kids",
    page_icon="🌟",
    layout="wide"
)
# Function to load a local GIF file and convert it to base64
def load_gif_base64(file_path):
    with open(file_path, "rb") as gif_file:
        gif_base64 = base64.b64encode(gif_file.read()).decode("utf-8")
    return gif_base64

# Path to your local GIF file
gif_path = r"C:\\Users\\SHYNI\\Storytelling_AI_for_Kids\\title.gif"
gif_base64 = load_gif_base64(gif_path)
model_save_path = './final_model'
model = GPT2LMHeadModel.from_pretrained(model_save_path)
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_with_pad")

def generate_response(prompt, model, tokenizer, max_length=300, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs['input_ids'], max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p,  repetition_penalty=repetition_penalty, no_repeat_ngram_size=2,pad_token_id=tokenizer.eos_token_id,do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Custom CSS for hover effect and styling
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+Bhai+2&display=swap');
html, body {
        height: 100%;
        margin: 0;
        padding: 0;
}
.header {
        display: flex;
        align-items: center;
        justify-content: center;
}
.title {
        text-align: center;
        font-size: 4.5em;
        font-family: 'Baloo Bhai 2', cursive;
}
.gif {
        width: 100px;  
        height: 100px;  
}
.centered-text {
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 0px;
        margin-right: 10%;
        padding-left: 80px;
        padding-bottom: 0px;
        font-family: 'Baloo Bhai 2', cursive;
}
.main .block-container {
        padding-top:0;
        padding-bottom:10;
        padding-left:10;
        padding-right: 10;
}
.stApp {
        background-image: url("https://img.freepik.com/premium-vector/seamless-baby-animal-style-outline-pattern-vector-illustration_611054-1818.jpg?w=740");
        background-size: cover;
}
.genre-card {
    width: 180px;
    height: 250px;
    display: inline-block;
    margin: 10px;
    border-radius: 10px;
    overflow: hidden;
    transition: transform 0.3s ease;
    cursor: pointer;
    text-align: center;
    background-color: #f0f0f0;
}
.genre-card img {
    width: 100%;
    height: 70%;
    object-fit: cover;
}
.genre-card:hover {
    transform: scale(1.1);
}
.genre-card span {
    display: block;
    padding: 10px 0;
    font-size: 18px;
    font-family: 'Baloo Bhai 2', cursive;
    color: black;
}
 div.stButton > button {
                width: 100%;
                height: 60px;
                font-size: 20px;
                margin: 10px 0;
                background-color:white ; 
                color: black;
                border: none;
                border-radius: 5px;
                cursor: pointer;
}
div.stButton > button:hover {
                background-color: #f0f0f0; 
}
@keyframes fadeIn {
        0% { opacity: 0; }
        10% { opacity: 1; }
    }
</style>
"""

genres = {
   
    "Fairy Tale 🧚‍♀️": "https://img.freepik.com/premium-vector/cute-fairy-blue-dress-girly-cartoon-character_178650-10278.jpg?w=740",
    "Adventure 🏕️": "https://img.freepik.com/premium-vector/children-s-adventure-concept-cute-kids-walking-hiking-exploring-discovering-world-together-curious-boy-girl-with-map-colored-flat-graphic-vector-illustration-isolated-white-background_198278-10794.jpg",
    "Fantasy 🧙‍♂️": "https://img.freepik.com/premium-vector/witch-flying-broomstick-cartoon-style-helloween-magic-concept-trendy-modern-vector-illustration-isolated-white-background-hand-drawn-flat-design_257471-439.jpg?w=740",
    "Sci-Fi 🚀": "https://img.freepik.com/free-psd/hand-drawn-astronaut-isolated_23-2151558224.jpg?t=st=1721120472~exp=1721124072~hmac=e0aa851c4a7cf14bf3fa78ccbe44f2cfbbf58e810eb5f86c3e15e85c64a69f10&w=740",
    "Mystery 🕵️‍♂️": "https://img.freepik.com/premium-vector/detective-character-investigation_884500-15877.jpg",
    "Animal Tale 🐯": "https://img.freepik.com/premium-photo/2d-cute-cartoon-jackel-animal-2d-cartoon-with-sharp-outlines-white-background_876282-3793.jpg",
    "Fable 📜": "https://img.freepik.com/premium-photo/cute-cartoon-bull-grass-kids-story-book-illustration-your-design_925324-5464.jpg",
    "Mythology 🏛️": "https://img.freepik.com/free-vector/hand-drawn-flat-design-greek-mythology-illustration_23-2149373320.jpg",
    "Historical Fiction 🏰": "https://img.freepik.com/premium-vector/cute-primitive-caveman-holding-cudgel-cartoon-icon-clipart-illustration_422763-980.jpg",
    "Humor 😂": "https://img.freepik.com/premium-vector/walking-cute-cartoon-dogs-funny-dogs-white-background-furry-human-friends-home-animals_177886-812.jpg",
    "Friendship 🤝": "https://as1.ftcdn.net/v2/jpg/02/77/29/96/1000_F_277299622_W0yvnmSHc7JJdtokKSYt1QTZf6Dqu09G.jpg",
    "Superheroes 🦸‍♂️": "https://img.freepik.com/premium-vector/cute-super-hero-flying-cartoon-illustration-people-profession-icon-concept_138676-1911.jpg",
    "Sports 🏅": "https://img.freepik.com/premium-vector/cute-boy-playing-soccer-cartoon-vector-illustration-white-background_1142-84572.jpg",
    "Holidays 🎉": "https://img.freepik.com/premium-vector/cute-cartoon-santa-claus-sleigh-with-christmas-gifts_214720-629.jpg",
    "Bedtime 🌙": "https://img.freepik.com/free-vector/scene-with-little-girl-sleeping-with-pink-teddy-bear_1308-44135.jpg"
}

genre_prompts = {
    "Fairy Tale 🧚‍♀️": "a brave knight, a clever princess, a talking animal or anything",
    "Adventure 🏕️": "a thrilling journey, a treasure hunt, a daring expedition or anything",
    "Fantasy 🧙‍♂️": "elves, dragons, wizards or anything",
    "Sci-Fi 🚀": "distant planets, encounter aliens, dive into futuristic technology or anything",
    "Mystery 🕵️‍♂️": "uncover a hidden treasure, solve a crime, reveal a secret or anything",
    "Animal Tale 🐯": "a wise owl, a brave lion, a mischievous monkey or anything",
    "Fable 📜": "wisdom, kindness, perseverance or anything",
    "Mythology 🏛️": "Greek, Norse, Egyptian or anything",
    "Historical Fiction 🏰": "ancient civilizations, medieval kingdoms, the roaring twenties or anything",
    "Humor 😂": "puns, slapstick comedy, witty banter or anything",
    "Friendship 🤝": "loyalty, compassion, teamwork or anything",
    "Superheroes 🦸‍♂️": "flight, super strength, invisibility or anything",
    "Sports 🏅": "soccer, basketball, swimming or anything",
    "Holidays 🎉": "Halloween, Christmas, New Year's Eve or anything",
    "Bedtime 🌙": "dreamlands, whispering forests, starlit skies or anything"
}

fallback_prompts = {
    "Fairy Tale 🧚‍♀️": "Once upon a time in a magical kingdom, there was a ",
    "Adventure 🏕️": "In a land far away, a brave explorer set out on a thrilling journey. Along the way, they encountered ",
    "Fantasy 🧙‍♂️": "In a world of magic and wonder, there lived ",
    "Sci-Fi 🚀": "In the distant future, humanity discovered ",
    "Mystery 🕵️‍♂️": "In a small town, a detective was called to investigate a strange occurrence. It all began when ",
    "Animal Tale 🐯": "In a lush forest, the animals were always up to something. One day, ",
    "Fable 📜": "Long ago, in a village, there lived a wise old ",
    "Mythology 🏛️": "In ancient times, the gods and heroes of legend often walked among mortals. One such legend tells of ",
    "Historical Fiction 🏰": "During the era of the great empires, a young person dreamed of ",
    "Humor 😂": "In a quirky town, the funniest thing happened to ",
    "Friendship 🤝": "In a bustling city, two best friends discovered ",
    "Superheroes 🦸‍♂️": "In a metropolis protected by heroes, one day a new hero emerged with the power of ",
    "Sports 🏅": "On the day of the big game, the team was ready to ",
    "Holidays 🎉": "During the festive season, everyone was excited about ",
    "Bedtime 🌙": "As the stars twinkled in the night sky, a child drifted off to sleep and dreamed of "
}

# Define the interaction options
interaction_options = {
    "Fairy Tale 🧚‍♀️": [
        "The character encounters a magical creature.",
        "The character makes a new friend.",
        "The character receives a magical gift.",
        "Summarize the story and finish."
    ],
    "Adventure 🏕️": [
        "The character faces a dangerous obstacle.",
        "The character meets a helpful guide.",
        "The character finds a mysterious map.",
        "Summarize the story and finish."
    ],
    "Fantasy 🧙‍♂️": [
        "The character learns a powerful spell.",
        "The character meets a wise wizard.",
        "The character discovers a hidden kingdom.",
        "Summarize the story and finish."
    ],
    "Sci-Fi 🚀": [
        "The character encounters an alien.",
        "The character discovers a new planet.",
        "The character invents a futuristic gadget.",
        "Summarize the story and finish."
    ],
    "Mystery 🕵️‍♂️": [
        "The character finds a hidden clue.",
        "The character meets a mysterious stranger.",
        "The character uncovers a secret.",
        "Summarize the story and finish."
    ],
    "Animal Tale 🐯": [
        "The animal makes a new friend.",
        "The animal faces a challenge in the wild.",
        "The animal discovers something new.",
        "Summarize the story and finish."
    ],
    "Fable 📜": [
        "The character learns an important lesson.",
        "The character helps someone in need.",
        "The character makes a wise decision.",
        "Summarize the story and finish."
    ],
    "Mythology 🏛️": [
        "The character meets a god or goddess.",
        "The character goes on a heroic quest.",
        "The character discovers a divine artifact.",
        "Summarize the story and finish."
    ],
    "Historical Fiction 🏰": [
        "The character witnesses a historical event.",
        "The character meets a famous historical figure.",
        "The character discovers an ancient relic.",
        "Summarize the story and finish."
    ],
    "Humor 😂": [
        "The character plays a funny prank.",
        "The character finds themselves in a silly situation.",
        "The character makes everyone laugh.",
        "Summarize the story and finish."
    ],
    "Friendship 🤝": [
        "The character helps their friend.",
        "The character makes a new friend.",
        "The character and their friend go on an adventure.",
        "Summarize the story and finish."
    ],
    "Superheroes 🦸‍♂️": [
        "The superhero saves the day.",
        "The superhero gains a new power.",
        "The superhero meets a new ally.",
        "Summarize the story and finish."
    ],
    "Sports 🏅": [
        "The character wins a big game.",
        "The character trains hard for a competition.",
        "The character makes a great play.",
        "Summarize the story and finish."
    ],
    "Holidays 🎉": [
        "The character celebrates with family and friends.",
        "The character receives a special gift.",
        "The character participates in a holiday tradition.",
        "Summarize the story and finish."
    ],
    "Bedtime 🌙": [
        "The character has a magical dream.",
        "The character meets a friendly nighttime creature.",
        "The character goes on a nighttime adventure.",
        "Summarize the story and finish."
    ]
}
def handle_interactive_story_continuation(prompt, model, tokenizer, max_length=300, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    next_story_part = generate_response(prompt, model, tokenizer, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
    return next_story_part

def interactive_storytelling():
    # Initialize session state variables
    if 'story_parts' not in st.session_state:
        st.session_state.story_parts = []
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = ""
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    if 'summarized' not in st.session_state:
        st.session_state.summarized = False
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    st.markdown(custom_css, unsafe_allow_html=True)
    st.write(f"""
    <div class="header">
        <h1 class="title">Storytelling AI for Kids</h1>
        <img src="data:image/gif;base64,{gif_base64}" alt="GIF" class="gif">
    </div>
    """, unsafe_allow_html=True)

    st.write('<div class="centered-text">Explore the magic of stories! Choose a genre, provide a character or theme, and let the adventure begin!</div>', unsafe_allow_html=True)

    # Get the current page state from the URL
    current_page = st.experimental_get_query_params().get("page", ["home"])[0]

    if current_page == "home":
        cols = st.columns(5)

        for i, (genre, img_url) in enumerate(genres.items()):
            with cols[i % 5]:
                st.markdown(f'<div class="genre-card"><a href="?page=story&genre={genre}"><img src="{img_url}" alt="{genre}"><span>{genre}</span></a></div>', unsafe_allow_html=True)

    elif current_page == "story":
        genre_selected = st.experimental_get_query_params().get("genre", [None])[0]
        if genre_selected:
            st.write(f"Wonderful! Do you want a story about {genre_prompts[genre_selected]}?")
            specific_choice = st.text_input("Describe your choice:")
            
            if specific_choice and not st.session_state.summarized:
                if not st.session_state.story_parts:
                    initial_prompt = f"Once upon a time, in a magical land, there was a {specific_choice} who"
                    if specific_choice not in genre_prompts[genre_selected]:
                        initial_prompt = f"{fallback_prompts[genre_selected]}{specific_choice} who"
                    story = generate_response(initial_prompt, model, tokenizer)
                    st.session_state.current_prompt = story
                    st.session_state.story_parts.append(story)

                # Display all story parts
                for part in st.session_state.story_parts:
                    st.write(part)

                # Create buttons
                col1, col2, col3, col4 = st.columns(4)
                next_step = None
              
                with col1:
                    if st.button(interaction_options[genre_selected][0], key=f"option1_{len(st.session_state.story_parts)}"):
                        next_step = interaction_options[genre_selected][0]
                        st.session_state.button_clicked = True
                with col2:
                    if st.button(interaction_options[genre_selected][1], key=f"option2_{len(st.session_state.story_parts)}"):
                        next_step = interaction_options[genre_selected][1]
                        st.session_state.button_clicked = True
                with col3:
                    if st.button(interaction_options[genre_selected][2], key=f"option3_{len(st.session_state.story_parts)}"):
                        next_step = interaction_options[genre_selected][2]
                        st.session_state.button_clicked = True
                with col4:
                    if st.button(interaction_options[genre_selected][3], key=f"summarize_{len(st.session_state.story_parts)}"):
                        next_step = interaction_options[genre_selected][3]
                        st.session_state.button_clicked = True

                # Process button click
                if st.session_state.button_clicked:
                    if next_step == "Summarize the story and finish.":
                        st.session_state.summary = " ".join(st.session_state.story_parts)
                        st.session_state.summarized = True
                    else:
                        # Generate the next story segment based on the user's choice
                        new_prompt = f"{st.session_state.current_prompt} {next_step}"
                        next_story_part = handle_interactive_story_continuation(new_prompt, model, tokenizer)
                        st.session_state.current_prompt = next_story_part  # Update the prompt with the new story part
                        st.session_state.story_parts.append(next_story_part)
                    
                    st.session_state.button_clicked = False
                    st.experimental_rerun()

            # Display summary if the story has been summarized
            if st.session_state.summarized:
                st.write(f"\nHere is the summary of your story:\n\n{st.session_state.summary}")
                if st.button("Start a New Story"):
                    # Reset all states for a new story
                    st.session_state.story_parts = []
                    st.session_state.current_prompt = ""
                    st.session_state.summarized = False
                    st.session_state.summary = ""
                    # Redirect to home page
                    st.experimental_set_query_params(page="home")
                    st.experimental_rerun()

if __name__ == "__main__":
    interactive_storytelling()
