import os
from openai import OpenAI

# Seting the API key
key_file = open('My OAI API Key.txt', 'r')
MY_OAI_API_KEY = key_file.readlines()[0]
key_file.close()

client = OpenAI(
    api_key = MY_OAI_API_KEY
)

# Function to analyse every book
def analyze_book(title, author, year, synopsis):
    # Create the prompt with the gathered data
    prompt = f"""
    Based on the following information about the book "{title}" by {author}, published in {year}, determine the following:
    1. When is the story set? (far past, near past, present, near future, far future, or other)
    2. How is the book’s tone? (pessimistic, optimistic, or neither)
    3. How is the book' setting? (utopic, dystopic, or neither)
    4. Is most of the story set on Earth? (yes or no)
    5. Are there aliens in the story? (yes or no)
    6. Are there robots in the story? (yes or no)
    Please, answer with only the words given as options for the questions.
    Book synopsis: {synopsis}"""
    
    # Call the OpenAI API with the crafted prompt
    ChatCompletion = client.chat.completions.create(
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes books based on provided information and your own knowledge about the book."},
            {"role": "user", "content": prompt}
        ],
        model = "gpt-4o-mini-2024-07-18",
        #model = "gpt-4o",
        max_tokens = 500,  # Adjust as necessary based on the detail needed
        temperature = 0.7  # Adjust for creativity vs. factual response balance
    )
    
    # Extract and print the response
    answer = ChatCompletion.choices[0].message.content
    print(answer)

# Example usage with data you have extracted
title = "Dune"
author = "Frank Herbert"
synopsis = "Set on the desert planet Arrakis, Dune is the story of the boy Paul Atreides, heir to a noble family tasked with ruling an inhospitable world where the only thing of value is the    spice    melange, a drug capable of extending life and enhancing consciousness. Coveted across the known universe, melange is a prize worth killing for...When House Atreides is betrayed, the destruction of Paul   s family will set the boy on a journey toward a destiny greater than he could ever have imagined. And as he evolves into the mysterious man known as Muad   Dib, he will bring to fruition humankind   s most ancient and unattainable dream."
year = "1965"

analyze_book(title, author, year, synopsis)