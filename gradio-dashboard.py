import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr


books = pd.read_csv("data/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&file=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "no-cover.png",
    books["large_thumbnail"]
)

raw_documents = TextLoader("data/tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size = 1, chunk_overlap = 0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

model_name = "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name
)

vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embeddings
)

def retrieve_semantic_recommendations(
    query, 
    category = None, 
    tone = None,
    initial_top_k = 50,
    final_top_k = 16
):
    recs = vectorstore.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)
    if category != "All":
        book_recs = book_recs[book_recs["simple_category"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)
    
    if tone != "All":
        tone_mapping = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspensful": "fear",
            "Sad": "sadness",
            "Disgusting (experimental)": "disgust"
        }
        searching_tone = tone_mapping[tone]
        book_recs.sort_values(by=searching_tone, ascending=False, inplace=True)

    return book_recs

def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        author_split = row["authors"].split(";")
        if len(author_split) == 2:
            authors_str = f"{author_split[0]} and {author_split[1]}"
        elif len(author_split) > 2:
            authors_str = f"{', '.join(author_split[:-1])}, and {author_split[-1]}"
        else:
            authors_str = row["authors"]
        
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_category"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspensful", "Sad", "Disgusting (experimental)"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book: ", 
                                placeholder = "e.g., A story about a happy day")
        category_dropdown = gr.Dropdown(choices=categories, label = "Select a cetegory: ", value = "All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone: ", value = "All")
        submit_button = gr.Button("Find recommendations")
    
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)
    submit_button.click(fn = recommend_books, 
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()

