import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generate_summary(input_folder_path, output_folder_path, num_sentences=3):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder_path, filename)
            with open(file_path, "r") as file:
                # Initialize a dictionary to store sentences for each label in the current file
                data = {}

                for line in file:
                    # Split each line by tab to separate the sentence and label
                    sentence, label = line.strip().split('\t')

                    # Remove any leading or trailing spaces from the label
                    label = label.strip()

                    # Check if the label already exists in the dictionary
                    if label in data:
                        # If the label exists, append the sentence to its corresponding text
                        data[label].append(sentence)
                    else:
                        # If the label does not exist, create a new entry in the dictionary
                        data[label] = [sentence]

                # Initialize a TF-IDF vectorizer
                vectorizer = TfidfVectorizer()

                # Initialize an empty string to store summaries for the current file
                file_summary = ""

                # Generate summaries for each label in the current file
                for label, sentences in data.items():
                    # Concatenate all sentences for the label into a single string
                    text = ' '.join(sentences)

                    # Fit the vectorizer and transform the text
                    tfidf_matrix = vectorizer.fit_transform([text])

                    # Calculate cosine similarity matrix
                    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

                    # Sort sentences by their TF-IDF scores
                    scores = [(i, score) for i, score in enumerate(similarity_matrix[0])]
                    scores = sorted(scores, key=lambda x: x[1], reverse=True)

                    # Select the top sentences for the summary
                    top_sentences = sorted(scores[:num_sentences])

                    # Generate the summary for the current label
                    label_summary = ' '.join([text.split('.')[i] for i, _ in top_sentences])

                    # Append the label summary to the file summary
                    file_summary += f"Summary for label '{label}':\n{label_summary}\n\n"

                # Write the file summary to a file in the output folder
                output_file_path = os.path.join(output_folder_path, f"{filename.split('.')[0]}.txt")
                with open(output_file_path, 'w') as output_file:
                    output_file.write(file_summary)

                print(f"Summary for file '{filename}' has been written to {output_file_path}")

# Input folder path containing the text files
input_folder_path = "D:/Legal Document summarization/Existing repositories/Python Library/judgement"
# Output folder path to store the summaries
output_folder_path = "D:/Legal Document summarization/Existing repositories/Python Library/general Summary"
# Number of sentences in the summary
num_sentences = 3

generate_summary(input_folder_path, output_folder_path, num_sentences)
