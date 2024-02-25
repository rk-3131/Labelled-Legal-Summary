import os
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_file(input_file_path, output_folder_path, num_sentences=3):
    with open(input_file_path, "r") as file:
        lines = file.readlines()

    sentences = [line.split('\t')[0] for line in lines]
    labels = [line.split('\t')[1].strip() for line in lines]

    sentences_by_label = {}
    for sentence, label in zip(sentences, labels):
        if label not in sentences_by_label:
            sentences_by_label[label] = []
        sentences_by_label[label].append(sentence)

    summarizer = LsaSummarizer()

    file_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_folder_path, f"{file_name.split('.')[0]}.txt")

    with open(output_file_path, 'w') as output_file:
        for label, sentences in sentences_by_label.items():
            text = ' '.join(sentences)
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summary = summarizer(parser.document, num_sentences)
            output_file.write(f"Summary for label '{label}':\n")
            for sentence in summary:
                output_file.write(str(sentence) + "\n")
            output_file.write("\n")

def summarize_files_in_folder(input_folder_path, output_folder_path, num_sentences=3):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(input_folder_path):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder_path, filename)
            summarize_file(input_file_path, output_folder_path, num_sentences)

# Input folder path containing the text files
input_folder_path = "D:/Legal Document summarization/Existing repositories/Python Library/judgement"
# Output folder path to store the summaries
output_folder_path = "D:/Legal Document summarization/Existing repositories/Python Library/Summy Summary"

summarize_files_in_folder(input_folder_path, output_folder_path, 3)
