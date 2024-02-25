from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load pre-trained BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Load your text from a file or provide it as a list of strings
# Each string should contain the text corresponding to a label
# Initialize an empty dictionary to store text for each label
data = {}

# Load your text from a file
with open("1953_L_1.txt", "r") as file:
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

# Initialize an empty dictionary to store summaries for each label
summaries = {}

# Generate summaries for each label
for label, text in data.items():
    # Preprocess the text and add label prefix
    input_text = f"{label}: {text}"
    
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True).input_ids

    # Generate summary using BART model
    summary_ids = model.generate(input_ids)
    
    # Decode the summary tokens back to text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Store the summary for the label
    summaries[label] = summary_text

# Print or use the summaries as needed
for label, summary in summaries.items():
    print(f"Summary for label '{label}':")
    print(summary)
    print()
