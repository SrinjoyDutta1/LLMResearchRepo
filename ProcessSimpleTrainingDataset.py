import pandas as pd

# Load the training dataset
training_dataset_path = 'LLMResearch/dataset/SimpleTrainingDataset.csv'
data = pd.read_csv(training_dataset_path)

# Remove rows with category 0
filtered_data = data[data['Category'] != 0]

# Save the filtered dataset
filtered_dataset_path = 'LLMResearch/dataset/SimpleTrainingDataset.csv'
filtered_data.to_csv(filtered_dataset_path, index=False)

# Print the number of rows for each category from 1 to 9
for category in range(1, 10):
    count = filtered_data[filtered_data['Category'] == category].shape[0]
    print(f"Number of rows in category {category}: {count}")


# Load the dataset
file_path = 'LLMResearch/dataset/SimpleTrainingDataset.csv'
data = pd.read_csv(file_path)

# Initialize an empty DataFrame for the final dataset
final_dataset = pd.DataFrame()

# Process each category
for category in range(1, 10):
    category_data = data[data['Category'] == category]
    sample_size = min(len(category_data), 200)
    category_sample = category_data.sample(n=sample_size, random_state=42)
    final_dataset = pd.concat([final_dataset, category_sample], ignore_index=True)
    print(f"Number of rows in category {category}: {len(category_sample)}")

# Save the final dataset to a new CSV file
final_dataset_path = 'LLMResearch/dataset/SampledSimpleSelectTrainingDataset.csv'
final_dataset.to_csv(final_dataset_path, index=False)

print(f"Selected training dataset saved to {final_dataset_path}")
