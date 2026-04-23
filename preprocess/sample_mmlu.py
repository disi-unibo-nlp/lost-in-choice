from datasets import load_dataset, concatenate_datasets

# 1. Load the dataset
ds = load_dataset('TIGER-Lab/MMLU-Pro', split="test")

# 2. Filter for exactly 10 options
ds_10opt = ds.filter(lambda example: len(example['options']) == 10)

# Target categories and configuration
target_categories = ['computer science', 'math', 'health', 'law']
samples_per_category = 250
seed = 42 # Set seed for reproducibility
sample_data_per_category = False

sampled_datasets = []

# 3. Iterate through each category to filter, shuffle, and sample
for category in target_categories:
    print(f"Processing category: {category}...")
    
    # Filter dataset for the current category
    ds_cat = ds_10opt.filter(lambda example: example['category'] == category)
    print(f"Found {len(ds_cat)} samples in category '{category}'.")
    
    if sample_data_per_category:
        # Check if we have enough samples (just to be safe)
        if len(ds_cat) < samples_per_category:
            raise ValueError(f"Category '{category}' only has {len(ds_cat)} samples, which is less than the required {samples_per_category}.")
        
        # Shuffle with a fixed seed and select the first 250 samples
        ds_cat_sampled = ds_cat.shuffle(seed=seed).select(range(samples_per_category))
    else:
        ds_cat_sampled = ds_cat
    
    sampled_datasets.append(ds_cat_sampled)

# 4. Concatenate all the individual category datasets into one
final_ds = concatenate_datasets(sampled_datasets)
total_samples_needed = len(final_ds)
print(f"Total samples in the final dataset: {total_samples_needed}")


# Optional: Shuffle the final combined dataset so categories aren't grouped together
final_ds = final_ds.shuffle(seed=seed)

print(f"\nFinal dataset size: {len(final_ds)} samples")
print(f"Categories included: {set(final_ds['category'])}")

# 5. Save the dataset to disk
output_dir = f"mmlu_pro_downsampled_{total_samples_needed}"
final_ds.save_to_disk(output_dir)
print(f"Dataset successfully saved to './{output_dir}'")

# from datasets import load_from_disk

#ds_downsampled = load_from_disk("mmlu_pro_downsampled_1000")