import csv
import json

# Create and write to the CSV file
csv_file_path = './COCO_captions.csv'

with open('MSCOCO_train_val_Korean.json', 'r') as f:
    data = json.load(f)

val_csv_file_path = 'COCO_val.csv'
train_csv_file_path = 'COCO_train.csv'

# Open both CSV files at once
with open(val_csv_file_path, mode='w', newline='', encoding='utf-8') as val_file, \
     open(train_csv_file_path, mode='w', newline='', encoding='utf-8') as train_file:
    
    # Create CSV writers for both files
    val_writer = csv.writer(val_file)
    train_writer = csv.writer(train_file)
    
    # Write the headers for both files
    val_writer.writerow(["id", "file_path", "captions", "caption_ko"])
    train_writer.writerow(["id", "file_path", "captions", "caption_ko"])
    
    # Process the data in a single loop
    for entry in data:
        for caption, caption_ko in zip(entry['captions'], entry['caption_ko']):
            if caption_ko.strip() != '':  # Skip rows with empty caption_ko
                if "val2014" in entry["file_path"]:
                    val_writer.writerow([entry['id'], entry['file_path'], caption, caption_ko])
                elif "train2014" in entry["file_path"]:
                    train_writer.writerow([entry['id'], entry['file_path'], caption, caption_ko])

