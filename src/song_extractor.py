import pandas as pd

# Define the input files
file_paths = [
    '../data/raw/features/metadata/metadata_2013.csv',
    '../data/raw/features/metadata/metadata_2014.csv',
    '../data/raw/features/metadata/metadata_2015.csv'
]

def clean_song_id(song_id):
    """Cleans the song_id by removing non-numeric characters and ensuring it is within a valid range."""
    try:
        # Strip leading/trailing spaces and remove non-numeric characters
        song_id_cleaned = ''.join(c for c in str(song_id).strip() if c.isdigit())  # Keep only digits
        
        # Convert to numeric and check if valid
        song_id_cleaned = pd.to_numeric(song_id_cleaned, errors='coerce')
        
        if pd.isna(song_id_cleaned):
            return None  # Return None if the ID is invalid
        
        return int(song_id_cleaned)  # Return the cleaned song_id as an integer
    except Exception as e:
        return None

def process_line_by_line(file_path, file_type):
    """Process each file line by line to extract song data based on its format."""
    song_data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            columns = line.strip().split(',')

            if file_type == 'metadata_2013':
                # For metadata_2013.csv
                if len(columns) >= 4:  # Ensure there are at least 4 columns
                    song_id = columns[0]
                    artist = columns[2]
                    title = columns[3]
                else:
                    continue  # Skip malformed rows
            elif file_type == 'metadata_2014':
                # For metadata_2014.csv
                if len(columns) >= 4:  # Ensure there are at least 4 columns
                    song_id = columns[0]
                    artist = columns[1]
                    title = columns[3]
                else:
                    continue  # Skip malformed rows
            elif file_type == 'metadata_2015':
                # For metadata_2015.csv
                if len(columns) >= 4:  # Ensure there are at least 4 columns
                    song_id = columns[0]
                    artist = columns[3]
                    title = columns[2]
                else:
                    continue  # Skip malformed rows
            else:
                continue  # Skip invalid lines
            
            # Clean song_id and skip invalid rows
            song_id_cleaned = clean_song_id(song_id)
            if song_id_cleaned is not None:
                song_data.append([song_id_cleaned, title, artist])
    
    return song_data

# Process each file and collect song data
all_song_data = []

# Process 2013 file
all_song_data.extend(process_line_by_line('../data/raw/features/metadata/metadata_2013.csv', 'metadata_2013'))

# Process 2014 file
all_song_data.extend(process_line_by_line('../data/raw/features/metadata/metadata_2014.csv', 'metadata_2014'))

# Process 2015 file
all_song_data.extend(process_line_by_line('../data/raw/features/metadata/metadata_2015.csv', 'metadata_2015'))

# Convert the list of song data into a DataFrame
all_data = pd.DataFrame(all_song_data, columns=['song_id', 'title', 'artist'])

# Clean text columns (strip leading/trailing spaces and remove invisible characters)
all_data['title'] = all_data['title'].astype(str).str.replace(r'\t', '', regex=True).str.strip()
all_data['artist'] = all_data['artist'].astype(str).str.replace(r'\t', '', regex=True).str.strip()

# Replace empty strings with NaN
all_data['song_id'] = all_data['song_id'].replace(r'^\s*$', pd.NA, regex=True)
all_data['title'] = all_data['title'].replace(r'^\s*$', pd.NA, regex=True)
all_data['artist'] = all_data['artist'].replace(r'^\s*$', pd.NA, regex=True)

# Drop rows with missing critical info
all_data = all_data.dropna(subset=['song_id', 'title', 'artist'])

# Save to CSV
output_path = '../data/raw/features/metadata/all_songs.csv'
all_data.to_csv(output_path, index=False)

output_path
