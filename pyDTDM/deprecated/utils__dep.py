





def find_filename_with_number2(folder, target_number):
    files = glob.glob(f"{folder}/*")
    
    # Convert target_number to a string and create a regex pattern for it
    target_number_str = str(target_number)
    
    # Regex pattern to match the target number as a whole word, with optional decimal point
    pattern = rf'\b{re.escape(target_number_str)}\b'
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        
        # Check for exact match
        if re.search(pattern, file_name):
            return file_path
    
    return None  # Return None if no matching filename found    
    
def find_filename_with_number(folder, target_number):
    files = glob.glob(f"{folder}/*")
    # Regex pattern to match the target number, including numbers with a decimal point
    pattern = re.compile(r'(\d+(\.\d+)?)')
    
    for file_name in files:
        # Find all numbers in the filename
        matches = pattern.findall(file_name)
        for match in matches:
            number = float(match[0])  # Convert the matched number to float
            if number == target_number:
                return file_name
    return None


def find_filename_with_number(folder, target_number):
    files = glob.glob(f"{folder}/*")
    pattern = rf'\b{target_number}\b'  # Regex pattern to match the exact target number as a whole word

    for file_path in files:
        file_name = os.path.basename(file_path)
        if re.search(pattern, file_name):
            return file_path
    
    return None  # Return None if no matching filename found

def find_filename_with_number1(folder, target_number):
    files = glob.glob(f"{folder}/*")
    # Regex pattern to match the target number, including numbers with a decimal point
    pattern = re.compile(r'(\d+(\.\d+)?)')
    
    for file_name in files:
        # Find all numbers in the filename
        matches = pattern.findall(file_name)
        for match in matches:
            number = float(match[0])  # Convert the matched number to float
            if number == target_number:
                return file_name
    return None


def find_filename_with_number2(folder, target_number):
    files = glob.glob(f"{folder}/*")
    pattern = rf'\b{target_number}\b'  # Regex pattern to match the exact target number as a whole word

    for file_path in files:
        file_name = os.path.basename(file_path)
        if re.search(pattern, file_name):
            return file_path
    
    return None  # Return None if no matching filename found
