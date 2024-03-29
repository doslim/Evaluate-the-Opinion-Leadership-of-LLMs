import regex

def extract_json_strings(input_string):
    json_pattern = regex.compile(r'(\{(?:[^{}]|(?1))*\}|\[(?:[^\[\]]|(?1))*\])')

    matches = regex.findall(json_pattern, input_string)

    return matches

