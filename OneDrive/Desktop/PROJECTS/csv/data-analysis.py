import csv
from tabulate import tabulate

# Read the CSV file with UTF-8 encoding
with open('olympics_data.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    # Initialize a counter and a list to store rows
    count = 0
    rows = []
    
    # Iterate through the rows and collect the first 10 rows
    for row in reader:
        if count < 5:
            rows.append(row)
            count += 1
        else:
            break

# Display the table
print(tabulate(rows, headers="keys", tablefmt="grid"))
        