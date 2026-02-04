import csv
from typing import List, Tuple

class CSVParser:
    def __init__(self, filename: str):
        """Initialize the parser with a CSV file name in the same directory."""
        self.filepath = os.path.join(os.path.dirname(__file__), filename)
        self.data = self._load_csv()
    
    def _load_csv(self) -> List[Tuple]:
        """Load CSV file and return data as list of tuples."""
        data = []
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    data.append(tuple(row))
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")
    
    def fetch_all(self) -> List[Tuple]:
        """Return all entries from the CSV file."""
        return self.data
    
    def first_n(self, n: int) -> List[Tuple]:
        """Return the first n entries."""
        return self.data[:n]
    
    def last_n(self, n: int) -> List[Tuple]:
        """Return the last n entries."""
        return self.data[-n:] if n > 0 else []
    
    def fetch_range(self, start: int, end: int) -> List[Tuple]:
        """Return entries between start and end indices."""
        return self.data[start:end]
    
    def fetch_by_index(self, index: int) -> Tuple:
        """Return a single entry by index."""
        return self.data[index]
    
    def count(self) -> int:
        """Return total number of entries."""
        return len(self.data)