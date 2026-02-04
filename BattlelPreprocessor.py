class DataAdapter:
    def __init__(self, data):
        """
        Initialize with a collection of tuples.
        
        Args:
            data: List of tuples containing raw data
        """
        self.data = data
        self.level_mapping = {
            "very low": 1,
            "low": 2,
            "medium": 3,
            "high": 4,
            "very high": 5
        }
    
    def extract_columns(self, column_indices):
        """
        Extract specific columns from the tuples.
        
        Args:
            column_indices: List of column indices to extract
            
        Returns:
            List of tuples with selected columns
        """
        return [tuple(row[i] for i in column_indices) for row in self.data]
    
    def extract_standard_columns(self):
        return self.extract_columns(list(range(26, len(self.data[0]))))
    
    def convert_strings_to_integers(self, column_index):
        """
        Convert string values to integers in a specific column.
        
        Args:
            column_index: Index of column to convert
            
        Returns:
            Modified data with converted column
        """
        converted = []
        for row in self.data:
            row_list = list(row)
            if isinstance(row_list[column_index], str):
                row_list[column_index] = int(row_list[column_index])
            converted.append(tuple(row_list))
        self.data = converted
        return self.data
    
    def map_level_values(self, column_index):
        """
        Map level strings to integer scale.
        
        Args:
            column_index: Index of column to map
            
        Returns:
            Modified data with mapped values
        """
        mapped = []
        for row in self.data:
            row_list = list(row)
            value = row_list[column_index].lower()
            row_list[column_index] = self.level_mapping.get(value, value)
            mapped.append(tuple(row_list))
        self.data = mapped
        return self.data
    
    def get_data(self):
        """Return the current data."""
        return self.data