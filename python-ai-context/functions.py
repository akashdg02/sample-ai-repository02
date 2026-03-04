def normalize_data( data, method='min-max'):
    """Normailze data for ML Models"""
    if method == 'min-max':
        min_val = min(data)
        max_val = max(data)
        return [(x-min_val) / (max_val - min_val) for x in data]
    elif method == 'z-score':
        mean = sum(data) / len(data)
        variance = sum((x-mean) ** 2 for x in data) / len(data)
        std = variance ** 0.5
        return [(x-mean)/std for x in data]
    
raw_data = [10,20,30,40,50]

normalized = normalize_data(raw_data)

print(f"Nornmalized: {normalized}")