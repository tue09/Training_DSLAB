from collections import defaultdict

doc_count = defaultdict(int)
doc_count['a'] += 1
doc_count['b'] += 5
doc_count['5'] -= 4
doc_count['@'] += 2
doc_count['abc'] += 5
doc_count['abc'] -= 2
print(doc_count)  # In ra: 0
