import pandas as pd
#categories
# information security - 1
# Ethics - 2
# Thread safe - 3
# C pointer based - 4
# Generic cs questions - 5
# Development tools and practices - 6
# Programming languages and syntax - 7
# Software design and architecture - 8
# Database and sql - 9
# Load the dataset
file_path = 'LLMResearch/dataset/Filtered_Merged_Questions_Answers.csv'
data = pd.read_csv(file_path)

# Define keywords for each category
keywords = {
    'information security': ['security', 'encryption', 'attack', 'breach', 'firewall', 'authentication', 'vulnerability', 'cybersecurity', 'privacy', 'data breach', 'network security', 'password', 'phishing', 'malware', 'cyber attack', 'data protection', 'cyber threat', 'cyber defense', 'cyber crime', 'cyber risk', 'cyber awareness'],
    'Ethics': ['ethics', 'moral', 'legal', 'privacy', 'responsibility', 'fairness', 'transparency', 'accountability', 'bias', 'conflict of interest', 'data ethics', 'ethical hacking', 'intellectual property', 'plagiarism', 'whistleblowing', 'professional ethics', 'social responsibility', 'trust', 'unbiased', 'informed consent'],
    'Thread safe': ['thread', 'concurrent', 'mutex', 'lock', 'race condition', 'synchronization', 'atomic', 'deadlock', 'semaphore', 'critical section', 'thread-safety', 'parallel', 'concurrency', 'shared resource', 'thread pool', 'thread-safe programming', 'reentrant', 'thread synchronization', 'thread-safety issues', 'thread-safety techniques'],
    'C pointer based': ['pointer', 'malloc', 'free', 'dereference', 'address', 'pointer arithmetic', 'null pointer', 'pointer to pointer', 'pointer casting', 'pointer array', 'pointer vs array', 'pointer to function', 'pointer to structure', 'pointer to const', 'pointer to void', 'pointer to member', 'pointer to array', 'pointer to constant', 'pointer to function in c', 'pointer to structure in c'],
    'Generic cs questions': ['algorithm', 'complexity', 'data structure', 'binary', 'sorting', 'recursion', 'graph', 'tree', 'hash table', 'linked list', 'stack', 'queue', 'heap', 'array', 'dynamic programming', 'searching', 'sorting algorithms', 'graph algorithms', 'tree algorithms', 'string algorithms'],
    'Development tools and practices': ['git', 'version control', 'debugging', 'testing', 'agile', 'continuous integration', 'continuous delivery', 'code review', 'refactoring', 'unit testing', 'integration testing', 'test-driven development', 'debugger', 'performance testing', 'load testing', 'static analysis', 'dependency management', 'build automation', 'issue tracking', 'agile methodologies', 'devops'],
    'Programming languages and syntax': ['syntax', 'compiler', 'interpreter', 'variable', 'function', 'class', 'object', 'inheritance', 'polymorphism', 'encapsulation', 'abstraction', 'data types', 'control structures', 'loops', 'conditionals', 'operators', 'arrays', 'strings', 'file handling', 'exception handling', 'modules'],
    'Software design and architecture': ['architecture', 'design pattern', 'MVC', 'singleton', 'interface', 'object-oriented design', 'component-based architecture', 'layered architecture', 'microservices', 'service-oriented architecture', 'model-view-controller', 'client-server architecture', 'event-driven architecture', 'domain-driven design', 'dependency injection', 'loose coupling', 'high cohesion', 'scalability', 'reusability', 'maintainability', 'modularity'],
    'Database and sql': ['SQL', 'database', 'query', 'table', 'join', 'index', 'transaction', 'constraint', 'normalization', 'stored procedure', 'view', 'trigger', 'relational database', 'database management system', 'data modeling', 'data manipulation language', 'data definition language', 'data control language', 'database design', 'database administration', 'database security', 'database optimization', 'database backup', 'database recovery', 'database replication', 'database schema', 'database query optimization']
}

# Function to categorize a question
def categorize_question(body):
    for category, kws in keywords.items():
        if any(kw.lower() in body.lower() for kw in kws):
            return list(keywords.keys()).index(category) + 1
    return 0  # Return 0 if no category matches

# Apply categorization
data['Category'] = data['Body_question'].apply(categorize_question)

# Save the new DataFrame to a new CSV file
training_dataset_path = 'LLMResearch/dataset/SimpleTrainingDataset.csv'
data.to_csv(training_dataset_path, index=False)

print(f"Training dataset with categories saved to {training_dataset_path}")
