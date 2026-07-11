eval_set = [
    {
        "question": "What is backtracking search?",
        "expected_tool": "search_notes",
        "expected_answer_contains": "backtrack"
    },
    {
        "question": "What is a zero-sum game?",
        "expected_tool": "search_notes",
        "expected_answer_contains": "lose"
    },
    {
        "question": "What is arc consistency?",
        "expected_tool": "search_notes",
        "expected_answer_contains": "arc"
    },
    {
        "question": "How many FIFA World Cup matches are in the database?",
        "expected_tool": "query_database",
        "expected_answer_contains": "1064"
    },
    {
        "question": "Which team has played the most FIFA World Cup matches?",
        "expected_tool": "query_database",
        "expected_answer_contains": "Germany"  # adjust if output differs
    },
    {
        "question": "How many matches did Brazil play in the FIFA World Cup?",
        "expected_tool": "query_database",
        "expected_answer_contains": "Brazil"
    },
    {
        "question": "What is 17 times 13?",
        "expected_tool": "direct",
        "expected_answer_contains": "221"
    },
    {
        "question": "What is the capital of France?",
        "expected_tool": "direct",
        "expected_answer_contains": "Paris"
    },
]