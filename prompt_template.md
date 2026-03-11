# Prompt Templates

## Triple Extraction Prompt Template

**Extract movie preferences from this conversation as triples.**

**CRITICAL RULES:**
1. ONLY extract if the user EXPLICITLY states their preference
2. ONLY extract specific movie titles - NO genres, actors, or franchises
3. Subject must ALWAYS be "User" (not "You")
4. Use lowercase for relations: likes, dislikes, seen, notSeen, suggested
5. Output ONLY triples in the exact format shown below
6. If no triples can be extracted, output: "No triples found."

**SCHEMA:**

Format: (User, relation, Movie Title (Year))

Relations:
- likes: User explicitly says they like/love/enjoy a specific movie
- dislikes: User explicitly says they dislike/hate a specific movie
- seen: User explicitly states they watched a specific movie
- notSeen: User explicitly states they have NOT watched a specific movie
- suggested: Assistant recommends a specific movie to the user

{K-SHOT_EXAMPLES}



**NEW CONVERSATION TO EXTRACT:**

{CONVERSATION}

