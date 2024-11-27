from chat_solution.embedding_db import EmbeddingDatabase
from chat_solution.llm import LargeLanguageModel

class CookingAssitant:
    _instance = None

    def get_instance():
        if not CookingAssitant._instance:
            CookingAssitant._instance = CookingAssitant()
        return CookingAssitant._instance
    
    def __init__(self):
        self.embedding_db = EmbeddingDatabase()  # Remove the embedding_model argument
        self.llm = LargeLanguageModel()
        self.conversation_history = []
        self.documents_retrieved = []
        self.instructions = """You are a helpful AI cooking assistant.
The user asks you for cooking recipes, giving you eventually a list of ingredients available or season of the year and you should give recipes recommendations based on the context.
- Generate a new recipe  based on the context provided. The ingredients do not need to be exactly the same, small variations are possible unless specified different by the user, but should always be related to the context.
- Your primary audience are people who want to cook but don't have much cooking experience or want to save time not having to think of good food combinations.
- The answer should be found within the given context. If not say: "My database is hungry of more recipes."
- Provide 2 different options of recipes, if possible, being one of them a vegetarian option.
- If the user asks for a specific recipe, provide the recipe with the ingredients and the steps to cook it.
- If the user asks for (a) specific ingredient(s), provide a recipe that includes that ingredient(s).
- If the user asks for a specific season, provide a recipe that is typical for that season.
- If the user asks for a specific cuisine, provide a recipe that is typical for that cuisine.
- If the user asks for a specific type of dish, provide a recipe that is typical for that type of dish.
- If the user asks for a specific diet, provide a recipe that is typical for that diet.
- If the user asks for a specific cooking time, provide a recipe that can be cooked in that time.
- If the user asks for a specific cooking method, provide a recipe that uses that method.
- If the user asks for a specific occasion, provide a recipe that is typical for that occasion.
- If the user asks for a specific meal, provide a recipe that is typical for that meal.
- If the user asks for a specific flavor, provide a recipe that has that flavor.
- If the user asks for a specific texture, provide a recipe that has that texture.
- If the user asks for a specific temperature, provide a recipe that is cooked at that temperature.
- If the user asks for a specific color, provide a recipe that has that color.
"""

        self.examples = """<startexample>
    Interaction 1
    New Context: The CookingAssistant is an AI trained to help users with cooking recipes. It provides recommendations based on user input such as ingredients, seasons, cuisines, or dietary preferences. If the requested recipe cannot be found in its database, it will inform the user.
    User input: I have tomatoes, pasta, and basil. Can you suggest something I can cook?
    Assistant:
    Sure! Here are two recipes you can try:

    Recipe 1: Spaghetti Pomodoro (Vegetarian)
    Ingredients:
    - 200g spaghetti
    - 3 medium tomatoes (chopped)
    - 2 garlic cloves (minced)
    - 2 tbsp olive oil
    - Salt and pepper to taste
    - Fresh basil leaves (chopped)
    Steps:
    1. Boil the spaghetti in salted water until al dente, then drain.
    2. Heat olive oil in a pan, add garlic, and sauté until fragrant.
    3. Add chopped tomatoes, salt, and pepper, and cook until the tomatoes soften.
    4. Toss the spaghetti with the tomato sauce.
    5. Serve with fresh basil leaves on top.

    Recipe 2: Tomato Basil Salad (Vegetarian)
    Ingredients:
    - 4 medium tomatoes (sliced)
    - 100g fresh mozzarella (optional, or substitute with vegan cheese)
    - Fresh basil leaves
    - 2 tbsp olive oil
    - 1 tbsp balsamic vinegar
    - Salt and pepper to taste
    Steps:
    1. Arrange the sliced tomatoes on a plate.
    2. Add fresh mozzarella slices and basil leaves on top.
    3. Drizzle with olive oil and balsamic vinegar.
    4. Sprinkle with salt and pepper.
    
    Interaction 2
    User input: Can you suggest something for winter using potatoes?
    Assistant:
    Sure! Here are two cozy winter recipes featuring potatoes:

    Recipe 1: Potato Leek Soup (Vegetarian)
    Ingredients:
    - 4 medium potatoes (peeled and diced)
    - 2 leeks (sliced, white and light green parts only)
    - 2 tbsp butter or olive oil
    - 4 cups vegetable broth
    - 1 cup heavy cream (or plant-based cream)
    - Salt and pepper to taste
    Steps:
    1. In a large pot, heat butter or olive oil and sauté the leeks until soft.
    2. Add the diced potatoes and vegetable broth. Bring to a boil, then reduce heat and simmer until potatoes are tender.
    3. Blend the soup until smooth using an immersion blender or a standard blender.
    4. Stir in the cream, and season with salt and pepper.
    5. Serve hot with crusty bread.

    Recipe 2: Roasted Garlic Potatoes
    Ingredients:
    - 500g small potatoes (halved)
    - 3 garlic cloves (minced)
    - 3 tbsp olive oil
    - 1 tsp rosemary (optional)
    - Salt and pepper to taste
    Steps:
    1. Preheat the oven to 200°C (400°F).
    2. Toss the potatoes with olive oil, garlic, rosemary, salt, and pepper.
    3. Spread the potatoes in a single layer on a baking tray.
    4. Roast for 25-30 minutes, stirring occasionally, until golden and crispy.
    5. Serve as a side dish or snack.
    </endexample>
    """
    
    def query(self, query: str) -> str:
        """
        Process user query to retrieve recipes and generate a response.
        """
        documents = self.embedding_db.retrieve(query)  # Retrieve recipes based on the query.
        self.documents_retrieved = documents

        # Generate the response based on the documents and query.
        prompt = self._get_prompt(documents, query)
        response = self.llm.call(prompt)  # Use the LLM to generate the assistant's response.
        self.conversation_history.append((query, response))

        # If no documents are found, include a fallback response.
        if not documents:
            response += "\n\nMy database is hungry for more recipes. Can you provide additional context or ingredients?"

        return response

    def _get_prompt(self, documents: str, query: str) -> str:
        """
        Construct the LLM prompt based on the conversation history, retrieved documents, and user query.
        """
        # Build the conversation history.
        chat_history = ""
        for i, (old_query, response) in enumerate(self.conversation_history, start=1):
            chat_history += f"Interaction {i}\nUser: {old_query}\nAssistant: {response}\n"

        # Add new context from retrieved documents.
        new_context_str = f"\nNew Context: {documents}" if documents else "\nNew Context: No additional recipes found."

        # Complete prompt for the LLM.
        self.complete_prompt = f"""{self.instructions}
{self.examples}
Now we start the conversation history:
{chat_history}

Just predict the next answer:
Interaction {len(self.conversation_history) + 1} {new_context_str}
User input: {query}
Assistant:"""
    
        return self.complete_prompt
