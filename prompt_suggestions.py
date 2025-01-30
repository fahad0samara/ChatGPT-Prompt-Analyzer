import random

class PromptSuggester:
    def __init__(self):
        self.templates = {
            'analysis': [
                "Analyze {subject} and identify {aspect}",
                "Can you analyze the {aspect} of {subject}?",
                "Please analyze {subject} in terms of {aspect}",
                "Provide a detailed analysis of {subject} focusing on {aspect}",
                "Examine {subject} and evaluate its {aspect}"
            ],
            'teaching': [
                "Explain {subject} in simple terms",
                "Teach me about {subject} with examples",
                "Help me understand {subject} step by step",
                "Can you explain how {subject} works?",
                "Guide me through the basics of {subject}"
            ],
            'guidance': [
                "Help me with {subject}",
                "Guide me through {subject}",
                "Assist me in {subject}",
                "Can you help me solve {subject}?",
                "I need help with {subject}"
            ],
            'creation': [
                "Create a {subject} that {aspect}",
                "Generate a {subject} with {aspect}",
                "Build a {subject} that can {aspect}",
                "Develop a {subject} for {aspect}",
                "Write a {subject} that implements {aspect}"
            ]
        }
        
        self.subjects = {
            'analysis': [
                "dataset", "algorithm performance", "system architecture",
                "code quality", "security vulnerabilities", "user behavior",
                "market trends", "test results", "resource usage",
                "application performance"
            ],
            'teaching': [
                "Python programming", "machine learning", "data structures",
                "algorithms", "web development", "database design",
                "software architecture", "API development", "testing strategies",
                "DevOps practices"
            ],
            'guidance': [
                "debugging issues", "optimizing code", "implementing features",
                "solving errors", "improving performance", "fixing bugs",
                "setting up development environment", "deploying applications",
                "managing dependencies", "version control"
            ],
            'creation': [
                "web application", "API endpoint", "database schema",
                "machine learning model", "user interface", "automation script",
                "testing framework", "monitoring system", "deployment pipeline",
                "documentation"
            ]
        }
        
        self.aspects = {
            'analysis': [
                "performance bottlenecks", "efficiency improvements",
                "potential optimizations", "security risks",
                "quality issues", "scalability concerns",
                "reliability metrics", "usage patterns",
                "resource consumption", "error patterns"
            ],
            'teaching': [
                "practical examples", "real-world applications",
                "common pitfalls", "best practices",
                "fundamental concepts", "advanced techniques",
                "implementation details", "design patterns",
                "optimization strategies", "debugging approaches"
            ],
            'guidance': [
                "step-by-step instructions", "best practices",
                "efficient solutions", "proper implementation",
                "optimal approaches", "practical solutions",
                "common issues", "potential problems",
                "performance considerations", "security concerns"
            ],
            'creation': [
                "handles user authentication", "processes data efficiently",
                "implements security best practices", "scales horizontally",
                "supports real-time updates", "maintains data consistency",
                "provides detailed logging", "includes comprehensive tests",
                "follows design patterns", "uses modern practices"
            ]
        }
    
    def generate_prompt(self, category):
        """Generate a random prompt for the specified category"""
        template = random.choice(self.templates[category])
        subject = random.choice(self.subjects[category])
        aspect = random.choice(self.aspects[category])
        
        return template.format(subject=subject, aspect=aspect)
    
    def generate_examples(self, category, count=5):
        """Generate multiple example prompts for a category"""
        return [self.generate_prompt(category) for _ in range(count)]
    
    def suggest_improvements(self, prompt, category):
        """Suggest improvements for a given prompt"""
        suggestions = []
        
        # Add category-specific elements
        if category == 'analysis':
            if 'analyze' not in prompt.lower():
                suggestions.append("Start with 'Analyze' to make the intent clear")
            if 'and' not in prompt.lower():
                suggestions.append("Add specific aspects to analyze using 'and'")
                
        elif category == 'teaching':
            if not any(word in prompt.lower() for word in ['explain', 'teach', 'understand']):
                suggestions.append("Use teaching-focused verbs like 'explain', 'teach', or 'help understand'")
            if 'example' not in prompt.lower():
                suggestions.append("Ask for examples to make learning more concrete")
                
        elif category == 'guidance':
            if not any(word in prompt.lower() for word in ['help', 'guide', 'assist']):
                suggestions.append("Start with 'Help', 'Guide', or 'Assist' to indicate need for guidance")
            if 'how' not in prompt.lower():
                suggestions.append("Consider adding 'how' to make the request more specific")
                
        elif category == 'creation':
            if not any(word in prompt.lower() for word in ['create', 'generate', 'build', 'develop']):
                suggestions.append("Use creation-focused verbs like 'create', 'generate', or 'build'")
            if 'with' not in prompt.lower():
                suggestions.append("Specify requirements using 'with' or 'that'")
        
        # General improvements
        if len(prompt.split()) < 5:
            suggestions.append("Add more detail to make the prompt more specific")
        if '?' not in prompt:
            suggestions.append("Consider phrasing as a question for better engagement")
        if not any(char in prompt for char in '.,?!'):
            suggestions.append("Add proper punctuation")
        
        return suggestions

# Example usage
if __name__ == "__main__":
    suggester = PromptSuggester()
    
    # Generate example prompts
    print("Example Analysis Prompts:")
    for prompt in suggester.generate_examples('analysis', 3):
        print(f"- {prompt}")
    
    print("\nExample Teaching Prompts:")
    for prompt in suggester.generate_examples('teaching', 3):
        print(f"- {prompt}")
    
    # Get improvement suggestions
    test_prompt = "analyze code"
    print(f"\nImprovement suggestions for '{test_prompt}':")
    for suggestion in suggester.suggest_improvements(test_prompt, 'analysis'):
        print(f"- {suggestion}")
