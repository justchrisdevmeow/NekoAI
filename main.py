# full_chatbot.py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import random
import pickle
import os

# ============================================
# PART 1: TRANSFORMER HELPERS (for training only)
# ============================================

class TransformerHelpers:
    def __init__(self):
        print("Loading transformer helpers (one-time setup)...")
        # Feature extractor
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Reward model
        self.reward_pipe = pipeline("text-generation", model="distilgpt2")
        
        # Simple response database (will grow from training)
        self.response_db = []
        
    def get_embedding(self, text):
        """Convert text to embedding vector"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[0, 0, :].numpy()
        return embedding
    
    def score_response(self, user_input, bot_response):
        """Simple scoring based on relevance and length"""
        # Basic scoring (no complex models needed)
        score = 0.5  # base
        
        # Longer responses are slightly better (up to a point)
        if 5 < len(bot_response.split()) < 20:
            score += 0.2
        elif len(bot_response.split()) >= 20:
            score += 0.1
            
        # Check if response contains words from user input
        user_words = set(user_input.lower().split())
        bot_words = set(bot_response.lower().split())
        overlap = len(user_words & bot_words) / max(len(user_words), 1)
        score += overlap * 0.3
        
        return min(0.9, max(0.1, score))
    
    def generate_synthetic_response(self, user_input):
        """Generate a synthetic response for training"""
        prompt = f"User: {user_input}\nBot:"
        try:
            output = self.reward_pipe(prompt, max_length=50, do_sample=True, temperature=0.9)
            response = output[0]['generated_text'].replace(prompt, "").strip()
            # Clean up
            response = response.split("\n")[0][:100]
            if len(response) < 3:
                response = f"That's interesting about {user_input[:20]}"
        except:
            response = f"Tell me more about {user_input[:30]}"
        return response


# ============================================
# PART 2: YOUR AI FROM SCRATCH
# ============================================

class YourScratchAI:
    def __init__(self, embedding_dim=768, memory_size=100):
        # Neural network weights (from scratch)
        self.W_input = np.random.randn(embedding_dim, 256) * 0.01
        self.b_input = np.zeros(256)
        self.W_hidden = np.random.randn(256, 256) * 0.01
        self.b_hidden = np.zeros(256)
        self.W_output = np.random.randn(256, embedding_dim) * 0.01
        self.b_output = np.zeros(embedding_dim)
        
        # Learning
        self.lr = 0.005
        self.training_pairs = []  # stores (input_emb, target_emb, reward)
        
        # Memory
        self.memory = []  # last N exchanges
        self.memory_size = memory_size
        
        # Helpers
        self.helpers = TransformerHelpers()
        
        # Response cache (learned patterns)
        self.response_cache = {}  # embedding_hash -> response_text
        self.load_cache()
        
    def forward(self, x):
        """Forward pass through your scratch network"""
        # Layer 1
        z1 = np.dot(x, self.W_input) + self.b_input
        a1 = np.tanh(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.W_hidden) + self.b_hidden
        a2 = np.tanh(z2)
        
        # Output layer
        z3 = np.dot(a2, self.W_output) + self.b_output
        output = np.tanh(z3)  # bounded embedding
        
        return output, a2
    
    def backward(self, x, target, reward_weight=1.0):
        """Backpropagation from scratch with reward modulation"""
        # Forward pass to get activations
        output, hidden = self.forward(x)
        
        # Error (scaled by reward)
        error = (output - target) * reward_weight
        d_output = error * (1 - np.power(output, 2))  # tanh derivative
        
        # Output layer gradients
        dW_output = np.outer(hidden, d_output)
        db_output = d_output
        
        # Hidden layer gradients
        d_hidden = np.dot(d_output, self.W_output.T)
        d_hidden_act = d_hidden * (1 - np.power(hidden, 2))
        
        dW_hidden = np.outer(x, d_hidden_act)
        db_hidden = d_hidden_act
        
        # Input layer gradients
        d_input = np.dot(d_hidden_act, self.W_hidden.T)
        d_input_act = d_input * (1 - np.power(np.tanh(np.dot(x, self.W_input) + self.b_input), 2))
        
        dW_input = np.outer(x, d_input_act)
        db_input = d_input_act
        
        # Update weights (clipped to prevent explosion)
        self.W_output -= self.lr * np.clip(dW_output, -1, 1)
        self.b_output -= self.lr * np.clip(db_output, -1, 1)
        self.W_hidden -= self.lr * np.clip(dW_hidden, -1, 1)
        self.b_hidden -= self.lr * np.clip(db_hidden, -1, 1)
        self.W_input -= self.lr * np.clip(dW_input, -1, 1)
        self.b_input -= self.lr * np.clip(db_input, -1, 1)
        
        return np.mean(np.abs(error))
    
    def embedding_to_text(self, embedding):
        """Convert embedding to actual text using cache or generation"""
        # Round embedding for hashing
        emb_hash = tuple(np.round(embedding[:10], 3))
        
        # Check cache first
        if emb_hash in self.response_cache:
            return self.response_cache[emb_hash]
        
        # Find closest matching response from training
        if len(self.training_pairs) > 0:
            best_match = None
            best_dist = float('inf')
            
            for inp_emb, target_emb, resp_text in self.training_pairs[-100:]:
                dist = np.linalg.norm(embedding - target_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_match = resp_text
            
            if best_match and best_dist < 1.5:
                self.response_cache[emb_hash] = best_match
                return best_match
        
        # Generate a simple fallback response
        return self.generate_fallback_response()
    
    def generate_fallback_response(self):
        """Simple but working response generator"""
        templates = [
            "That's really interesting.",
            "I see what you mean.",
            "Tell me more about that.",
            "Hmm, I'm thinking about that.",
            "Thanks for sharing that.",
            "That makes sense to me.",
            "I hadn't thought of it that way."
        ]
        return random.choice(templates)
    
    def learn_from_pair(self, user_input, response_text, reward=1.0):
        """Learn from a user-bot exchange"""
        # Get embeddings
        input_emb = self.helpers.get_embedding(user_input)
        target_emb = self.helpers.get_embedding(response_text)
        
        # Store for future reference
        self.training_pairs.append((input_emb, target_emb, response_text))
        
        # Keep only recent training pairs
        if len(self.training_pairs) > 500:
            self.training_pairs = self.training_pairs[-500:]
        
        # Train the network
        loss = self.backward(input_emb, target_emb, reward)
        
        # Save response to cache
        emb_hash = tuple(np.round(target_emb[:10], 3))
        self.response_cache[emb_hash] = response_text
        
        self.save_cache()
        return loss
    
    def respond(self, user_input):
        """Generate a response using your scratch AI"""
        # Get embedding of user input
        user_emb = self.helpers.get_embedding(user_input)
        
        # Add context from memory (average of last 3 exchanges)
        if len(self.memory) > 0:
            context_embs = [m[0] for m in self.memory[-3:]]
            context_avg = np.mean(context_embs, axis=0)
            final_input = (user_emb + context_avg) / 2
        else:
            final_input = user_emb
        
        # Forward pass through your network
        predicted_emb, _ = self.forward(final_input)
        
        # Convert embedding to text
        response = self.embedding_to_text(predicted_emb)
        
        # If response is too short or generic, enhance it
        if len(response) < 15 or response in ["That's really interesting.", "I see what you mean."]:
            # Try to generate something more specific
            synthetic = self.helpers.generate_synthetic_response(user_input)
            if len(synthetic) > len(response):
                response = synthetic
        
        # Store in memory
        self.memory.append((user_emb, response, user_input))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        return response
    
    def pretrain_on_topics(self, topics, iterations_per_topic=5):
        """Pretrain using transformer-generated data"""
        print("\n🤖 Pretraining your AI with transformer help...")
        total_losses = []
        
        for topic in topics:
            for _ in range(iterations_per_topic):
                # Generate synthetic conversation
                synthetic_response = self.helpers.generate_synthetic_response(topic)
                
                # Learn from it with positive reward
                loss = self.learn_from_pair(topic, synthetic_response, reward=0.8)
                total_losses.append(loss)
        
        print(f"✅ Pretraining complete. Average loss: {np.mean(total_losses):.4f}")
        print(f"   Learned {len(self.response_cache)} response patterns\n")
    
    def save_cache(self):
        """Save learned responses to disk"""
        with open('bot_memory.pkl', 'wb') as f:
            pickle.dump({
                'cache': self.response_cache,
                'weights': {
                    'W_input': self.W_input,
                    'b_input': self.b_input,
                    'W_hidden': self.W_hidden,
                    'b_hidden': self.b_hidden,
                    'W_output': self.W_output,
                    'b_output': self.b_output
                }
            }, f)
    
    def load_cache(self):
        """Load learned responses from disk"""
        if os.path.exists('bot_memory.pkl'):
            with open('bot_memory.pkl', 'rb') as f:
                data = pickle.load(f)
                self.response_cache = data['cache']
                weights = data['weights']
                self.W_input = weights['W_input']
                self.b_input = weights['b_input']
                self.W_hidden = weights['W_hidden']
                self.b_hidden = weights['b_hidden']
                self.W_output = weights['W_output']
                self.b_output = weights['b_output']
                print(f"✅ Loaded {len(self.response_cache)} remembered responses")
    
    def chat(self):
        print("\n" + "="*50)
        print("🤖 YOUR AI FROM SCRATCH (Trained with Transformer Help)")
        print("="*50)
        print("\nI learn from every conversation. Rate responses 1-3 when asked.")
        print("Type 'quit' to exit and save my memory.\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nSaving my memory...")
                self.save_cache()
                print("Goodbye! I'll remember this conversation next time.")
                break
            
            # Generate response
            response = self.respond(user_input)
            print(f"\n🤖 AI: {response}")
            
            # Get feedback to learn
            print("\n[Rate this response: 1=bad, 2=ok, 3=good]")
            rating = input("Rating (or press Enter to skip): ").strip()
            
            if rating in ['1', '2', '3']:
                reward = (int(rating) - 2) * 0.5 + 0.5  # Convert to 0-1 range
                loss = self.learn_from_pair(user_input, response, reward)
                print(f"   ✓ Learned from that (loss: {loss:.4f})")
                self.save_cache()  # Save after each learning
            else:
                # Still learn but with neutral reward
                self.learn_from_pair(user_input, response, 0.5)
            
            print()


# ============================================
# RUN THE CHATBOT
# ============================================

if __name__ == "__main__":
    # Create your AI
    bot = YourScratchAI()
    
    # Pretrain on common topics (using transformer helper)
    pretraining_topics = [
        "hello how are you",
        "what is your name", 
        "tell me a joke",
        "what is the weather like",
        "I like programming",
        "tell me about yourself",
        "what can you do",
        "do you like music",
        "what is your favorite color",
        "how does AI work"
    ]
    
    bot.pretrain_on_topics(pretraining_topics, iterations_per_topic=3)
    
    # Start chatting
    bot.chat()