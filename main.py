# main.py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import random
import pickle
import os
import sys

# ============================================
# TRANSFORMER HELPERS
# ============================================

class TransformerHelpers:
    def __init__(self):
        print("Loading transformer helpers...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.generator = pipeline("text-generation", model="distilgpt2")
        
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0, 0, :].numpy()
    
    def score_response(self, user_input, bot_response):
        score = 0.5
        if 5 < len(bot_response.split()) < 20:
            score += 0.2
        user_words = set(user_input.lower().split())
        bot_words = set(bot_response.lower().split())
        overlap = len(user_words & bot_words) / max(len(user_words), 1)
        score += overlap * 0.3
        return min(0.9, max(0.1, score))
    
    def generate_synthetic_response(self, user_input):
        prompt = f"User: {user_input}\nBot:"
        try:
            output = self.generator(prompt, max_new_tokens=30, do_sample=True, temperature=0.9)
            response = output[0]['generated_text'].replace(prompt, "").strip()
            if len(response) < 3:
                response = f"That's interesting about {user_input[:20]}"
        except:
            response = f"Tell me more about {user_input[:30]}"
        return response


# ============================================
# YOUR AI FROM SCRATCH (FIXED SHAPES)
# ============================================

class YourScratchAI:
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=768):
        # Neural network weights (correct shapes)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b3 = np.zeros((1, output_dim))
        
        self.lr = 0.005
        self.training_pairs = []
        self.memory = []
        self.response_cache = {}
        
        self.helpers = TransformerHelpers()
        self.load_cache()
        
    def forward(self, x):
        # x shape: (1, 768) or (768,)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Layer 1: (1,768) @ (768,256) = (1,256)
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.tanh(z1)
        
        # Layer 2: (1,256) @ (256,256) = (1,256)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.tanh(z2)
        
        # Layer 3: (1,256) @ (256,768) = (1,768)
        z3 = np.dot(a2, self.W3) + self.b3
        output = np.tanh(z3)
        
        return output, a1, a2
    
    def backward(self, x, target, reward=1.0):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        
        # Forward pass
        output, a1, a2 = self.forward(x)
        
        # Error (scaled by reward)
        error = (output - target) * reward
        
        # Output layer gradient
        d_output = error * (1 - output**2)  # tanh derivative
        dW3 = np.dot(a2.T, d_output)
        db3 = np.sum(d_output, axis=0, keepdims=True)
        
        # Hidden layer 2 gradient
        d_a2 = np.dot(d_output, self.W3.T)
        d_a2_act = d_a2 * (1 - a2**2)
        dW2 = np.dot(a1.T, d_a2_act)
        db2 = np.sum(d_a2_act, axis=0, keepdims=True)
        
        # Hidden layer 1 gradient
        d_a1 = np.dot(d_a2_act, self.W2.T)
        d_a1_act = d_a1 * (1 - a1**2)
        dW1 = np.dot(x.T, d_a1_act)
        db1 = np.sum(d_a1_act, axis=0, keepdims=True)
        
        # Clip gradients to prevent explosion
        dW3 = np.clip(dW3, -1, 1)
        dW2 = np.clip(dW2, -1, 1)
        dW1 = np.clip(dW1, -1, 1)
        
        # Update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        return float(np.mean(np.abs(error)))
    
    def embedding_to_text(self, embedding):
        emb_hash = tuple(np.round(embedding[:10], 3))
        
        if emb_hash in self.response_cache:
            return self.response_cache[emb_hash]
        
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
        
        templates = [
            "That's interesting.",
            "I see what you mean.",
            "Tell me more.",
            "Hmm, interesting point.",
            "Thanks for sharing."
        ]
        return random.choice(templates)
    
    def learn_from_pair(self, user_input, response_text, reward=0.7):
        input_emb = self.helpers.get_embedding(user_input)
        target_emb = self.helpers.get_embedding(response_text)
        
        self.training_pairs.append((input_emb, target_emb, response_text))
        if len(self.training_pairs) > 500:
            self.training_pairs = self.training_pairs[-500:]
        
        loss = self.backward(input_emb, target_emb, reward)
        
        emb_hash = tuple(np.round(target_emb[:10], 3))
        self.response_cache[emb_hash] = response_text
        
        self.save_cache()
        return loss
    
    def respond(self, user_input):
        user_emb = self.helpers.get_embedding(user_input)
        
        if len(self.memory) > 0:
            context_embs = [m[0] for m in self.memory[-3:]]
            context_avg = np.mean(context_embs, axis=0)
            final_input = (user_emb + context_avg) / 2
        else:
            final_input = user_emb
        
        predicted_emb, _, _ = self.forward(final_input)
        response = self.embedding_to_text(predicted_emb.flatten())
        
        if len(response) < 15:
            synthetic = self.helpers.generate_synthetic_response(user_input)
            if len(synthetic) > len(response):
                response = synthetic
        
        self.memory.append((user_emb, response, user_input))
        if len(self.memory) > 50:
            self.memory.pop(0)
        
        return response
    
    def pretrain_on_topics(self, topics, iterations_per_topic=10):
        print("Pretraining...")
        total_losses = []
        for topic in topics:
            for _ in range(iterations_per_topic):
                synthetic = self.helpers.generate_synthetic_response(topic)
                loss = self.learn_from_pair(topic, synthetic, reward=0.8)
                total_losses.append(loss)
        print(f"Pretraining done. Avg loss: {np.mean(total_losses):.4f}")
    
    def save_cache(self):
        with open('bot_memory.pkl', 'wb') as f:
            pickle.dump({
                'cache': self.response_cache,
                'pairs': self.training_pairs[-200:],
                'W1': self.W1, 'b1': self.b1,
                'W2': self.W2, 'b2': self.b2,
                'W3': self.W3, 'b3': self.b3
            }, f)
    
    def load_cache(self):
        if os.path.exists('bot_memory.pkl'):
            with open('bot_memory.pkl', 'rb') as f:
                data = pickle.load(f)
                self.response_cache = data['cache']
                self.training_pairs = data['pairs']
                self.W1 = data['W1']
                self.b1 = data['b1']
                self.W2 = data['W2']
                self.b2 = data['b2']
                self.W3 = data['W3']
                self.b3 = data['b3']
                print(f"Loaded {len(self.response_cache)} cached responses")
    
    def chat(self):
        print("🤖 AI Ready. Type 'quit' to exit.\n")
        while True:
            user = input("You: ")
            if user.lower() == 'quit':
                self.save_cache()
                break
            response = self.respond(user)
            print(f"AI: {response}")
            print("\n[Rate 1-3 or Enter to skip]")
            rating = input("> ")
            if rating in ['1','2','3']:
                reward = (int(rating) - 2) * 0.5 + 0.5
                self.learn_from_pair(user, response, reward)
                print("✓ Learned")


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    bot = YourScratchAI()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        topics = ['hello', 'hi', 'how are you', 'what is your name', 'tell me a joke',
                  'weather', 'sports', 'movies', 'music', 'coding', 'python', 'ai']
        bot.pretrain_on_topics(topics, iterations_per_topic=15)
        bot.save_cache()
        print("✅ Training complete. Model saved.")
    else:
        bot.chat()
